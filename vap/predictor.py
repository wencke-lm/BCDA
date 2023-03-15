"""predictor.py - Transformer based sequence classifier"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vap.utils import get_activity_history, decimal_to_binary_tensor


class StaticPositionEmbedding(nn.Module):
    """Add information about the position to sequence tokens."""
    def __init__(self, dim, max_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2) * (-math.log(1e4) / dim)
        )
        pe = torch.zeros(max_len, 1, dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Transformer(nn.Module):
    """Causal decoder-only transformer for sequence encoding.

    Args:
        dim (int):
            Number of expected input features.
        ffn_dim (int):
            Dimension of the feedforward model.
        n_heads (int):
            Number of heads in multihead attention model.
        n_layers (int):
            Number of transformer layers.
        activation (str):
            Activation function of the feedforward model.
        dropout (float):
            Dropout applied during training.
        max_len (int):
            Maximum sequence length.

    """
    def __init__(
        self, dim, ffn_dim, n_heads, n_layers, activation, dropout, max_len
    ):
        super().__init__()

        self.pos_encoder = StaticPositionEmbedding(
            dim,
            max_len,
            dropout
        )
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            n_layers
        )

    def forward(self, x):
        """Feed into causal decoder-only transformer.

        Args:
            x (torch.tensor):
                (B_Batch, L_Strides, F_Features)

        Returns:
            (torch.tensor):
                (B_Batch, L_Strides, F_Features)

        """
        x = self.pos_encoder(x)

        # for each token mask all token to its right
        seq_len = x.shape[1]
        encoder_mask = (
            torch.triu(
                torch.ones((seq_len, seq_len), device=x.device),
                diagonal=1
            ) == 1
        )
        x = self.transformer(x, mask=encoder_mask)

        assert not x.isnan().any()

        return x


class VAPHead(nn.Module):
    """Voice activity classification head.

    Predicts the probability of future time windows
    being active, where windows are modelled in
    dependence of each other.

    Args:
        dim_in (int):
            Number of input features.
        pred_bins (list[float]):
            Relative size of each bin over
            the whole prediction window.
        threshold (float):
            Threshold determining when a bin is considered
            active. Must be value between 0.0 and 1.0.

    """
    def __init__(self, dim_in, pred_bins, threshold):
        super().__init__()

        # discrete bin activation prediction
        self.pred_bins = pred_bins
        self.threshold = threshold

        self.n_classes = 2**(2*len(pred_bins) - 2)
        self.projection_head = nn.Linear(
            dim_in, self.n_classes
        )

    def extract_gold_labels(self, activity):
        """Get index encoded gold activity labels.

        The bin configuration is treated as a binary number
        and encoded by transforming it to a decimal.

        Args:
            activity (torch.tensor):
                Binary notion of whether a speaker is
                speaking at a point in time.
                (B_Batch, N_Speakers, L_Strides)

        Returns:
            torch.tensor

        """
        n_batch, n_speakers, n_frames = activity.shape
        hist = torch.ones(
            n_batch, n_speakers, n_frames, len(self.pred_bins)-1,
            device=activity.device
        ) * -1

        # else it is the sum of all values inside an interval sized moving window
        for i, (start, end) in enumerate(zip(self.pred_bins, self.pred_bins[1:])):
            # size of the interval/window
            ws = start - end

            # padding left so that every frame is once at the right edge of window
            # so we have as many individual windows as frames
            va = F.pad(activity, [ws - 1, 0])

            # skip last frames not part of the history of the current frame
            if end > 0:
                va = va[:, :, :-end]

            # implicit loop over all windows
            filters = torch.ones((2, 1, ws), dtype=va.dtype, device=va.device)
            window_out = F.conv1d(va, weight=filters, groups=2)
            hist[:, :, -window_out.shape[-1]:, i] = window_out

        assert torch.all(hist >= -1)

        bin_activity = hist[:, :, self.pred_bins[0]:]

        # get activity of bin in percentage
        bin_activity /= torch.diff(
            torch.tensor(self.pred_bins, device=activity.device)
        ).abs()

        # apply a threshold to determine active bins
        bin_config = torch.where(
            bin_activity >= self.threshold, 1, 0
        ).permute((0, 2, 1, 3)).flatten(start_dim=2)

        # translate binary active bin config to decimal value
        powers_of_two = torch.pow(
            2, torch.arange(
                start=2*len(self.pred_bins)-3, end=-1, step=-1,
                device=activity.device
            )
        )
        labels = (bin_config*powers_of_two).sum(dim=-1)

        return labels

    def get_next_speaker(self, raw_pred):
        """Get speaker who will continue speaking.
        The next speaker is determined as the speaker,
        that is predicted to be the only active speaker
        in the prediction window and speaks at least
        for the last two bins.
        Args:
            raw_pred (torch.tensor):
                (, C_Classes)
                Unaltered output of the forward pass.
                Single sample, no batch.
        Returns:
            torch.tensor: (*, 1)
        """
        prob_speakers = torch.zeros(
            2, device=raw_pred.device
        )

        for i, prob in enumerate(raw_pred):
            bin_conf = decimal_to_binary_tensor(
                i, int(math.log2(self.n_classes))
            )
            bin_conf_by_speaker = bin_conf.unfold(
                0, len(self.pred_bins)-1, len(self.pred_bins)-1
            )

            # we are only interested in states with single active speaker
            if torch.count_nonzero(bin_conf_by_speaker.sum(dim=-1)) == 1:
                active_speaker = int(torch.nonzero(
                    bin_conf_by_speaker.sum(dim=-1)
                ))
                # states should also have much speech to the end of window
                total_va = 0
                for i, va in enumerate(
                    bin_conf_by_speaker[active_speaker].flip(dims=[0]), 1
                ):
                    if not va:
                        break
                    total_va += torch.tensor(self.pred_bins).diff().abs()[-i]
                # print(total_va)
                if total_va >= self.pred_bins[0]/2:
                    prob_speakers[active_speaker] += prob

        return int(torch.argmax(prob_speakers))

    def is_backchannel(self, raw_pred, speaker=None):
        """Determine whether a backchannel is upcoming.
        Args:
            raw_pred (torch.tensor):
                (, C_Classes)
                Unaltered output of the forward pass.
                Single sample, no batch.
            speaker (int):
                Index of the speaker that is considered
                to currently hold the floor, i.e. had
                long speeach activity preceding the
                prediction window.
        Returns:
            bool:
                True, if a backchannel is seen as likely.
                False else.
        """
        prob_no_bc_bc = torch.zeros(
            2, device=raw_pred.device
        )
        count_states_no_bc_bc = torch.zeros(
            2, device=raw_pred.device
        )


        for i, prob in enumerate(raw_pred):
            bin_conf = decimal_to_binary_tensor(
                i, int(math.log2(self.n_classes))
            )
            bin_conf_by_speaker = bin_conf.unfold(
                0, len(self.pred_bins)-1, len(self.pred_bins)-1
            )
            active_speaker = speaker
            total_va = 0

            # va is an array of size equal to number of speakers
            # starting from their activity at the end of window
            for i, va in enumerate(
                bin_conf_by_speaker.permute(1, 0).flip(dims=[0]), 1
            ):
                # active speaker should have much activity towards the end
                if total_va < self.pred_bins[0]*0.4:
                    if va.sum() == 1:
                        if (
                            active_speaker is None
                            or active_speaker == int(torch.nonzero(va))
                        ):
                            active_speaker = int(torch.nonzero(va))
                            total_va += torch.tensor(
                                self.pred_bins
                            ).diff().abs()[-i]
                            continue

                    # without it, we can no longer consider the config a BC
                    total_va = -float("inf")
                # non-active speaker should have short activity towards the start
                else:
                    all_speakers = torch.nonzero(va)

                    if (
                        len(all_speakers) > 1
                        or (
                            len(all_speakers) == 1
                            and active_speaker not in all_speakers
                        )
                    ):
                        prob_no_bc_bc[1] += prob
                        count_states_no_bc_bc[1] += 1
                        break
            else:
                prob_no_bc_bc[0] += prob
                count_states_no_bc_bc[0] += 1

        return bool(torch.argmax(prob_no_bc_bc/count_states_no_bc_bc))

    def forward(self, x):
        """Feed into voice activity classification head.

        Args:
            x (torch.tensor): (*, F_Features)
                Where * means any number of dimensions and
                the last dimension holds the input features.

        Returns:
            torch.tensor: (*, C_Classes)

        """
        return self.projection_head(x)


class Predictor(nn.Module):
    """Transformer based sequence classification.

    Args:
        See Docstrings of Transformer, VAPHead.

    """
    def __init__(
        self,
        dim,
        ffn_dim,
        n_heads,
        n_layers,
        activation,
        dropout,
        max_len,
        pred_bins,
        threshold
    ):
        super().__init__()

        self.transformer = Transformer(
            dim, ffn_dim, n_heads, n_layers, activation, dropout, max_len
        )
        self.classification_head = VAPHead(dim, pred_bins, threshold)

    def forward(self, x):
        """Apply sequence classification.

        Args:
            x (torch.tensor):
                (B_Batch, L_Strides, F_Features)

        Returns:
            torch.tensor: (B_Batch, L_Strides, C_Classes)

        """
        x = self.transformer(x)
        x = self.classification_head(x)

        return x
