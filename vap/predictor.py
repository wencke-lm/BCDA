"""predictor.py - Transformer based sequence classifier"""
from functools import reduce
import math

import torch
import torch.nn as nn


class StaticPositionEmbedding(nn.Module):
    """Add information about the position of sequence tokens."""
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
                torch.ones((seq_len, seq_len), device=x.device), diagonal=1
            ) == 1
        )
        x = self.transformer(x, mask=encoder_mask)

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

        if sum(pred_bins) != 1:
            raise ValueError(
                "Values in pred_bins must sum to 1."
            )

        # discrete bin activation prediction
        self.pred_bins = pred_bins
        self.threshold = threshold

        self.n_classes = 2 ** (2*len(pred_bins))
        self.projection_head = nn.Linear(
            dim_in, self.n_classes
        )
        self.softmax = nn.Softmax(dim=-1)

    def get_gold_label(self, activity):
        """Get one-hot encoded gold activity labels.

        Args:
            activity (torch.tensor):
                Binary notion of whether a speaker is
                speaking at a point in time.
                (B_Batch, N_Speakers, L_Strides)

        Returns:
            torch.tensor: (B_Batch, C_Classes)

        """
        n_samples, _, n_strides = activity.shape
        start = 0
        all_va = []

        # iterate over all bins and determine activity status
        for size in self.pred_bins:
            step = round(size*n_strides)

            # a bin is considered active, if ratio of active frames
            # exceed a given threshold
            va_ratio = activity[:, :, start:start+step].sum(-1) / step
            va = (va_ratio >= self.threshold)
            all_va.append(va)

            start += step
        all_va = torch.stack(all_va, dim=2).flatten(1)

        # convert active bin configurations to class indices
        def binary_tensor_to_decimal(t):
            return int(reduce(lambda x, y: x + str(int(y)), t, ''), 2)

        one_hot_label = torch.zeros(
            (n_samples, self.n_classes), device=activity.device
        )

        for i in range(n_samples):
            idx = binary_tensor_to_decimal(all_va[i])
            one_hot_label[i, idx] = 1

        return one_hot_label

    def forward(self, x):
        """Feed into voice activity classification head.

        Args:
            x (torch.tensor): (*, F_Features)
                Where * means any number of dimensions and
                the last dimension holds the input features.

        Returns:
            torch.tensor: (*, C_Classes)

        """
        return self.softmax(self.projection_head(x))


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
            torch.tensor: (B_Batch, C_Classes)

        """
        # last hidden state of last sequence token used
        # TODO: may instead use max- or average-pooling
        x = self.transformer(x)[:, -1, :]
        x = self.classification_head(x)

        return x
