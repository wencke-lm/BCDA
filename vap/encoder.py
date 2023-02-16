"""encoder.py - Speech feature encoder"""
import torch.nn as nn

from vap.cpc import load_CPC


class Encoder(nn.Module):
    """Encode speech information.

    Args:
        hist_bins_dim (int):
            Dimension of the voice activity history.
        pretrained (str):
            Either url to a downloadable checkpoint or
            path to a locally saved checkpoint.
            If None, clean state is initialised.
            Defaults to None.
        freeze (bool):
            Whether to freeze/not update the cpc wave
            encoder parameters during end-to-end training.
            Defaults to False.
        device (str):
            Storage location to place the model into.
            Defaults to cpu.

    """
    def __init__(
        self,
        hist_bins_dim,
        pretrained=None,
        freeze=True,
        device="cpu"
    ):
        super().__init__()
        # voice activity conditioning
        self.va_proj = nn.Linear(2, 256)
        self.va_hist_proj = nn.Linear(hist_bins_dim, 256)
        self.va_cond_norm = nn.LayerNorm(256)

        # waveform
        self.wave_encoder = load_CPC(pretrained, device)
        self.set_cpc_mode(freeze)

    def set_cpc_mode(self, freeze):
        """Freeze or unfreeze wave encoder parameters.

        Args:
            freeze (bool):
                Whether to freeze/not update the cpc wave
                encoder parameters during end-to-end training.

        """
        for p in self.wave_encoder.parameters():
            p.requires_grad_(not freeze)

    def forward(self, x):
        """Encode speech information.

        Args:
            x (dict):
                Holds the keys 'va', 'va_hist', 'waveform'
                with values of type torch.tensor and dimension:
                    va: ([B_Batch,] N_Speakers, L_Strides)
                    va_hist: ([B_Batch,] M_Bins, L_Strides)
                    waveform: ([B_Batch,] 1, K_Frames)

        Returns:
            torch.tensor:
                (B_Batch, L_Strides, F_Features(=256))

        """
        va = x["va"]
        va_hist = x["va_hist"]
        waveform = x["waveform"]

        if waveform.ndim < 3:
            # (N_Speakers, K_Frames)
            # -> (Batch, N_Speakers, K_Frames)
            waveform = waveform.unsqueeze(0)
            if va.ndim < 3:
                va = va.unsqueeze(0)
            if va_hist.ndim < 3:
                va_hist = va_hist.unsqueeze(0)

        # map voice activity features to the waveform feature space
        va_emb = self.va_proj(va.permute(0, 2, 1))
        va_hist_emb = self.va_hist_proj(va_hist.permute(0, 2, 1))
        va_cond_emb = self.va_cond_norm(va_emb + va_hist_emb)

        wave_emb = self.wave_encoder.gEncoder(waveform).permute(0, 2, 1)
        wave_emb = self.wave_encoder.gAR(wave_emb)

        return va_cond_emb + wave_emb
