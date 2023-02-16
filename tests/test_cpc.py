import os

import pytest
import torch

from vap.cpc import load_CPC
from vap.utils import load_waveform


@pytest.mark.depends(
    on=[
        f"tests/test_utils.py::TestLoadWaveform",
    ]
)
class TestCPC:
    def test_load_cpc_clean_state(self):
        model = load_CPC()
        audio, sr = load_waveform(
            os.path.join(
                "tests",
                "data",
                "example_student_long_female_en-US-Wavenet-G.wav"
            ),
            start_time=1.0, end_time=2.0, sample_rate=16000
        )
        z = model.gEncoder(audio.unsqueeze(0)).permute(0, 2, 1)
        z = model.gAR(z)

        assert z.shape == torch.Size([1, 100, 256])

    def test_load_cpc_pretrained(self):
        model = load_CPC(
            "https://dl.fbaipublicfiles.com/librilight/"
            "CPC_checkpoints/60k_epoch4-d0f474de.pt"
        )
        audio, sr = load_waveform(
            os.path.join(
                "tests",
                "data",
                "example_student_long_female_en-US-Wavenet-G.wav"
            ),
            start_time=1.0, end_time=2.0, sample_rate=16000
        )
        z = model.gEncoder(audio.unsqueeze(0)).permute(0, 2, 1)
        z = model.gAR(z)

        assert z.shape == torch.Size([1, 100, 256])

    def test_load_cpc_pretrained_to_gpu(self):
        model = load_CPC(
            "https://dl.fbaipublicfiles.com/librilight/"
            "CPC_checkpoints/60k_epoch4-d0f474de.pt",
            device="cuda:0"
        )
        audio, sr = load_waveform(
            os.path.join(
                "tests",
                "data",
                "example_student_long_female_en-US-Wavenet-G.wav"
            ),
            start_time=1.0, end_time=2.0, sample_rate=16000
        )
        z = model.gEncoder(audio.unsqueeze(0)).permute(0, 2, 1)
        z = model.gAR(z)

        assert z.shape == torch.Size([1, 100, 256])
