import os

import pytest
import torch
from torch.utils.data import DataLoader

from vap.data_loader import SwitchboardCorpusAll
from vap.encoder import Encoder


@pytest.fixture(scope="class")
def batch():
    try:
        swb_path = os.path.join(
           "tests",
           "data",
           "pseudo_switchboard"
        )
        swb = SwitchboardCorpusAll(
            os.path.join(swb_path, "swb_audios"),
            os.path.join(swb_path, "swb_ms98_transcriptions"),
            mono=True,
            sample_rate=16000
        )
        yield next(iter(DataLoader(swb, batch_size=2)))
    except:
        pytest.skip()


@pytest.mark.depends(
    on=[
        f"tests/test_data_loader.py::TestSwitchboardCorpus",
    ]
)
class TestEncoder:
    def test_encoder_no_batch_dim(self, batch):
        encoder = Encoder(hist_bins_dim=5)
        sample = {k: v[0] for k, v in batch.items()}
        out = encoder(sample)

        assert out.shape == torch.Size([1, 1000, 256])
        assert not out.isnan().any()

    def test_encode_with_batch_dim(self, batch):
        encoder = Encoder(hist_bins_dim=5)
        out = encoder(batch)

        assert out.shape == torch.Size([2, 1000, 256])
        assert not out.isnan().any()

    def test_encode_same_result_batch_no_batch(self, batch):
        encoder = Encoder(hist_bins_dim=5)
        sample = {k: v[0] for k, v in batch.items()}
        out_sample = encoder(sample)
        out_batch = encoder(batch)

        torch.testing.assert_close(out_sample[0], out_batch[0])
