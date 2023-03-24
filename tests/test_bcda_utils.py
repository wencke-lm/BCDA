import os

import pytest

from bcda.utils import BCDataset
from vap.data_loader import SwitchboardCorpus


@pytest.mark.depends(
    on=[
        f"tests/test_data_loader.py::TestSwitchboardCorpus",
    ]
)
class TestBCDataset:
    def test_load_bc_samples(self):
        swb_path = os.path.join(
           "tests",
           "data",
           "pseudo_switchboard"
        )
        swb = SwitchboardCorpus(
            os.path.join(swb_path, "swb_audios"),
            os.path.join(swb_path, "swb_ms98_transcriptions"),
            mono=True,
            sample_rate=16000
        )
        bc_path = os.path.join(
           "data",
           "swb",
           "utterance_is_backchannel.csv"
        )

        dataset = BCDataset(swb, bc_path)
        out = list(iter(dataset))

        assert len(out) == 278

        assert out[4]["speakers"] == "A"
        assert out[4]["labels"] == "ASSESSMENT"

        assert out[30]["speakers"] == "B"
        assert out[30]["labels"] == "CONTINUER"

    def test_load_bc_samples_mask_for_shorter_samples(self):
        swb_path = os.path.join(
           "tests",
           "data",
           "pseudo_switchboard"
        )
        swb = SwitchboardCorpus(
            os.path.join(swb_path, "swb_audios"),
            os.path.join(swb_path, "swb_ms98_transcriptions"),
            mono=True,
            sample_rate=16000
        )
        bc_path = os.path.join(
           "data",
           "swb",
           "utterance_is_backchannel.csv"
        )

        dataset = BCDataset(swb, bc_path)

        for out in iter(dataset):
            assert (out["va"][:, out["masks"]] == 0).all()
            assert (out["va_hist"][:, out["masks"]] == 0).all()
            assert (out["waveform"][:, ::160][:, out["masks"]] == 0).all()
