import os

import pytest
import torch
from torch.utils.data import DataLoader

from vap.data_loader import SwitchboardCorpusAll
from vap.encoder import Encoder
from vap.predictor import Predictor, VAPHead


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
        f"tests/test_encoder.py::TestEncoder",
        f"tests/test_data_loader.py::TestSwitchboardCorpus"
    ]
)
class TestPredictor:
    def test_predictor(self, batch):
        # feed batch through encoder
        encoder = Encoder(hist_bins_dim=5)
        batch = encoder(batch)

        predictor = Predictor(
            256, 3*256, 4, 4, "gelu", 0.1, 1024,
            [.1, .2, .3, .4], 0.5
        )
        out = predictor(batch)

        assert out.shape == torch.Size([2, 1000, 256])
        assert not out.isnan().any()

    def test_vap_head_extract_gold_labels_threshold(self):
        vap_head = VAPHead(256, [10, 9, 7, 4, 0], 0.5)
        # binary config: [1, 0, 0, 1, 0, 1, 0, 1]
        activity = torch.tensor([[
            [0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0.],
            [0., 0., 1., 1., 0., 1., 0., 1., 1., 1., 0.]
        ]])

        labels_out = vap_head.extract_gold_labels(activity)
        labels_expected = torch.tensor([[149]])

        torch.testing.assert_close(labels_out, labels_expected)

    def test_vap_head_extract_gold_labels_multiple_dimension_same_size(self):
        vap_head = VAPHead(256, [10, 9, 7, 4, 0], 0.5)
        # binary configs:
        # [1, 0, 0, 1, 0, 1, 0, 1]
        ...
        # [0, 1, 0, 1, 0, 1, 0, 1]
        # [1, 1, 0, 0, 1, 1, 0, 0]
        # [1, 0, 1, 0, 1, 1, 1, 0]
        activity = torch.tensor([[
            [0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0.],
            [0., 0., 1., 1., 0., 1., 0., 1., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0.]
        ]])

        labels_out = vap_head.extract_gold_labels(activity)
        labels_expected = torch.tensor([[149,  15,  47,  55,  95,  85, 204, 174]])

        torch.testing.assert_close(labels_out, labels_expected)

    def test_vap_head_extract_gold_labels_batch(self):
        vap_head = VAPHead(256, [10, 9, 7, 4, 0], 0.5)
        activity = torch.tensor([
            [
                [0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0.],
                [0., 0., 1., 1., 0., 1., 0., 1., 1., 1., 0., 0.]
            ],
            [
                [0., 1., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0.],
                [0., 1., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0.]
            ]
        ])

        labels_out = vap_head.extract_gold_labels(activity)
        labels_expected = torch.tensor([[149,  15], [204, 174]])

        torch.testing.assert_close(labels_out, labels_expected)