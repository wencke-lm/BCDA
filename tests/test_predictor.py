import os

import pytest
import torch
from torch.utils.data import DataLoader

from vap.data_loader import SwitchboardCorpus
from vap.encoder import Encoder
from vap.predictor import Predictor, VAPHead


@pytest.mark.depends(
    on=[
        f"tests/test_encoder.py::TestEncoder",
        f"tests/test_data_loader.py::TestSwitchboardCorpus"
    ]
)
class TestPredictor:
    def test_predictor(self):
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
        batch = next(iter(DataLoader(swb, batch_size=2)))

        # feed batch through encoder
        encoder = Encoder(history_dim=5)
        batch = encoder(batch)

        predictor = Predictor(
            256, 3*256, 4, 4, "gelu", 0.1, 1024,
            [.1, .2, .3, .4], 0.5
        )
        out = predictor(batch)

        assert out.shape == torch.Size([2, 256])
        assert not out.isnan().any()

    def test_vap_head_get_gold_label(self):
        vap_head = VAPHead(256, [.1, .2, .3, .4], 0.5)
        activity = torch.tensor([
            [[1., 0., 0., 1., 1., 1., 0., 0., 0., 0.],
             [0., 1., 1., 0., 0., 0., 1., 1., 1., 1.]],

            [[1., 1., 0., 1., 0., 0., 1., 1., 1., 1.],
             [0., 0., 0., 1., 1., 0., 0., 0., 0., 0.]],

            [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]
        ])

        labels_out = vap_head.get_gold_label(activity)
        labels_expected = torch.zeros(3, 256)
        labels_expected[0, 165] = 1
        labels_expected[1, 210] = 1
        labels_expected[2, 255] = 1

        torch.testing.assert_close(labels_out, labels_expected)

    def test_vap_head_get_gold_label_small_window(self):
        vap_head = VAPHead(256, [.1, .2, .3, .4], 0.5)
        activity = torch.tensor([
            [[1., 1., 1., 1.],
             [1., 1., 1., 0.]]
        ])

        labels_out = vap_head.get_gold_label(activity)
        labels_expected = torch.zeros(1, 256)
        labels_expected[0, 119] = 1

        torch.testing.assert_close(labels_out, labels_expected)
