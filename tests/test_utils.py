import os

import torch

from vap.utils import (
    activity_start_end_idx_to_onehot,
    decimal_to_binary_tensor,
    get_activity_history,
    load_waveform
)


class TestDecimalToBinary:
    def test_decimal_to_binary_tensor(self):
        out = decimal_to_binary_tensor(9, tensor_len=5)
        expected = torch.tensor([0., 1., 0., 0., 1.])

        torch.testing.assert_close(out, expected)


class TestLoadWaveform:
    def test_load_waveform_with_start_time(self):
        audio, sr = load_waveform(
            os.path.join(
                "tests",
                "data",
                "example_student_long_female_en-US-Wavenet-G.wav"
            ),
            start_time=1.5
        )
        assert audio.shape[1] == 20088

    def test_load_waveform_with_end_time(self):
        audio, sr = load_waveform(
            os.path.join(
                "tests",
                "data",
                "example_student_long_female_en-US-Wavenet-G.wav"
            ),
            end_time=1.5
        )
        assert audio.shape[1] == 36000

    def test_load_waveform_with_start_and_end_time(self):
        audio, sr = load_waveform(
            os.path.join(
                "tests",
                "data",
                "example_student_long_female_en-US-Wavenet-G.wav"
            ),
            start_time=1.0, end_time=1.5
        )
        assert audio.shape[1] == 12000

    def test_load_waveform_and_normalize(self):
        audio, sr = load_waveform(
            os.path.join(
                "tests",
                "data",
                "example_student_long_female_en-US-Wavenet-G.wav"
            ),
            normalize=True
        )
        assert audio.abs().max() == 1.0

    def test_load_waveform_with_new_sample_rate(self):
        audio, sr = load_waveform(
            os.path.join(
                "tests",
                "data",
                "example_student_long_female_en-US-Wavenet-G.wav"
            ),
            start_time=1.0, end_time=2.0, sample_rate=12000
        )
        assert audio.shape[1] == 12000


class TestGetActivityOnehot:
    def test_get_activity_onehot(self):
        va_idx = [
            [[0.5, 1], [2, 3]],
            [[1, 2.5]]
        ]

        out = activity_start_end_idx_to_onehot(va_idx, 4.0, 2)
        expected = torch.tensor([
            [0, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0]
        ], dtype=torch.float32)

        torch.testing.assert_close(out, expected)

    def test_get_activity_onehot_partially_filled_frame(self):
        va_idx = [
            [[0.30, 1.30]],
            [[1.20, 2.5]]
        ]

        out = activity_start_end_idx_to_onehot(va_idx, 4.0, 2)
        expected = torch.tensor([
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0]
        ], dtype=torch.float32)

        torch.testing.assert_close(out, expected)

    def test_get_activity_onehot_single_speaker_always_active(self):
        va_idx = [
            [[0.0, 4.0]],
            []
        ]

        out = activity_start_end_idx_to_onehot(va_idx, 4.0, 2)
        expected = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=torch.float32)

        torch.testing.assert_close(out, expected)


class TestGetHistory:
    def test_get_hist(self):
        va = torch.tensor([
            [1, 0, 0, 1, 1, 0],
            [1, 1, 0, 0, 1, 1]
        ])
        bins = [3, 1, 0]

        out = torch.round(
            get_activity_history(va, bins),
            decimals=4
        )
        expected = torch.tensor([
            [
                [0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.0000],
                [0.5000, 0.3333, 0.5000],
                [0.5000, 0.0000, 1.0000],
                [0.3333, 1.0000, 0.5000],
                [0.3333, 0.6667, 0.0000]
            ],
            [
                [0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 1.0000],
                [0.5000, 0.6667, 0.5000],
                [0.5000, 1.0000, 0.0000],
                [0.6667, 0.0000, 0.5000],
                [0.6667, 0.3333, 1.0000]
            ]
        ])

        torch.testing.assert_close(out, expected)

    def test_get_hist_for_explicit_steps(self):
        va = torch.tensor([
            [1, 0, 0, 1, 1, 0],
            [1, 1, 0, 0, 1, 1]
        ])
        bins = [3, 1, 0]

        out = torch.round(
            get_activity_history(va, bins, n_step=6),
            decimals=4
        )
        expected = torch.tensor([
            [
                [0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.0000],
                [0.5000, 0.3333, 0.5000],
                [0.5000, 0.0000, 1.0000],
                [0.3333, 1.0000, 0.5000],
                [0.3333, 0.6667, 0.0000]
            ],
            [
                [0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 1.0000],
                [0.5000, 0.6667, 0.5000],
                [0.5000, 1.0000, 0.0000],
                [0.6667, 0.0000, 0.5000],
                [0.6667, 0.3333, 1.0000]
            ]
        ])

        torch.testing.assert_close(out, expected)

    def test_get_hist_for_smaller_step_size_than_smallest_bin(self):
        va = torch.tensor([
            [1, 1, 0, 1, 1, 0],
            [0, 0, 1, 1, 1, 1]
        ])
        bins = [4, 2, 0]

        out = torch.round(
            get_activity_history(va, bins, n_step=1),
            decimals=4
        )
        expected = torch.tensor([
            [
                [1.0000, 0.3333, 0.3333]
            ],
            [
                [0.0000, 0.6667, 0.6667]
            ]
        ])

        torch.testing.assert_close(out, expected)

    def test_get_hist_for_more_steps_than_possible(self):
        va = torch.tensor([
            [1, 0, 0, 1, 1, 0],
            [1, 1, 0, 0, 1, 1]
        ])
        bins = [3, 1, 0]

        out = torch.round(
            get_activity_history(va, bins, n_step=8),
            decimals=4
        )
        expected = torch.tensor([
            [
                [0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.0000],
                [0.5000, 0.3333, 0.5000],
                [0.5000, 0.0000, 1.0000],
                [0.3333, 1.0000, 0.5000],
                [0.3333, 0.6667, 0.0000]
            ],
            [
                [0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 0.5000],
                [0.5000, 0.5000, 1.0000],
                [0.5000, 0.6667, 0.5000],
                [0.5000, 1.0000, 0.0000],
                [0.6667, 0.0000, 0.5000],
                [0.6667, 0.3333, 1.0000]
            ]
        ])

        torch.testing.assert_close(out, expected)

    def test_get_hist_for_less_steps_than_possible(self):
        va = torch.tensor([
            [1, 0, 0, 1, 1, 0],
            [1, 1, 0, 0, 1, 1]
        ])
        bins = [3, 1, 0]

        out = torch.round(
            get_activity_history(va, bins, n_step=2),
            decimals=4
        )
        expected = torch.tensor([
            [
                [0.3333, 1.0000, 0.5000],
                [0.3333, 0.6667, 0.0000]
            ],
            [
                [0.6667, 0.0000, 0.5000],
                [0.6667, 0.3333, 1.0000]
            ]
        ])

        torch.testing.assert_close(out, expected)

    def test_get_hist_no_voice_activity(self):
        va = torch.tensor([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        bins = [3, 1, 0]

        out = torch.round(
            get_activity_history(va, bins),
            decimals=4
        )
        expected = torch.full((2, 6, 3), 0.5)

        torch.testing.assert_close(out, expected)

    def test_get_hist_short_context(self):
        va = torch.tensor([
            [1, 0],
            [0, 1]
        ])
        bins = [3, 1, 0]

        out = torch.round(
            get_activity_history(va, bins),
            decimals=4
        )
        expected = torch.tensor([
          [
              [0.5000, 0.5000, 1.0000],
              [0.5000, 1.0000, 0.0000]
          ],
          [
              [0.5000, 0.5000, 0.0000],
              [0.5000, 0.0000, 1.0000]
          ]
        ])

        torch.testing.assert_close(out, expected)

    def test_get_hist_more_than_two_speakers(self):
        va = torch.tensor([
            [0, 1, 1],
            [0, 0, 1],
            [0, 1, 0]
        ])
        bins = [1, 0]

        out = torch.round(
            get_activity_history(va, bins),
            decimals=4
        )
        expected = torch.tensor([
            [
                [0.3333, 0.3333],
                [0.3333, 0.5000],
                [0.5000, 0.5000]
            ],
            [
                [0.3333, 0.3333],
                [0.3333, 0.0000],
                [0.0000, 0.5000]
            ],
            [
                [0.3333, 0.3333],
                [0.3333, 0.5000],
                [0.5000, 0.0000]
            ]
        ])

        torch.testing.assert_close(out, expected)
