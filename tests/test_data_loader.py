import os
import random
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader

from vap.data_loader import (
    SwitchboardCorpus,
    SwitchboardCorpusAll
)
from vap.utils import (
    activity_start_end_idx_to_onehot,
    get_audio_duration,
    load_waveform
)


@pytest.mark.depends(
    on=[
        f"tests/test_data_loader.py::TestSwitchboardCorpus",
    ]
)
class TestShuffledIterableDataset:
    def test_prepare_dialogue(self):
        swb_path = os.path.join(
           "tests",
           "data",
           "pseudo_switchboard"
        )
        swb = SwitchboardCorpus(
            os.path.join(swb_path, "swb_audios"),
            os.path.join(swb_path, "swb_ms98_transcriptions"),
            os.path.join(swb_path, "conversations.split"),
            sample_rate=12000,
            n_stride=100
        )
        diag = swb._prepare_dialogue("2158", load_audio=True)

        assert diag["va"].shape[-1] == diag["va_hist"].shape[-1]
        assert diag["va"].shape[-1] == diag["waveform"].shape[-1]//120

    def test_select_samples(self):
        swb_path = os.path.join(
           "tests",
           "data",
           "pseudo_switchboard"
        )
        swb = SwitchboardCorpus(
            os.path.join(swb_path, "swb_audios"),
            os.path.join(swb_path, "swb_ms98_transcriptions"),
            os.path.join(swb_path, "conversations.split"),
            sample_rate=12000
        )
        sample = next(swb.select_samples(2158, [9.0]))

        torch.testing.assert_close(
            sample["va"].shape, torch.Size([2, 1000])
        )
        torch.testing.assert_close(
            sample["va_hist"].shape, torch.Size([5, 1000])
        )
        torch.testing.assert_close(
            sample["waveform"].shape, torch.Size([2, 120000])
        )

        assert sample["va"][0, :90].sum() == 0
        assert sample["va"][0, 91] == 1
        assert sample["va"][1, :95].sum() == 0
        assert sample["va"][1, 96] == 1

    def test_select_samples_input_window_ends_at_prediction(self):
        swb_path = os.path.join(
           "tests",
           "data",
           "pseudo_switchboard"
        )
        swb = SwitchboardCorpus(
            os.path.join(swb_path, "swb_audios"),
            os.path.join(swb_path, "swb_ms98_transcriptions"),
            os.path.join(swb_path, "conversations.split"),
            sample_rate=12000
        )
        sample = next(swb.select_samples(2158, [260.057125]))

        assert sample["va"][0, 799] == 0
        assert sample["va"][1, 800] == 1

    def test_generate_samples_from_split(self):
        swb_path = os.path.join(
           "tests",
           "data",
           "pseudo_switchboard"
        )
        swb = SwitchboardCorpus(
            os.path.join(swb_path, "swb_audios"),
            os.path.join(swb_path, "swb_ms98_transcriptions"),
            os.path.join(swb_path, "conversations.split")
        )
        with patch(
            "vap.data_loader.load_waveform", wraps=load_waveform
        ) as mock_function:
            for item in swb.generate_samples():
                pass
            mock_function.assert_any_call(
                os.path.join(swb_path, "swb_audios", "sw02158.sph")
            )
            mock_function.assert_any_call(
                os.path.join(swb_path, "swb_audios", "sw04709.sph")
            )
            assert mock_function.call_count == 2

    def test_generate_samples_same_output_independent_of_audio_load_mode(self):
        swb_path = os.path.join(
           "tests",
           "data",
           "pseudo_switchboard"
        )
        swb = SwitchboardCorpus(
            os.path.join(swb_path, "swb_audios"),
            os.path.join(swb_path, "swb_ms98_transcriptions"),
            os.path.join(swb_path, "conversations.split")
        )

        random.seed(24)
        sample1 = next(
            swb.generate_random_samples(swb.generate_samples(load_audio=True))
        )
        random.seed(24)
        sample2 = next(
            swb.generate_random_samples(swb.generate_samples(load_audio=False))
        )
        # only works when no resampling is applied
        torch.testing.assert_close(sample1["waveform"], sample2["waveform"])

    def test_generate_shift_hold_samples(self):
        swb_path = os.path.join(
           "tests",
           "data",
           "pseudo_switchboard"
        )
        swb = SwitchboardCorpus(
            os.path.join(swb_path, "swb_audios"),
            os.path.join(swb_path, "swb_ms98_transcriptions"),
            ["2158"]
        )
        sample = next(swb.generate_shift_hold_samples(load_audio=False))

        assert round(float(sample["start"]), 2) == 1.61
        assert sample["event"] == ("SHIFT", 0)

    def test_generate_shift_hold_samples_no_incomplete_samples(self):
        swb_path = os.path.join(
           "tests",
           "data",
           "pseudo_switchboard"
        )
        swb = SwitchboardCorpus(
            os.path.join(swb_path, "swb_audios"),
            os.path.join(swb_path, "swb_ms98_transcriptions"),
            ["2158"]
        )
        for sample in swb.generate_shift_hold_samples(load_audio=False):
            assert sample["va"].shape[-1] == 1000

    def test_generate_shift_hold_samples_same_output_independent_of_audio_load_mode(self):
        swb_path = os.path.join(
           "tests",
           "data",
           "pseudo_switchboard"
        )
        swb = SwitchboardCorpus(
            os.path.join(swb_path, "swb_audios"),
            os.path.join(swb_path, "swb_ms98_transcriptions"),
            os.path.join(swb_path, "conversations.split"),
            n_stride=100
        )

        random.seed(24)
        sample1 = next(
            swb.generate_random_samples(swb.generate_shift_hold_samples(load_audio=True))
        )
        random.seed(24)
        sample2 = next(
            swb.generate_random_samples(swb.generate_shift_hold_samples(load_audio=False))
        )
        # only works when no resampling is applied
        torch.testing.assert_close(sample1["waveform"], sample2["waveform"])

    def test_generate_random_samples_same_number_independent_of_buffer(self):
        swb_path = os.path.join(
           "tests",
           "data",
           "pseudo_switchboard"
        )
        # less buffer than samples
        swb1 = SwitchboardCorpus(
            os.path.join(swb_path, "swb_audios"),
            os.path.join(swb_path, "swb_ms98_transcriptions"),
            buffer_size=10
        )
        # all samples loaded into buffer
        swb2 = SwitchboardCorpus(
            os.path.join(swb_path, "swb_audios"),
            os.path.join(swb_path, "swb_ms98_transcriptions"),
            buffer_size=float("inf")
        )
        assert (
            sum(1 for _ in swb1.generate_random_samples(swb1.generate_samples()))
            == sum(1 for _ in swb2.generate_random_samples(swb2.generate_samples()))
        )

    def test_generate_random_samples_in_pytorch_dataloder_all_samples_included(self):
        swb_path = os.path.join(
           "tests",
           "data",
           "pseudo_switchboard"
        )
        swb = SwitchboardCorpusAll(
            os.path.join(swb_path, "swb_audios"),
            os.path.join(swb_path, "swb_ms98_transcriptions"),
            buffer_size=10
        )

        # check that each loader includes all/same samples
        sb_loader1 = DataLoader(swb, batch_size=2)
        all_va_1 = []
        for batch in sb_loader1:
            all_va_1.append(batch["va"][0])
            all_va_1.append(batch["va"][1])

        sb_loader2 = DataLoader(swb, batch_size=2)
        all_va_2 = []
        for batch in sb_loader2:
            all_va_2.append(batch["va"][0])
            all_va_2.append(batch["va"][1])

        assert len(all_va_1) == len(all_va_2)
        assert all(
            any(torch.equal(va1, va2) for va1 in all_va_1)
            for va2 in all_va_2
        )
        assert all(
            any(torch.equal(va1, va2) for va2 in all_va_2)
            for va1 in all_va_1
        )


@pytest.mark.depends(
    on=[
        f"tests/test_vap_utils.py",
    ]
)
class TestSwitchboardCorpus:
    def _get_activity_inefficently(self, swb, dialogue_id):
        va = [[], []]
        dialogue_id = str(dialogue_id)

        for i, speaker in enumerate(["A", "B"]):
            filename = os.path.join(
                swb.text_path,
                dialogue_id[:2],
                dialogue_id,
                f"sw{dialogue_id}{speaker}-ms98-a-word.text"
            )

            skip = False

            with open(filename, 'r', encoding="utf-8") as trans:
                for line in trans:
                    _, start, end, word = line.split(maxsplit=3)

                    if word.strip() == "<b_aside>":
                        skip = True
                    if word.strip() == "<e_aside>":
                        skip = False
                        continue
                    if skip:
                        continue

                    if word.strip() not in {"[silence]", "[noise]"}:
                        va[i].append([float(start), float(end)])

        return va

    def test_get_activity_compared_to_less_efficient_implementation(self):
        # we compare the results of our complex but efficient implementation
        # with an easy but inefficient implementation

        # verifying the output of the later is more straightforward
        # if both have the same output, we can be sure both are correct
        swb_path = os.path.join(
           "tests",
           "data",
           "pseudo_switchboard"
        )
        swb = SwitchboardCorpus(
            os.path.join(swb_path, "swb_audios"),
            os.path.join(swb_path, "swb_ms98_transcriptions")
        )

        for file in os.listdir(os.path.join(swb_path, "swb_audios")):
            audio_len = get_audio_duration(
                os.path.join(swb.audio_path, file)
            )

            efficient_out = activity_start_end_idx_to_onehot(
                swb._get_activity(file[3:7]),
                audio_len,
                swb.n_stride
            )
            inefficient_out = activity_start_end_idx_to_onehot(
                self._get_activity_inefficently(swb, file[3:7]),
                audio_len,
                swb.n_stride
            )

            torch.testing.assert_close(efficient_out, inefficient_out)

    def test_get_activity(self):
        swb_path = os.path.join(
           "tests",
           "data",
           "pseudo_switchboard"
        )
        swb = SwitchboardCorpus(
            os.path.join(swb_path, "swb_audios"),
            os.path.join(swb_path, "swb_ms98_transcriptions")
        )

        out = self._get_activity_inefficently(swb, 2158)

        assert len(out[0]) == 955
        assert len(out[1]) == 1329
