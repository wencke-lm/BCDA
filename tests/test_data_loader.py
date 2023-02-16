import os
import random

import pytest
import torch
from torch.utils.data import DataLoader

from vap.data_loader import SwitchboardCorpus
from vap.utils import activity_start_end_idx_to_onehot, get_audio_duration


@pytest.mark.depends(
    on=[
        f"tests/test_data_loader.py::TestSwitchboardCorpus",
    ]
)
class TestShuffledIterableDataset:
    def test_iter_same_sample_number_independent_of_buffer(self):
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
        assert sum(1 for _ in iter(swb1)) == sum(1 for _ in iter(swb2))

    def test_iter_in_pytorch_dataloder_all_samples_included(self):
        swb_path = os.path.join(
           "tests",
           "data",
           "pseudo_switchboard"
        )
        swb = SwitchboardCorpus(
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

    def test_iter_same_output_independent_of_audio_load_mode(self):
        swb_path = os.path.join(
           "tests",
           "data",
           "pseudo_switchboard"
        )
        swb1 = SwitchboardCorpus(
            os.path.join(swb_path, "swb_audios"),
            os.path.join(swb_path, "swb_ms98_transcriptions"),
            load_audio=True
        )
        swb2 = SwitchboardCorpus(
            os.path.join(swb_path, "swb_audios"),
            os.path.join(swb_path, "swb_ms98_transcriptions"),
            load_audio=False
        )
        random.seed(24)
        sample1 = next(iter(swb1))
        random.seed(24)
        sample2 = next(iter(swb2))
        torch.testing.assert_close(sample1["waveform"], sample2["waveform"])


@pytest.mark.depends(
    on=[
        f"tests/test_utils.py",
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

                    if word.strip() not in {"[silence]", "[noise]"} :
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
