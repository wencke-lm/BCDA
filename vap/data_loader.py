"""data_loader.py - Datasets & Dataloaders"""
from abc import ABC, abstractmethod
from decimal import *
import os
import random

import torch
from torch.utils.data import IterableDataset

from vap.utils import (
    activity_start_end_idx_to_onehot,
    get_activity_history,
    get_audio_duration,
    load_waveform
)


class ShuffledIterableDataset(IterableDataset, ABC):
    """Dataset loader for arbitrary dialogue datasets.

    Args:
        audio_path (str):
            Path to a folder holding audio files.
            One audio file per dialogue.
        split (str/list):
            Path to an utf8 encoded txt-file holding one
            conversation identifer per line or
            collection of relevant conversation identifiers.
        n_stride (int):
            Number of discrete activity values to save per second.
            Should equal waveform sample_rate divided by 160.
        va_hist_bins (list[int]):
            Right boundaries of regions for which the
            ratio of speaker's activity shall be calculated.
        sample_len (int):
            Sample length in seconds per audio chunk.
            Holding input and prediction window.
        sample_overlap (int):
            Length in seconds of the audio chunk overlap.
            Holding the prediction window.
        buffer_size (int):
            Storage for non-randomly sampled data points
            from which one is randomly sampled at each
            time step. The larger it is, the closer to true
            randomness.

    """
    def __init__(
        self,
        audio_path,
        split,
        n_stride=100,
        va_hist_bins=[60, 30, 10, 5, 0],
        sample_len=10,
        sample_overlap=2,
        buffer_size=1600,
        **kwargs
    ):
        if isinstance(split, str):
            with open(split, 'r', encoding="utf-8") as data:
                self.split = data.read().strip().split("\n")
        else:
            self.split = split

        self.n_stride = n_stride
        self.va_hist_bins = va_hist_bins
        self.sample_len = sample_len
        self.sample_overlap = sample_overlap
        self.buffer_size = buffer_size

        self.audio_path = audio_path
        self.audio_kwargs = kwargs

    def _prepare_dialogue(self, dialogue_id, load_audio=True):
        """Extract voice activity (history) and audio information."""
        if str(dialogue_id) not in self.split:
            raise ValueError("Conversation with that identifier not included.")

        audio_file = os.path.join(
            self.audio_path,
            self._id_to_audio_filename(dialogue_id)
        )
        audio_len = get_audio_duration(audio_file)

        # voice activity
        va = self._get_activity(dialogue_id)
        va = activity_start_end_idx_to_onehot(
            va, audio_len, self.n_stride
        )
        # voice activity history
        bins = [idx*self.n_stride for idx in self.va_hist_bins]
        va_hist = get_activity_history(va, bins)[0].permute(1, 0)

        # waveform
        if load_audio:
            wf, sr = load_waveform(audio_file, **self.audio_kwargs)

            return {
                "va": va,
                "va_hist": va_hist,
                "waveform": wf,
                "sample_rate": sr
            }
        # add info necessary to load audio
        return {
            "va": va,
            "va_hist": va_hist,
            "file": audio_file
        }

    def select_samples(self, dialogue_id, time_stamps, test=False):
        """Generate targeted data samples.

        Args:
            dialogue_id (str):
                Dialogue identifier to sample from.
            time_stamps (list):
                Collection of timestamps in seconds, each
                marking the start of a prediction window.
            test (bool):
                Whether the sample is to be used at train
                or test time.

        Yields:
            dict: {
                "va": (N_Speakers, L_Strides),
                "va_hist": (M_Bins, L_Strides),
                "waveform": (N_Speakers, K_Frames)
                "sample_rate": int
            }

        """
        if str(dialogue_id) not in self.split:
            raise ValueError("Conversation with that identifier not included.")

        diag = self._prepare_dialogue(dialogue_id)
        sr = diag["sample_rate"]

        for ts in time_stamps:
            start = max(ts - (self.sample_len - self.sample_overlap), 0)
            end = ts + self.sample_overlap

            if not test:
                # yield item including prediction window
                yield {
                    "va": diag["va"]
                        [:, int(self.n_stride*start):int(self.n_stride*end)],
                    "va_hist": diag["va_hist"]
                        [:, int(self.n_stride*start):int(self.n_stride*end)],
                    "waveform": diag["waveform"]
                        [:, int(sr*start):int(sr*end)],
                    "sample_rate": sr
                }
            else:
                # yield item including only model input
                yield {
                    "va": diag["va"]
                        [:, int(self.n_stride*start):int(self.n_stride*ts)],
                    "va_hist": diag["va_hist"]
                        [:, int(self.n_stride*start):int(self.n_stride*ts)],
                    "waveform": diag["waveform"]
                        [:, int(sr*start):int(sr*ts)],
                    "sample_rate": sr
                }

    def generate_samples(self, load_audio=True):
        """Generate non-shuffled consequtive dialogue samples.

        Args:
            load_audio (bool):
                Whether to load the audiofile into a tensor
                or instead only return filename and timestamp
                marking the start of a sample.

        Returns:
            If load_audio is TRUE
            dict: {
                "va": (N_Speakers, L_Strides),
                "va_hist": (M_Bins, L_Strides),
                "waveform": (N_Speakers, K_Frames)
            }
            If load_audio is FALSE
            dict: {
                "va": (N_Speakers, L_Strides),
                "va_hist": (M_Bins, L_Strides),
                "file": str,
                "start": int
            }

        """
        step = self.sample_len - self.sample_overlap

        for dialogue_id in self.split:
            diag = self._prepare_dialogue(dialogue_id, load_audio)

            # voice activity
            va_sample = diag["va"].unfold(
                dimension=1,
                size=self.sample_len*self.n_stride,
                step=step*self.n_stride,
            )

            # voice activity history
            va_hist_sample = diag["va_hist"].unfold(
                dimension=1,
                size=self.sample_len*self.n_stride,
                step=step*self.n_stride,
            )

            # waveform
            if load_audio:
                sr = diag["sample_rate"]
                wf_sample = diag["waveform"].unfold(
                    dimension=1,
                    size=self.sample_len*sr,
                    step=step*sr
                )
                assert (
                    va_sample.shape[1]
                    == va_hist_sample.shape[1]
                    == wf_sample.shape[1]
                )
            else:
                assert (
                    va_sample.shape[1]
                    == va_hist_sample.shape[1]
                )

            for s_i in range(va_sample.shape[1]):
                if load_audio:
                    yield {
                        "va": va_sample[:, s_i],
                        "va_hist": va_hist_sample[:, s_i],
                        "waveform": wf_sample[:, s_i],
                        "sample_rate": sr
                    }
                else:
                    # add info necessary to load audio
                    yield {
                        "va": va_sample[:, s_i],
                        "va_hist": va_hist_sample[:, s_i],
                        "file": diag["file"],
                        "start": s_i*step
                    }

    def generate_shift_hold_samples(self, silence=10, offset=100, load_audio=True):
        """Generate non-shuffled shift/hold dialogue samples.

        Shift/Hold describe turn-taking events where in the former a
        speaker that was active previously stops speaking and the
        one of its listeners begins speaking. In the latter,
        the speaker simply continues speaking.

        To limit the analysis to proper events, we require a minimum
        duration of mutual silence at the beginning of the prediction
        window and a offset before and after the silence, where only
        one speaker is allowed to be active.

        Args:
            silence (int):
                Minimum duration of mutual silence in activity frames.
            offset (int):
                Minimum duration of regions of single speakership
                before and after the mutual silence in activity frames.
            load_audio (bool):
                Whether to load the audiofile into a tensor
                or instead only return filename and timestamp
                marking the start of a sample.

        Returns:
            If load_audio is TRUE
            dict: {
                "va": (N_Speakers, L_Strides),
                "va_hist": (M_Bins, L_Strides),
                "waveform": (N_Speakers, K_Frames)
            }
            If load_audio is FALSE
            dict: {
                "va": (N_Speakers, L_Strides),
                "va_hist": (M_Bins, L_Strides),
                "file": str,
                "start": int
            }

        """
        for dialogue_id in self.split:
            diag = self._prepare_dialogue(dialogue_id, load_audio)

            non_silence = torch.nonzero(diag["va"].sum(dim=0)).flatten()

            for sil_start, sil_end in zip(non_silence, non_silence[1:]):
                # long mutual silence
                if sil_end - sil_start > silence + 1:
                    pre_who_speaks = (
                        diag["va"][:, sil_start-offset-1:sil_start+1].sum(dim=-1)
                    )
                    post_who_speaks = (
                        diag["va"][:, sil_end:sil_end+offset].sum(dim=-1)
                    )

                    # pre-offset only one speaker active
                    # post-offset only one speaker active
                    if (
                        torch.count_nonzero(pre_who_speaks) == 1
                        and torch.count_nonzero(post_who_speaks) == 1
                    ):
                        mid = (int(sil_start) + silence//2)/self.n_stride
                        start = mid - (self.sample_len - self.sample_overlap)
                        end = mid + self.sample_overlap

                        if start < 0 or end > diag["va"].shape[-1]/self.n_stride:
                            continue

                        yield_item = {
                            "va": diag["va"]
                                [:, round(start*self.n_stride):round(end*self.n_stride)],
                            "va_hist": diag["va_hist"]
                                [:, round(start*self.n_stride):round(end*self.n_stride)]
                        }

                        pre_speaker = int(torch.nonzero(pre_who_speaks))
                        post_speaker = int(torch.nonzero(post_who_speaks))

                        # HOLD
                        if pre_speaker == post_speaker:
                            yield_item["event"] = ("HOLD", pre_speaker)
                        # SHIFT
                        else:
                            yield_item["event"] = ("SHIFT", pre_speaker)

                        if load_audio:
                            sr = diag["sample_rate"]
                            yield_item["sample_rate"] = sr
                            yield_item["waveform"] = (
                                diag["waveform"][:, round(sr*start):round(sr*end)]
                            )
                        else:
                            yield_item["file"] = diag["file"]
                            yield_item["start"] = start

                        yield yield_item

    def generate_random_samples(self, sampler, load_audio=True):
        """Generate random data samples.

        As random reads of conversation samples are expensive,
        we read in conversations as a whole. Shuffling the order
        of conversations for each epoch and maintaining a buffer
        large enough to hold samples from several conversations,
        allows us to create approximate random batches.

        Yields:
            dict: {
                "va": (N_Speakers, L_Strides),
                "va_hist": (M_Bins, L_Strides),
                "waveform": (N_Speakers, K_Frames)
                "sample_rate": int
            }

        """
        # sample origins inside batch should vary between epochs
        random.shuffle(self.split)
        buffer = []

        # fill buffer with ordered/not random samples
        try:
            while len(buffer) < self.buffer_size:
                buffer.append(next(sampler))
        except StopIteration:
            self.buffer_size = len(buffer)

        # randomly sample from buffer
        while len(buffer) > 0:
            yield_idx = random.randrange(len(buffer))
            yield_item = buffer[yield_idx]

            if len(buffer) == self.buffer_size:
                try:
                    new_item = next(sampler)
                    buffer[yield_idx] = new_item
                except StopIteration:
                    buffer.pop(yield_idx)
            else:
                buffer.pop(yield_idx)

            if load_audio and "waveform" not in yield_item:
                # load waveform if selected for batch
                yield_item_start = yield_item.pop("start")
                yield_item["waveform"], sr = load_waveform(
                    yield_item.pop("file"),
                    yield_item_start,
                    yield_item_start + self.sample_len,
                    **self.audio_kwargs
                )
                yield_item["sample_rate"] = sr

            yield yield_item

    @abstractmethod
    def _id_to_audio_filename(self, dialogue_id):
        """Transform conversation id to valid audio filename.

        Args:
            dialogue_id (str):
                Conversation identifier.

        Returns:
            str:
                Filename including file-ending (e.g. .wav),
                but without path.

        """
        pass

    @abstractmethod
    def _get_activity(self, dialogue_id):
        """Load voice activity information.

        Usually start and end timestamps of each unit of
        active speech are read from text transcriptions,
        but one might also use an automatic approach for
        extracting them.

        Args:
            dialogue_id (str):
                Identifier of one file in the corpus.

        Returns:
            list[list[list[int]]]: (N_Speakers, _, 2),
                where _ is a variable number of voice activities
                for speaker n each defined as the start and end
                timestamp of active speech in seconds.

        """
        pass


class SwitchboardCorpus(ShuffledIterableDataset):
    """Dataset loader for the Switchboard Corpus.

    Args:
        text_path (str):
            Path to a folder holding the Mississippi State
            Transcript of the Switchboard-1 Release 2.
            The folder has to have the structure:

            text_path/
                ->20/
                    ->2001/
                        -> sw2001A-ms98-a-trans.text
                        -> sw2001A-ms98-a-word.text
                        -> sw2001B-ms98-a-trans.text
                        -> sw2001B-ms98-a-word.text
                    ->2005/
                    ...
                    ->2096/
                ->21/
                ...
                ->49/

    """
    def __init__(self, audio_path, text_path, split_info=None, **kwargs):
        if split_info is None:
            split_info = [f[3:7] for f in os.listdir(audio_path)]
        super().__init__(audio_path, split_info, **kwargs)

        self.text_path = text_path

    def _id_to_audio_filename(self, dialogue_id):
        """Transform dialogue id to valid audio filename."""
        return f"sw0{dialogue_id}.sph"

    def _get_activity(self, dialogue_id):
        """Load from Mississippi State word-unit transcript."""
        va = [[], []]
        dialogue_id = str(dialogue_id)

        for i, speaker in enumerate(["A", "B"]):
            filename = os.path.join(
                self.text_path,
                dialogue_id[:2],
                dialogue_id,
                f"sw{dialogue_id}{speaker}-ms98-a-word.text"
            )

            skip = False
            last_start, last_end = None, None

            with open(filename, 'r', encoding="utf-8") as trans:
                for line in trans:
                    _, start, end, word = line.split(maxsplit=3)

                    if word.strip() == "<e_aside>":
                        skip = False
                    if skip:
                        continue
                    if word.strip() == "<b_aside>":
                        skip = True

                    # collect full IPUs
                    # (= speech units preceeded and followed by silence/noise)
                    if word.strip() not in {
                        "[silence]", "[noise]", "[laughter]", "[vocalized-noise]",
                        "<b_aside>", "<e_aside>"
                    }:
                        if last_start is None:
                            last_start = start
                        last_end = end
                    else:
                        if last_start is not None:
                            va[i].append([float(last_start), float(start)])
                            last_start, last_end = None, None

                if last_start is not None:
                    va[i].append([float(last_start), float(last_end)])
        return va


class SwitchboardCorpusAll(SwitchboardCorpus):
    def __iter__(self):
        for s in self.generate_random_samples(self.generate_samples()):
            yield s


class SwitchboardCorpusShiftHold(SwitchboardCorpus):
    def __iter__(self):
        for s in self.generate_shift_hold_samples():
            yield s
