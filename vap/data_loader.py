"""data_loader.py - Datasets & Dataloaders"""
from abc import ABC, abstractmethod
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
    """Iterable dataset that returns samples approximately random.

    Args:
        audio_path (str):
            Path to a folder holding audio files.
        split_info (str/list):
            Path to an utf8 encoded txt-file holding one
            conversation identifer per line.
            List holding all conversation identifiers.
        n_stride (int):
            Number of discrete activity values to save per s.
            Should equal waveform sample_rate divided by 160.
        va_hist_bins (list[int]):
            Right boundaries of regions for which the
            ratio of speaker's activity shall be calculated.
        sample_len (int):
            Length of audio chunks to feed into model.
            Given in seconds.
        sample_overlap (int):
            Overlap between consecutive audio chunks.
            Given in seconds.
        buffer_size (int):
            Storage for non-randomly sampled data points
            from which one is randomly sampled at each
            time step. The larger it, the closer to true
            randomness.
    """
    def __init__(
        self,
        audio_path,
        split_info,
        mode="train",
        n_stride=100,
        va_hist_bins=[60, 30, 10, 5, 0],
        sample_len=10,
        sample_overlap=2,
        pred_window=2,
        buffer_size=1600,
        **kwargs
    ):
        # collect conversation identifiers
        if isinstance(split_info, str):
            with open(split_info, 'r', encoding="utf-8") as data:
                self.data = data.read().strip().split("\n")
        else:
            self.data = split_info

        self.mode = mode

        self.n_stride = n_stride
        self.va_hist_bins = va_hist_bins
        self.sample_len = sample_len
        self.sample_overlap = sample_overlap
        self.pred_window = pred_window
        self.buffer_size = buffer_size

        self.audio_path = audio_path
        self.load_audio = kwargs.pop(
            "load_audio", True
        )
        self.audio_kwargs = kwargs

    def select_samples(self, dialogue_id, time_stamps):
        """Generate targeted data samples.

        Args:
            dialogue_id (int):
                Dialogue to sample from.
            time_stamps (list/float):
                Timestamp marking start of prediction window
                or list of such timestamps in seconds.

        Yields:
            dict: {
                "va": (N_Speakers, L_Strides),
                "va_hist": (M_Bins, L_Strides),
                "waveform": (N_Speakers, K_Frames)
            }

        """
        file = self._id_to_audio_filename(dialogue_id)

        # voice activity
        audio_len = get_audio_duration(
            os.path.join(self.audio_path, file)
        )
        va = self._get_activity(dialogue_id)
        va = activity_start_end_idx_to_onehot(
            va, audio_len, self.n_stride
        )

        # voice activity history
        bins = [idx*self.n_stride for idx in self.va_hist_bins]
        va_hist = get_activity_history(va, bins)[0].permute(1, 0)

        # waveform
        wf, sr = load_waveform(
            os.path.join(self.audio_path, file),
            **self.audio_kwargs
        )

        if not isinstance(time_stamps, list):
            time_stamps = [time_stamps]

        for ts in time_stamps:
            inp_window = self.sample_len - self.pred_window

            yield {
                "va": va
                    [:, :int(self.n_stride*ts)]
                    [:, -int(self.n_stride*inp_window):],
                "va_hist": va_hist
                    [:, :int(self.n_stride*ts)]
                    [:, -int(self.n_stride*inp_window):],
                "waveform": wf
                    [:, :int(sr*ts)]
                    [:, -int(sr*inp_window):],
                "labels": va
                    [:, int(self.n_stride*ts):]
                    [:, :int(self.n_stride*self.pred_window)]
            }

    def generate_shift_hold_samples(self, silence=10, offset=100):
        for dialogue_id in self.data:
            file = self._id_to_audio_filename(dialogue_id)

            # voice activity
            audio_len = get_audio_duration(
                os.path.join(self.audio_path, file)
            )
            va = self._get_activity(dialogue_id)
            va = activity_start_end_idx_to_onehot(
                va, audio_len, self.n_stride
            )

            # waveform
            wf, sr = load_waveform(
                os.path.join(self.audio_path, file),
                **self.audio_kwargs
            )

            # voice activity history
            bins = [idx*self.n_stride for idx in self.va_hist_bins]
            va_hist = get_activity_history(va, bins)[0].permute(1, 0)

            non_silence = torch.nonzero(va.sum(dim=0)).flatten()

            for start, end in zip(non_silence, non_silence[1:]):
                # long mutual silence
                if end - start > silence + 1:
                    pre_who_speaks = (
                        va[:, start-offset-1:start+1].sum(dim=-1)
                    )
                    post_who_speaks = (
                        va[:, end:end+offset].sum(dim=-1)
                    )

                    # pre-offset only one speaker active
                    # post-offset only one speaker active
                    if (
                        torch.count_nonzero(pre_who_speaks) == 1
                        and torch.count_nonzero(post_who_speaks) == 1
                    ):
                        pre_speaker = int(torch.nonzero(pre_who_speaks))
                        post_speaker = int(torch.nonzero(post_who_speaks))

                        # HOLD
                        if pre_speaker == post_speaker:
                            event = "HOLD"
                        # SHIFT
                        else:
                            event = "SHIFT"

                        va_idx = int(start + silence/2)
                        va_step = int(self.sample_overlap*self.n_stride) 
                        wave_idx = int(va_idx*(sr/self.n_stride))
                        wave_step = int(self.sample_overlap*sr)

                        if va_idx + va_step <= va.shape[-1]:
                            yield {
                                "va": va[:, va_idx:va_idx+va_step+1],
                                "va_hist": va_hist[:, va_idx:va_idx+va_step+1],
                                "waveform": wf[:, wave_idx:wave_idx+wave_step+1],
                                "sample_rate": sr,
                                "event": event,
                                "pre_speaker": pre_speaker
                            }

    def generate_samples(self):
        step = self.sample_len - self.sample_overlap

        for dialogue_id in self.data:
            file = self._id_to_audio_filename(dialogue_id)

            # voice activity
            audio_len = get_audio_duration(
                os.path.join(self.audio_path, file)
            )
            va = self._get_activity(dialogue_id)
            va = activity_start_end_idx_to_onehot(
                va, audio_len, self.n_stride
            )
            va_sample = va.unfold(
                dimension=1,
                size=self.sample_len*self.n_stride,
                step=step*self.n_stride,
            )

            # voice activity history
            bins = [idx*self.n_stride for idx in self.va_hist_bins]
            va_hist = get_activity_history(va, bins)[0].permute(1, 0)
            va_hist_sample = va_hist.unfold(
                dimension=1,
                size=self.sample_len*self.n_stride,
                step=step*self.n_stride,
            )

            # waveform
            if self.load_audio:
                wf, sr = load_waveform(
                    os.path.join(self.audio_path, file),
                    **self.audio_kwargs
                )
                wf_sample = wf.unfold(
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
                if self.load_audio:
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
                        "file": os.path.join(self.audio_path, file),
                        "start": s_i*step
                    }

    def generate_random_samples(self):
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
                "sample_rate": (1, )
            }

        """
        # sample origins inside batch should vary between epochs
        random.shuffle(self.data)
        sampler = self.generate_samples()

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

            if "waveform" not in yield_item:
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
            dialogue_id (int):
                Identifier of one file in the corpus.

        Returns:
            list[list[list[int]]]: (N_Speakers, _, 2),
                where _ is a variable number of voice activities
                for speaker n each defined as the start and end
                timestamp of active speech in seconds.

        """
        pass

    def __iter__(self):
        """Dataset Iterable."""
        if self.mode == "train":
            for item in self.generate_random_samples():
                yield item
        elif self.mode == "valid":
            for item in self.generate_shift_hold_samples():
                yield item
        else:
            raise ValueError("Unknown mode.")


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
                        "[silence]", "[noise]", "<b_aside>", "<e_aside>"
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
