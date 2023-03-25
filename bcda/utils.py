"""utils.py - Data Preprocessing"""
from decimal import *

import torch


class BCDataset(torch.utils.data.IterableDataset):
    """Load backchannel samples.

    Args:
        corpus (vap.vap_loader.ShuffledIterableDataset):
            Dialogue data loader.
        filename (str):
            Path to plain text-file holding backchannel sample
            information. See data/swb/utterance_is_backchannel.csv
        encoding (str):
            File encoding. Defaults to utf-8.
    """
    def __init__(self, corpus, filename, encoding="utf-8"):
        super().__init__()

        self.corpus = corpus
        self.filename = filename
        self.encoding = encoding

    def __iter__(self):
        return self.load_bc_samples()

    def load_bc_samples(self, encoding="utf-8"):
        """Load backchannel samples.

        Yields:
            tuple: (Input Features, BC Speaker, BC Type).

        """
        prev_diag_id = None
        time_stamps = []
        add_info = []

        va_mask = None
        wave_mask = None

        with open(self.filename, "r", encoding=encoding) as file:
            sample = file.readline()
            while sample:
                info, time_stamp, bc_type = sample.rstrip().split("\t")
                diag_id, bc_speaker = info[2:-1], info[-1]

                sample = file.readline()

                # load all samples belonging to one dialogue together
                if time_stamps and (prev_diag_id != diag_id or not sample):
                    for i, s in enumerate(
                        self.corpus.select_samples(
                            prev_diag_id, time_stamps, test=True
                        )
                    ):
                        if va_mask is None:
                            va_mask = torch.zeros(s["va"].shape[-1])
                            wave_mask = torch.zeros(s["waveform"].shape[-1])

                        s["speakers"], s["labels"] = add_info[i]
                        # generate padding mask
                        s["masks"] = va_mask == 0
                        s["masks"][-s["va"].shape[-1]:] = False

                        # pad input
                        if True in s["masks"]:
                            s["va"] = torch.nn.functional.pad(
                                s["va"],
                                (va_mask.shape[-1]-s["va"].shape[-1], 0),
                            )
                            s["va_hist"] = torch.nn.functional.pad(
                                s["va_hist"],
                                (va_mask.shape[-1]-s["va_hist"].shape[-1], 0),
                            )
                            s["waveform"] = torch.nn.functional.pad(
                                s["waveform"],
                                (wave_mask.shape[-1]-s["waveform"].shape[-1], 0),
                            )
                        yield s
                    time_stamps = []
                    add_info = []


                if diag_id in self.corpus.split:
                    prev_diag_id = diag_id
                    time_stamps.append(Decimal(time_stamp))
                    add_info.append((bc_speaker, bc_type))

        print(time_stamps)