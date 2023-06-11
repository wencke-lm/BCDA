"""bpm_utils.py - BPM Baseline Dataloader"""
import os
import random

import librosa
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import BertTokenizer


class BCDataset(torch.utils.data.IterableDataset):
    # text tokenizer for bc model preprocessing
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased"
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[S]", "[L]"]}
    )

    # sentiment soft label extractor
    sent_tokenizer = AutoTokenizer.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    sent_model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment-latest"
    ).to("cuda:0" if torch.cuda.is_available() else "cpu")
    for p in sent_model.parameters():
        # this model should be permanently frozen
        p.requires_grad_(False)

    def __init__(self, split_info, text_path, audio_path):
        """Dataset loader.

        Args:
            split_info (str/list):
                Path to an utf8 encoded txt-file holding one
                conversation identifer per line or
                collection of relevant conversation identifiers.
            text_path (str):
                Path to a utf8-encoded file holding sample info.
                The following tab-separated info is expected:
                    + dialogID with BC_speaker
                    + timestamp
                    + BC_type
                    + utterance history
            audio_path (str):
                Path to a folder holding audio files.
                One audio file per dialogue.

        """
        super().__init__()

        # data sources
        if isinstance(split_info, str):
            with open(split_info, 'r', encoding="utf-8") as data:
                self.split_info = data.read().strip().split("\n")
        else:
            self.split_info = split_info
        self.text_path = text_path
        self.audio_path = audio_path

    @classmethod
    def tokenize(cls, utt):
        """Prepare raw text data to be suitable model input.

        Args:
            utt (str/list): Text instance(s).

        Returns:
            dict: dict_keys(['input_ids', 'attention_mask'])
                Each value of shape (N_Inputs, 128), where 128
                is the max length of the model and longer
                strings are truncated.

        """
        return cls.tokenizer(
            utt, return_tensors='pt',
            padding="max_length", max_length=256, truncation=True
        )

    @classmethod
    def get_sentiment_score(cls, utt):
        """Assign soft sentiment label to text utterance.

        Args:
            utt (str/list): Text instance(s).

        Returns:
            torch.tensor: (N, 3)
                N: Number of text instances

        """
        encoded_input = cls.sent_tokenizer(
            utt, return_tensors='pt',
            padding="max_length", max_length=256, truncation=True
        ).to("cuda:0" if torch.cuda.is_available() else "cpu")
        output = cls.sent_model(**encoded_input)

        return output[0].softmax(1).squeeze(0)

    @staticmethod
    def get_mfcc_feature(audio_path, end, window_len=25, step_len=10, channel=None):
        """Extract Mel-frequency Cepstrum Coefficients.

        Args:
            audio_path (str): Path to audio file.
            end (int): The end of the extraction interval in s.
            window_len (int): Pooled audio signal per MFCC vector in ms.
            step_len(int): Step width between consecutive windows.

        Returns:
            tuple: (N, F)
              N: Number of frames (depends on audio-, window-, step-len)
              F: Number of MFCC coefficients (= 13)

        """
        if channel is None:
            audio, sr = librosa.load(
                audio_path, offset=end-1.5, duration=1.5, mono=True
            )
        else:
            audio, sr = librosa.load(
                audio_path, offset=end-1.5, duration=1.5, mono=False
            )
            audio = audio[channel]

        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=13, center=False,
            n_fft=sr//1000*window_len, hop_length=sr//1000*step_len
        )
        return torch.from_numpy(mfcc).permute(1, 0)

    def _preprocess(self, line):
        """Prepare relevant input features."""
        # info, time_stamp, bc_type, *hist = line.rstrip("\n").split("\t")
        (
            info, time_stamp, bc_type, neg_sent, neut_sent, pos_sent, *hist
        ) = line.rstrip("\n").split("\t")
        diag_id, bc_speaker = info[2:-1], info[-1]

        # feature preprocessing
        # sent_feat = self.get_sentiment_score(hist[0])
        sent_feat = torch.tensor(
            [float(neg_sent), float(neut_sent), float(pos_sent)]
        )

        text = []
        for t in hist[:3]:
            if t.startswith(hist[0][:4]):
                text.append(t[4:])
            else:
                break
        text = hist[0][:4] + " ".join(reversed(text))

        text_feat = self.tokenize(text)
        audio_feat = self.get_mfcc_feature(
            os.path.join(self.audio_path, f"sw0{diag_id}.sph"),
            float(time_stamp),
            channel={"A": 1, "B": 0}[bc_speaker]
        )

        return {
            "input_ids": text_feat["input_ids"].squeeze(0),
            "attention_mask": text_feat["attention_mask"].squeeze(0),
            "acoustic_input": audio_feat,
            "main_labels": bc_type,
            "sub_labels": sent_feat
        }

    def _load_bc_samples(self):
        """Create pseudo-random data samples."""
        with open(self.text_path, "r", encoding="utf-8") as file:
            # memory for pseudo-random sampling
            memory = []
            for i, line in enumerate(file, 1):
                if len(memory) % 500 == 0:
                    random.shuffle(memory)
                    for sample in memory:
                        # create sample on demand
                        yield self._preprocess(sample)
                    memory = []
                if line[2:6] in self.split_info:
                    memory.append(line)

            random.shuffle(memory)
            for sample in memory:
                yield self._preprocess(sample)

    def __iter__(self):
        return self._load_bc_samples()
