import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel


class SpartaModel(nn.Module):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def __init__(self):
        super().__init__()

        self.model = RobertaModel.from_pretrained('roberta-base')

    def forward(self, encoded_inp):
        out = self.model(**encoded_inp)
        last_hidden_state = out[0]
        cls_embedding = last_hidden_state[:, 0, :]

        return cls_embedding

    @classmethod
    def tokenize(cls, inp):
        return tokenizer(inp, return_tensors='pt', padding=True, truncation=True)
