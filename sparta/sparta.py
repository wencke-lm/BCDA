import torch.nn as nn
from transformers import BertTokenizer, BertModel

from sparta.ta_attention import TimeAwareAttention


class SpartaModel(nn.Module):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # add special tokens for speaker embedding
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ["[S]", "[L]"]}
    )

    def __init__(self, dropout, hist_n):
        super().__init__()

        self.hist_n = hist_n
        # large pretrained language model
        self.model = BertModel.from_pretrained(
                "bert-base-uncased",
                hidden_dropout_prob=0.3,
                attention_probs_dropout_prob=0.3
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.set_llm_mode(True)

        self.ta_attention = TimeAwareAttention(768, 768)

    def forward(self, input_ids, attention_mask):
        if len(input_ids.shape) < 3:
            input_ids = input_ids.unsqueeze(0)
        if len(attention_mask.shape) < 3:
            attention_mask = attention_mask.unsqueeze(0)

        batch_size, hist_len, _ = input_ids.shape

        out = self.model(
            input_ids=input_ids.flatten(start_dim=0, end_dim=1),
            attention_mask=attention_mask.flatten(start_dim=0, end_dim=1)
        )
        last_hidden_state = out[0]
        # last_hidden_state.shape (batch_size*hist_len, max_len, 768)
        cls_embedding = last_hidden_state[:, 0, :]
        # cls_embedding.shape (batch_size*hist_len, 768)

        hist = cls_embedding.unflatten(0, (batch_size, hist_len))
        # hist.shape (batch_size, hist_len, 768)
        context_emb, _ = self.ta_attention(hist[:, :1], hist[:, :self.hist_n])
        # context_emb.shape (batch_size, 768)

        return context_emb

    def set_llm_mode(self, freeze):
        """Freeze or unfreeze language model parameters.

        Args:
            freeze (bool):
                Whether to freeze/not update the 
                parameters during end-to-end training.

        """
        for name, p in self.model.named_parameters():
            p.requires_grad_(not freeze)

    @classmethod
    def tokenize(cls, inp):
        """Prepare raw text data to be suitable model input.

        Args:
            str/list:
                Text string(s) to be tokenized.

        Returns:
            dict: dict_keys(['input_ids', 'attention_mask'])
                Each value of shape (N_Inputs, 512), where 512
                is the max length of the model and longer
                strings are truncated.

        """
        return cls.tokenizer(
            inp, return_tensors='pt',
            padding="max_length", max_length=150, truncation=False
        )
