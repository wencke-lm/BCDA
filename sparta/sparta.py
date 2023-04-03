import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel

from sparta.ta_attention import TimeAwareAttention


class SpartaModel(nn.Module):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    # add special tokens for speaker embedding
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ["[A]", "[B]"]}
    )

    def __init__(self, dim_out):
        super().__init__()

        # large pretrained language model
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.set_llm_mode(True)
        # feature space projection layer
        self.fc = nn.Linear(768, dim_out)

        self.ta_attention = TimeAwareAttention(768, 768)

    def forward(self, input_ids, attention_mask):
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
        context_emb, _ = self.ta_attention(hist[:, :1], hist)
        # context_emb.shape (batch_size, 768)
        reduced_context_emb = self.fc(context_emb)
        # reduced_context_emb.shape (batch_size, 256)

        return reduced_context_emb

    def set_llm_mode(self, freeze):
        """Freeze or unfreeze language model parameters.

        Args:
            freeze (bool):
                Whether to freeze/not update the 
                parameters during end-to-end training.

        """
        for p in self.model.parameters():
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
