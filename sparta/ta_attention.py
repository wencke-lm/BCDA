"""ta_attention.py - Time-Aware Attention Module"""
import torch
import torch.nn as nn


class TimeAwareAttention(nn.Module):
    """Attention for time-aware feature extraction.

    Learn the importance of contextual utterances based
    on their distance from the current position.

    As one moves deeper into the dialogue history
    the utterances influence on the current utterance reduces.
    So this attention scales-down the dot product between query
    and the key by a monotonically decreasing function of time.

    Based on:
    https://arxiv.org/abs/2111.06647

    Args:
        inp_dim (int)
        h_dim (int)

    """
    def __init__(self, inp_dim, h_dim):
        super().__init__()

        # linearly projection layers
        self.k = nn.Linear(inp_dim, h_dim, bias=False)
        self.q = nn.Linear(inp_dim, h_dim, bias=False)
        self.v = nn.Linear(inp_dim, h_dim, bias=False)

    def forward(self, current_utterance, memory, attention_mask=None, t=0.2):
        """Create contextualized embedding through attention weighting.

        Arg:
            current_utterance (torch.Tensor): (1, emb_size)
                Embedding of the current utterance.
            memory (torch.Tensor): (window_size, emb_size)
                Stores the embedding of k previous utterances,
                including the current utterance.
            attention_mask (torch.tensor): (window_size, )
                Which previous utterances to regard (value 1),
                and which to disregard (value 0).
            t (float):
                Temperature of the time-aware weighting.
                If low, the max value of weight will be high
                but the range from min to max weight low.
                Suggested t are between 0.10 and 0.30.

        Returns:
            tuple: (1, emb_size), (window_size, )
                Contextualized embedding and utterance relevance.

        """
        if len(memory.shape) == 2:
            # add batch dimension
            current_utterance.unsqueeze_(0)
            memory.unsqueeze_(0)

        window_size = memory.shape[1]

        key = self.k(memory)
        value = self.v(memory)
        query = self.q(current_utterance)

        # compute similarity function between utterances
        similarity = torch.cosine_similarity(key, query, dim=-1)
        # similarity.shape = ((batch_size,) window_size,)

        # get the time-aware distance weights
        weights = 1/torch.exp(
            torch.linspace(
                1, window_size, window_size,
                device=similarity.device
            )
        )**t
        # weights.shape = (window_size,)
        
        # weigh contribution per utterance
        similarity = similarity*weights
        # similarity.shape ((batch_size,) window_size,)

        # mask the not useful context
        if attention_mask is not None:
            similarity = similarity.masked_fill(
                attention_mask == 0, -1e10
            )

        # compute proportional relevance per utterance
        relevance = similarity.softmax(dim=-1)
        # relevance.shape ((batch_size,) window_size,)

        # get relevance based contextual embedding
        x = torch.einsum('bj,bjk->bk', relevance, value)
        # x.shape ((batch_size,) h_dim)

        return x, relevance
