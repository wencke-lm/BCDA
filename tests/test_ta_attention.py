import torch

from sparta.ta_attention import TimeAwareAttention


class TestTimeAwareAttention:
    def test_time_aware_attention(self):
        attention = TimeAwareAttention(4, 3)
        memory = torch.tensor([
            [1., 2., 0., 1.],
            [1., 1., 0., 0.]
        ])

        out = attention(memory[:1], memory)

        assert out[0].shape == torch.Size([1, 3])
        assert out[1].shape == torch.Size([2])
