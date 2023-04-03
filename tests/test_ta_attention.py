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
        assert out[1].shape == torch.Size([1, 2])

    def test_time_aware_attention_same_result_single_batch(self):
        attention = TimeAwareAttention(4, 3)
        memory_single = torch.tensor([
            [1., 2., 0., 1.],
            [1., 1., 0., 0.]
        ])
        memory_single2 = torch.tensor([
            [1., 2., 0., 1.],
            [1., 2., 0., 1.]
        ])
        memory_batch = torch.tensor([[
            [1., 2., 0., 1.],
            [1., 1., 0., 0.]
        ],[
            [1., 2., 0., 1.],
            [1., 2., 0., 1.]
        ]])

        out_single = attention(memory_single[:1], memory_single)
        out_single2 = attention(memory_single2[:1], memory_single2)

        out_batch = attention(memory_batch[:, :1], memory_batch)

        assert out_batch[0].shape == torch.Size([2, 3])
        assert out_batch[1].shape == torch.Size([2, 2])

        torch.testing.assert_close(out_single[0][0], out_batch[0][0])
        torch.testing.assert_close(out_single[1][0], out_batch[1][0])
        torch.testing.assert_close(out_single2[0][0], out_batch[0][1])
        torch.testing.assert_close(out_single2[1][0], out_batch[1][1])
