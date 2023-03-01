"""events.py - Functionality to extract turn-taking events."""
import torch


def is_shift_or_hold(sample, silence_frms, offset_frms):
    """TODO"""
    va_frames = sample["va"].shape[-1]

    # mutual silence
    pre_speech = torch.nonzero(sample["va"].permute(1, 0).flatten())
    if pre_speech.nelement() == 0:
        silence_start = 0
    else:
        silence_start = torch.div(pre_speech[-1], 2, rounding_mode='floor')

    post_speech = torch.nonzero(sample["labels"].permute(1, 0).flatten())
    if post_speech.nelement() == 0:
        silence_end = va_frames - 1
    else:
        silence_end = torch.div(post_speech[0], 2, rounding_mode='floor')

    if (
        va_frames - offset_frms >= va_frames - silence_start+1 >= silence_frms
        and offset_frms >= silence_end+1 >= silence_frms
    ):
        # pre-offset only one speaker active
        # post-offset only one speaker active
        pre_who_speaks = (
            sample["va"][:, silence_start+1-offset_frms:].sum(dim=-1)
        )
        post_who_speaks = (
            sample["labels"][:, :silence_end+offset_frms].sum(dim=-1)
        )
        if (
            torch.count_nonzero(pre_who_speaks) == 1
            and torch.count_nonzero(post_who_speaks) == 1
        ):
            pre_speaker = int(torch.nonzero(pre_who_speaks))
            post_speaker = int(torch.nonzero(post_who_speaks))

            # HOLD
            if pre_speaker == post_speaker:
                return "HOLD", pre_speaker
            # SHIFT
            return "SHIFT", post_speaker
