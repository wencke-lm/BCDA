"""events.py - Turn Taking Event Extraction"""
import torch


def is_shift_or_hold(sample, silence_frms, offset_frms):
    """Determine turn taking event present in sample.

    Task used to evaluate how well the model predicts the
    next speaker at a point of mutual silence.

    Args:
        sample (dict):
            {
                "va": (N_Speakers, L_Strides),
                "va_hist": (M_Bins, L_Strides),
                "waveform": (N_Speakers, K_Frames)
            }
            As returned by data_loader.ShuffledIterableDataset
        silence_frms (int):
            Duration of mutual silence in number of frames.
        offset_frms (int):
            Areas around mutual silence where only a single
            speaker can be active.

    Returns:
        tuple or None:
            If turn taking returns (event_name, next_speaker_id).
            Else returns None.

    """
    va_frames = sample["va"].shape[-1]

    pre_speech = torch.nonzero(sample["va"].permute(1, 0).flatten())
    post_speech = torch.nonzero(sample["labels"].permute(1, 0).flatten())

    if pre_speech.nelement() == 0:
        silence_start = 0
    else:
        silence_start = int(pre_speech[-1]) // 2

    if post_speech.nelement() == 0:
        silence_end = va_frames - 1
    else:
        silence_end = int(post_speech[0]) // 2

    # mutual silence
    if (
        va_frames - offset_frms >= va_frames - silence_start+1 >= silence_frms
        and offset_frms >= silence_end+1 >= silence_frms
    ):
        pre_who_speaks = (
            sample["va"][:, silence_start+1-offset_frms:].sum(dim=-1)
        )
        post_who_speaks = (
            sample["labels"][:, :silence_end+offset_frms].sum(dim=-1)
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
                return "HOLD", pre_speaker
            # SHIFT
            return "SHIFT", pre_speaker
    return None, None
