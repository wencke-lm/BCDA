"""utils.py - Data Preprocessing"""
import torch
import torch.nn.functional as F
import torchaudio


def time2frames(time, frame_per_s):
    return int(time * frame_per_s)


def get_audio_duration(audio_path):
    info = torchaudio.info(audio_path)
    return info.num_frames / info.sample_rate


def normalize_audio(audio, audio_normalize_threshold):
    """Inplace audio amplitude normalization."""
    max_ampl = audio.abs().max(dim=1, keepdim=True)[0]
    # only normalize channels with peaks above the threshold
    audio /= torch.where(
        max_ampl > audio_normalize_threshold, max_ampl, 1
    )


def load_waveform(
    path,
    start_time=None,
    end_time=None,
    sample_rate=None,
    mono=False,
    normalize=False,
    audio_normalize_threshold=0.05,
):
    """Loads an audio file from disk.

    Arguments:
        path (str):
            Path to audio file.
        start_time (float):
            Number of s from start of audio to begin loading.
            If None start from 0 seconds. Defaults to None.
        end_time (float):
            Number of s from start of audio to end loading.
            If None load until end of file. Defaults to None.
        sample_rate (int):
            Number of samples taken from the waveform per s.
            If None use sample rate that audio was recorded at.
            Defaults to None.
        mono (bool):
            If True merge all audio channels into a single
            averaged channel. Defaults to False.
        normalize (False):
            If True scale waveform amplitude so that the highest
            peak value equals 1. Defaults to False.
        audio_normalize_threshold (float):
            Only normalize if the peak surpasses this value.
            Effectivly stops the amplification of silence.
            Defaults to 0.05.

    Returns:
        tuple[tensor, int]:
            Audio_Tensor (N_Channels, K_Frames), Sample_Rate

    """
    info = torchaudio.info(path)
    start_frame = 0
    n_frames = info.num_frames

    if end_time is not None:
        n_frames = time2frames(end_time, info.sample_rate)
    if start_time is not None:
        start_frame = time2frames(start_time, info.sample_rate)
        n_frames -= start_frame

    audio, sr = torchaudio.load(
        path, frame_offset=start_frame, num_frames=n_frames
    )

    if normalize:
        normalize_audio(audio, audio_normalize_threshold)

    if mono and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
        if normalize:
            normalize_audio(audio, audio_normalize_threshold)

    if sample_rate is not None and sr != sample_rate:
        audio = torchaudio.functional.resample(
            audio, orig_freq=sr, new_freq=sample_rate
        )
        sr = sample_rate

    return audio, sr


def activity_start_end_idx_to_onehot(activity_idx, duration, n_stride):
    """Unfold voice activity information across time.

    Arguments:
        activity_idx (list[list[list[int]]]): (N_Speakers, _, 2),
            where _ is a variable number of voice activities
            for speaker n each defined as the start and end
            timestamp of active speech in seconds.
        duration (float):
            Length of audio that activity was extracted from in s.
        n_stride (float):
            Number of discrete activity values to save per s.

    Returns:
        torch.tensor: (N_Speakers, K_Frames)

    Example:
        >>> va_idx = [
            [[0.5, 1], [2, 3]],
            [[1, 2.5]]
        ]
        >>> activity_start_end_idx_to_onehot(va_idx, 4.0, 0.5)
        tensor([
            [0, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0]
        ])

    """
    onehot_va = torch.zeros(
        len(activity_idx), time2frames(duration, n_stride)
    )

    for i, speaker in enumerate(activity_idx):
        for start, end in speaker:
            start = time2frames(start, n_stride)
            end = time2frames(end, n_stride)

            onehot_va[i, start:end] = 1

    return onehot_va


def get_activity_history(activity, bin_end_idx, n_step=None):
    """Concise representation of voice activity history.

    At each point in time/frame we represent the previous utterance
    history as the relative participation of each speaker during
    custom time intervals/regions.

    Arguments:
        activity (torch.tensor): (N_Speakers, K_Frames)
            1 if speaker n is speaking at time point k. 0 else.
        bin_end_idx (list[int]):
            Right boundaries of regions for which the
            ratio of speaker's activity shall be calculated.
        n_step (int):
            Number of frames for which history should be calculated.
            e.g if n_step=2 calculate history for frame k and k-1
            If None calculate for every frame. Defaults to None.

    Returns:
        torch.tensor: (N_Speakers, K_Frames, M_Bins)

    Example:
        >>> va = torch.tensor([
            [1, 0, 0, 1, 1, 0],
            [1, 1, 0, 0, 1, 1]
        ])
        >>> bins = [3, 1, 0]  # 3 intervals (-inf, 3], (3, 1], (1, 0]
        >>> get_activity_history(va, bins)
        tensor([
        [[0.5000, 0.5000, 0.5000],
         [0.5000, 0.5000, 0.0000],
         [0.5000, 0.3333, 0.5000],
         [0.5000, 0.0000, 1.0000],
         [0.3333, 1.0000, 0.5000],  # history of speaker A at frame k-1
         [0.3333, 0.6667, 0.0000]],  # history of speaker A at frame k

        [[0.5000, 0.5000, 0.5000],
         [0.5000, 0.5000, 1.0000],
         [0.5000, 0.6667, 0.5000],
         [0.5000, 1.0000, 0.0000],
         [0.6667, 0.0000, 0.5000],
         [0.6667, 0.3333, 1.0000]]  # history of speaker B at frame k
        ])

    """
    # (N_Speakers, K_Frames, M_Bins)
    # object used to save the total number of frames a speaker n has spoken
    # during the interval m calculated with each current frame k as offset
    # e.g. if the current frame is 3, then the intervall [3, 1)
    # adds all activity from frame 6 to 5 (included)
    n_speakers, n_frames = activity.shape
    hist = torch.ones(
        n_speakers, n_step or n_frames, len(bin_end_idx)
    ) * -1

    # for the left-most bin starting at -inf this is the cumulative sum
    cum_sum = activity[:, :-bin_end_idx[0]].cumsum(dim=-1)

    if n_step is None:
        hist[:, bin_end_idx[0]:, 0] = cum_sum
    else:
        hist[:, max(n_step - bin_end_idx[0], 0):, 0] = cum_sum[:, -n_step:]

    # else it is the sum of all values inside an interval sized moving window
    for i, (start, end) in enumerate(zip(bin_end_idx, bin_end_idx[1:]), 1):
        # size of the interval/window
        ws = start - end

        # padding left so that every frame is once at the right edge of window
        # so we have as many individual windows as frames
        va = F.pad(activity, [ws - 1, 0])

        # skip last frames not part of the history of the current frame
        if end > 0:
            va = va[:, :-end]

        # reduce number of windows by trunc possible leftward window movements
        if n_step is not None:
            va = va[:, -(ws + n_step - 1):]

        # implicit loop over all windows
        filters = torch.ones((1, 1, ws), dtype=va.dtype)
        window_out = F.conv1d(
            va.unsqueeze(1), weight=filters
        ).squeeze(1)
        hist[:, -window_out.shape[1]:, i] = window_out

    assert torch.all(hist >= -1)

    ratios = hist / hist.sum(dim=0)
    # segments where both speakers are silent result in division by zero (nan)
    # define those segments as having equal ratios across speakers
    equal_ratio = 1 / ratios.shape[0]
    ratios = torch.where(ratios.isnan(), equal_ratio, ratios)

    return ratios
