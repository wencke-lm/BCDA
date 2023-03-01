import torch

from vap.events import is_shift_or_hold


class TestMetrics:
    def test_is_shift_or_hold_minimal_silence(self):
        # shift
        va = torch.zeros((2, 800))
        va[0, -104:-4] = 1
        labels = torch.zeros((2, 200))
        labels[1, 5:105] = 1
        sample = {"va": va, "labels": labels}

        assert is_shift_or_hold(sample, 5, 100) == ("SHIFT", 1)

    def test_is_shift_or_hold_minimal_offset(self):
        # hold
        va = torch.zeros((2, 800))
        va[0, 670:770] = 1
        va[1, :670] = 1
        labels = torch.zeros((2, 200))
        labels[0, 20:120] = 1
        labels[1, 120:150] = 1
        sample = {"va": va, "labels": labels}

        assert is_shift_or_hold(sample, 5, 100) == ("HOLD", 0)

    def test_is_shift_or_hold_no_speech(self):
        # no turn taking event
        va = torch.zeros((2, 800))
        labels = torch.zeros((2, 200))
        sample = {"va": va, "labels": labels}

        assert is_shift_or_hold(sample, 5, 100) is None

    def test_is_shift_or_hold_overlaping_past_speech(self):
        # no turn taking event
        va = torch.zeros((2, 800))
        va[0, -150:-50] = 1
        va[1, :-75] = 1
        labels = torch.zeros((2, 200))
        labels[1, 75:125] = 1
        sample = {"va": va, "labels": labels}

        assert is_shift_or_hold(sample, 5, 100) is None

    def test_is_shift_or_hold_overlaping_upcoming_speech(self):
        # no turn taking event
        va = torch.zeros((2, 800))
        va[0, -150:-50] = 1
        labels = torch.zeros((2, 200))
        labels[0, 50:100] = 1
        labels[1, 75:125] = 1
        sample = {"va": va, "labels": labels}

        assert is_shift_or_hold(sample, 5, 100) is None

    def test_is_shift_or_hold_no_mutual_silence(self):
        # no turn taking event
        va = torch.zeros((2, 800))
        va[0, :] = 1
        labels = torch.zeros((2, 200))
        labels[1, :] = 1
        sample = {"va": va, "labels": labels}

        assert is_shift_or_hold(sample, 5, 100) is None
