import numpy as np


class TemporalSequenceSampler:
    """
    Samples fixed-length temporal window sequences from variable-length video frames.
    """
    def __init__(self, sequence_length: int = 16, stride: int = 2, is_train: bool = True):
        self.sequence_length = sequence_length
        self.stride = stride
        self.is_train = is_train

    def sample_indices(self, total_frames: int) -> list:
        req_len = self.sequence_length * self.stride
        if total_frames >= req_len:
            start_idx = np.random.randint(0, total_frames - req_len + 1) if self.is_train else 0
            indices = list(range(start_idx, start_idx + req_len, self.stride))
        else:
            indices = np.linspace(0, total_frames - 1, self.sequence_length).astype(int).tolist()
        return indices
