from collections import deque
from time import monotonic


class TemporalPERCLOSBuffer:
    """Track closed-eye frames and require sustained danger before alarming."""

    def __init__(
        self,
        window_size=60,
        warning_threshold=0.40,
        danger_threshold=0.70,
        danger_duration_seconds=1.5,
        clock=monotonic,
    ):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0.0 <= warning_threshold <= danger_threshold <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= warning <= danger <= 1")
        if danger_duration_seconds < 0.0:
            raise ValueError("danger_duration_seconds must be non-negative")

        self.window_size = window_size
        self.warning_threshold = warning_threshold
        self.danger_threshold = danger_threshold
        self.danger_duration_seconds = danger_duration_seconds
        self._clock = clock
        self._closed_frames = deque(maxlen=window_size)
        self._danger_started_at = None
        self.state = "ALERT"

    def reset(self):
        self._closed_frames.clear()
        self._danger_started_at = None
        self.state = "ALERT"

    def update(self, is_closed):
        self._closed_frames.append(bool(is_closed))
        now = self._clock()
        perclos = sum(self._closed_frames) / len(self._closed_frames)

        if perclos >= self.danger_threshold and len(self._closed_frames) == self.window_size:
            if self._danger_started_at is None:
                self._danger_started_at = now
            danger_duration = now - self._danger_started_at
            self.state = "DANGER" if danger_duration >= self.danger_duration_seconds else "WARNING"
        elif perclos >= self.warning_threshold:
            self._danger_started_at = None
            danger_duration = 0.0
            self.state = "WARNING"
        else:
            self._danger_started_at = None
            danger_duration = 0.0
            self.state = "ALERT"

        return {
            "perclos": perclos,
            "state": self.state,
            "buffer_fill": len(self._closed_frames),
            "danger_duration_seconds": max(0.0, danger_duration),
        }