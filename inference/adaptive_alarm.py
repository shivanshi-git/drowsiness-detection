import time
import collections
import numpy as np


class AdaptiveAlarmSystem:
    """
    Simplified Drowsiness Alarm System.

    Rules:
      - Eyes closed >= 7 seconds → CRITICAL alarm (Drowsy)
      - Yawn count tracks total yawns detected this session
      - Fatigue score drives early-warning tiers below eye-closure threshold

    Tiers:
      0: Attentive (Green)
      1: Caution – early fatigue signs or >=2 yawns (Yellow)
      2: Warning  – high fatigue probability         (Orange)
      3: CRITICAL – eyes closed 7+ seconds           (Red, Alarm Sound)
    """

    # Eye-closure duration that triggers a Drowsy alarm (seconds)
    EYE_CLOSURE_ALARM_SEC = 7.0

    # A yawn event is registered when the mouth-open (MAR) state is held for
    # this many seconds, to avoid double-counting a single yawn.
    YAWN_MIN_DURATION_SEC = 1.0

    def __init__(
        self,
        smoothing_window: int = 10,
        level_1_thresh: float = 0.45,
        level_2_thresh: float = 0.65,
        perclos_window_sec: float = 60.0,
        perclos_drowsy_thresh: float = 0.15,
    ):
        self.smoothing_window = smoothing_window
        self.level_1_thresh = level_1_thresh
        self.level_2_thresh = level_2_thresh
        self.perclos_window_sec = perclos_window_sec
        self.perclos_drowsy_thresh = perclos_drowsy_thresh

        # Rolling score history
        self.prob_history = collections.deque(maxlen=smoothing_window)

        # PERCLOS rolling buffer: list of (timestamp, is_closed)
        self.perclos_history = collections.deque()

        # Eye-closure tracking
        self.eye_closure_start_time = None

        # Yawn tracking
        self.yawn_count = 0
        self.yawn_active = False          # True while mouth is currently open
        self.yawn_start_time = None       # When the current open-mouth event started

        # Sound throttle
        self.last_alarm_time = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        raw_drowsy_prob: float,
        predicted_class: int,
        fps: float = 30.0,
        ear: float = 0.30,
        mar: float = 0.15,
    ) -> dict:
        """
        Update internal state and return alarm action dict.

        Args:
            raw_drowsy_prob : float in [0, 1] – model drowsiness confidence
            predicted_class : int – class index from the model
            fps             : current stream FPS
            ear             : Eye Aspect Ratio  (< 0.16 → closed)
            mar             : Mouth Aspect Ratio (> 0.50 → yawning)
        """
        now = time.time()

        # ── Smooth probability ────────────────────────────────────────
        self.prob_history.append(raw_drowsy_prob)
        smoothed_prob = float(np.mean(self.prob_history))

        # ── Eye closure detection & duration ─────────────────────────
        # Eyes are considered closed when EAR is very low OR model says so
        is_eye_closed = (ear < 0.16) or (predicted_class == 4) or (raw_drowsy_prob > 0.80)

        if is_eye_closed:
            if self.eye_closure_start_time is None:
                self.eye_closure_start_time = now
            closure_duration = now - self.eye_closure_start_time
        else:
            self.eye_closure_start_time = None
            closure_duration = 0.0

        # ── PERCLOS ───────────────────────────────────────────────────
        self.perclos_history.append((now, 1.0 if is_eye_closed else 0.0))
        cutoff = now - self.perclos_window_sec
        while self.perclos_history and self.perclos_history[0][0] < cutoff:
            self.perclos_history.popleft()
        perclos = float(np.mean([v for _, v in self.perclos_history])) if self.perclos_history else 0.0

        # ── Yawn counting ─────────────────────────────────────────────
        # Yawn starts when MAR > 0.50 (mouth wide open)
        is_mouth_open = mar > 0.50

        if is_mouth_open:
            if not self.yawn_active:
                # New mouth-open event – start timing it
                self.yawn_active = True
                self.yawn_start_time = now
        else:
            if self.yawn_active:
                # Mouth just closed – check whether it was long enough to be a yawn
                duration = now - (self.yawn_start_time or now)
                if duration >= self.YAWN_MIN_DURATION_SEC:
                    self.yawn_count += 1
                self.yawn_active = False
                self.yawn_start_time = None

        # ── Tier decision ─────────────────────────────────────────────
        alarm_level = 0
        status_msg  = "Attentive"
        hud_color   = (0, 255, 0)    # Green  (BGR)
        trigger_sound = False

        if closure_duration >= self.EYE_CLOSURE_ALARM_SEC:
            alarm_level   = 3
            status_msg    = "DROWSY: EYES CLOSED – WAKE UP!"
            hud_color     = (0, 0, 255)   # Red
            trigger_sound = True

        elif smoothed_prob >= self.level_2_thresh or perclos >= self.perclos_drowsy_thresh:
            alarm_level = 2
            status_msg  = "WARNING: Drowsiness Detected"
            hud_color   = (0, 140, 255)   # Orange
            trigger_sound = True

        elif smoothed_prob >= self.level_1_thresh or self.yawn_count >= 2:
            alarm_level = 1
            status_msg  = f"CAUTION: Fatigue Signs (Yawns: {self.yawn_count})"
            hud_color   = (0, 255, 255)   # Yellow

        # ── Audible alert (throttled) ─────────────────────────────────
        if trigger_sound and (now - self.last_alarm_time > 1.5):
            self._sound_alert(alarm_level)
            self.last_alarm_time = now

        return {
            "alarm_level":          alarm_level,
            "status_text":          status_msg,
            "hud_color":            hud_color,
            "smoothed_fatigue_score": smoothed_prob,
            "perclos":              perclos,
            "closure_duration":     closure_duration,
            "predicted_class":      predicted_class,
            "yawn_count":           self.yawn_count,
        }

    def reset_yawn_count(self):
        """Manually reset the yawn counter (e.g. at start of new trip)."""
        self.yawn_count = 0
        self.yawn_active = False
        self.yawn_start_time = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sound_alert(self, level: int):
        """Synthesise audible alarm (Windows only, non-blocking)."""
        try:
            import winsound
            if level == 3:
                winsound.Beep(2500, 400)
            elif level == 2:
                winsound.Beep(1200, 200)
        except Exception:
            pass  # Headless / non-Windows – skip silently
