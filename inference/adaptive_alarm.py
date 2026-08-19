import time
import collections
import numpy as np


class AdaptiveAlarmSystem:
    """
    Intelligent Multi-Tier Adaptive Alarm & Fatigue Monitor.
    
    Tiers:
      0: Normal / Attentive (Green)
      1: Visual Notice (Yellow) - Early signs of sluggishness or yawning
      2: Caution Chime (Orange) - Sustained slow blinking / high fatigue
      3: Critical Siren (Flashing Red) - Prolonged eye closure / microsleep event
    """
    def __init__(
        self,
        smoothing_window: int = 10,
        level_1_thresh: float = 0.45,
        level_2_thresh: float = 0.65,
        level_3_thresh: float = 0.85,
        perclos_window_sec: float = 60.0,
        perclos_drowsy_thresh: float = 0.15,
        consecutive_closure_sec: float = 1.5
    ):
        self.smoothing_window = smoothing_window
        self.level_1_thresh = level_1_thresh
        self.level_2_thresh = level_2_thresh
        self.level_3_thresh = level_3_thresh
        self.perclos_window_sec = perclos_window_sec
        self.perclos_drowsy_thresh = perclos_drowsy_thresh
        self.consecutive_closure_sec = consecutive_closure_sec

        # Rolling history
        self.prob_history = collections.deque(maxlen=smoothing_window)
        self.perclos_history = collections.deque() # (timestamp, is_closed)
        self.eye_closure_start_time = None
        self.last_alarm_time = 0.0

    def update(self, raw_drowsy_prob: float, predicted_class: int, fps: float = 30.0) -> dict:
        """
        Updates internal fatigue states and returns alarm action recommendations.
        Args:
            raw_drowsy_prob: float in [0, 1]
            predicted_class: 0: Normal, 1: Slow Blink, 2: Yawn, 3: Nod, 4: Eye Closure
            fps: Current stream frames per second
        Returns:
            dict containing alert status, level, smoothed score, and PERCLOS.
        """
        now = time.time()
        self.prob_history.append(raw_drowsy_prob)
        smoothed_prob = float(np.mean(self.prob_history))

        # Check eye closure state (Class 4 or High prob)
        is_closed = (predicted_class == 4) or (raw_drowsy_prob > 0.80)

        # Track consecutive eye closure duration
        if is_closed:
            if self.eye_closure_start_time is None:
                self.eye_closure_start_time = now
            closure_duration = now - self.eye_closure_start_time
        else:
            self.eye_closure_start_time = None
            closure_duration = 0.0

        # Update PERCLOS rolling buffer (remove events older than window)
        self.perclos_history.append((now, 1.0 if is_closed else 0.0))
        cutoff = now - self.perclos_window_sec
        while self.perclos_history and self.perclos_history[0][0] < cutoff:
            self.perclos_history.popleft()

        perclos = float(np.mean([val for _, val in self.perclos_history])) if self.perclos_history else 0.0

        # Tier Decision Logic
        alarm_level = 0
        status_msg = "Attentive"
        hud_color = (0, 255, 0) # Green (BGR)
        trigger_sound = False

        if closure_duration >= self.consecutive_closure_sec or smoothed_prob >= self.level_3_thresh:
            alarm_level = 3
            status_msg = "CRITICAL: WAKE UP!"
            hud_color = (0, 0, 255) # Red
            trigger_sound = True
        elif perclos >= self.perclos_drowsy_thresh or smoothed_prob >= self.level_2_thresh:
            alarm_level = 2
            status_msg = "WARNING: Drowsiness Detected"
            hud_color = (0, 140, 255) # Orange
            trigger_sound = True
        elif smoothed_prob >= self.level_1_thresh or predicted_class in [1, 2, 3]:
            alarm_level = 1
            status_msg = "CAUTION: Fatigue Signs"
            hud_color = (0, 255, 255) # Yellow
            trigger_sound = False

        # Sound throttle / trigger
        if trigger_sound and (now - self.last_alarm_time > 1.2):
            self._sound_alert(alarm_level)
            self.last_alarm_time = now

        return {
            "alarm_level": alarm_level,
            "status_text": status_msg,
            "hud_color": hud_color,
            "smoothed_fatigue_score": smoothed_prob,
            "perclos": perclos,
            "closure_duration": closure_duration,
            "predicted_class": predicted_class
        }

    def _sound_alert(self, level: int):
        """
        Synthesizes audible alarm frequencies.
        """
        try:
            import winsound
            if level == 3:
                winsound.Beep(2500, 300)
            elif level == 2:
                winsound.Beep(1200, 150)
        except Exception:
            pass  # Non-blocking if running in headless or non-Windows system
