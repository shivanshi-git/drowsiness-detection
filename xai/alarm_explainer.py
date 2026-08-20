import time
import numpy as np


class ExplainableAlarmReasoner:
    """
    Synthesizes Multi-Factor Explanations for Driver Alarm Events.
    Translates raw deep learning tensors & geometric metrics into an interpretable
    diagnostic alarm card for drivers and fleet managers.
    """
    def __init__(self):
        pass

    def generate_alarm_card(
        self,
        drowsy_prob: float,
        predicted_class: int,
        ear_value: float,
        mar_value: float,
        head_pitch: float,
        perclos: float,
        closure_duration: float,
        temporal_window_sec: float = 5.0,
        alarm_level: int = 2
    ) -> dict:
        """
        Synthesizes a human-readable Explainable Alarm Card.
        """
        # 1. Evaluate factor states
        if ear_value < 0.18 or closure_duration >= 1.5:
            eye_status = "CRITICAL CLOSURE"
        elif ear_value < 0.22 or closure_duration >= 0.8:
            eye_status = "HIGH (DROOPING)"
        elif ear_value < 0.25:
            eye_status = "MODERATE"
        else:
            eye_status = "NORMAL (OPEN)"

        is_yawning = mar_value > 0.55 or predicted_class == 2
        yawn_status = "DETECTED" if is_yawning else "NONE"

        is_head_tilt = abs(head_pitch) > 18.0 or predicted_class == 3
        tilt_status = "DETECTED" if is_head_tilt else "STABLE"

        # 2. Formulate Root Cause Reasons
        reasons = []
        if closure_duration >= 1.5:
            reasons.append(f"prolonged eye closure ({closure_duration:.1f}s)")
        elif ear_value < 0.22:
            reasons.append("eyelid drooping")

        if perclos > 0.15:
            reasons.append(f"high PERCLOS ({perclos*100:.1f}%)")

        if is_yawning:
            reasons.append("frequent yawning")

        if is_head_tilt:
            reasons.append(f"head nodding/tilt ({head_pitch:+.1f}°)")

        if not reasons:
            if drowsy_prob >= 0.70:
                reasons.append("optical flow facial velocity slowdown")
            else:
                reasons.append("subtle fatigue micro-expressions")

        reason_text = " + ".join(reasons)

        # 3. Format ASCII/Dashboard Card
        banner_title = "🚨 CRITICAL EMERGENCY ALERT" if alarm_level == 3 else "⚠️ DROWSINESS WARNING"
        card_text = (
            f"╔════════════════════════════════════════════════════╗\n"
            f"║           {banner_title:^38}       ║\n"
            f"╠════════════════════════════════════════════════════╣\n"
            f"║ Drowsiness Probability: {drowsy_prob * 100:>5.1f}%                     ║\n"
            f"║                                                    ║\n"
            f"║ Eye Closure:       {eye_status:<28}║\n"
            f"║ PERCLOS:           {perclos * 100:>5.1f}%                          ║\n"
            f"║ Yawning:           {yawn_status:<28}║\n"
            f"║ Head Tilt/Pitch:   {tilt_status:<28}║\n"
            f"║ Event Duration:    {closure_duration:>4.1f} sec                        ║\n"
            f"║                                                    ║\n"
            f"║ Reason: {reason_text[:40]:<43}║\n"
            f"╚════════════════════════════════════════════════════╝"
        )

        return {
            "drowsiness_prob_pct": float(drowsy_prob * 100.0),
            "eye_status": eye_status,
            "perclos_pct": float(perclos * 100.0),
            "yawn_status": yawn_status,
            "tilt_status": tilt_status,
            "duration_sec": float(closure_duration),
            "reason_summary": reason_text,
            "formatted_card": card_text,
            "alarm_level": alarm_level
        }
