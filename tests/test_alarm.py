import unittest
from inference.adaptive_alarm import AdaptiveAlarmSystem
from xai.alarm_explainer import ExplainableAlarmReasoner


class TestAlarmSystem(unittest.TestCase):
    def test_alarm_escalation(self):
        alarm = AdaptiveAlarmSystem(smoothing_window=3)
        res = alarm.update(raw_drowsy_prob=0.1, predicted_class=0)
        self.assertEqual(res["alarm_level"], 0)

        for _ in range(5):
            res = alarm.update(raw_drowsy_prob=0.95, predicted_class=4)
        self.assertEqual(res["alarm_level"], 3)

    def test_alarm_reasoning_card(self):
        reasoner = ExplainableAlarmReasoner()
        card = reasoner.generate_alarm_card(
            drowsy_prob=0.92,
            predicted_class=4,
            ear_value=0.12,
            mar_value=0.68,
            head_pitch=-20.0,
            perclos=0.40,
            closure_duration=2.5,
            alarm_level=3
        )
        self.assertIn("formatted_card", card)
        self.assertIn("prolonged eye closure", card["reason_summary"])


if __name__ == "__main__":
    unittest.main()
