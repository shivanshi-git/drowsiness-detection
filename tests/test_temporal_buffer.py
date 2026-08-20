import unittest

from utils.temporal_buffer import TemporalPERCLOSBuffer


class TemporalPERCLOSBufferTests(unittest.TestCase):
    def test_danger_requires_full_window_and_duration(self):
        current_time = [0.0]
        buffer = TemporalPERCLOSBuffer(
            window_size=4,
            danger_threshold=0.75,
            danger_duration_seconds=2.0,
            clock=lambda: current_time[0],
        )

        for _ in range(3):
            result = buffer.update(True)
        self.assertNotEqual(result["state"], "DANGER")

        result = buffer.update(True)
        self.assertEqual(result["state"], "WARNING")

        current_time[0] = 2.0
        result = buffer.update(True)
        self.assertEqual(result["state"], "DANGER")

    def test_alert_clears_buffer_state(self):
        buffer = TemporalPERCLOSBuffer(window_size=2, warning_threshold=0.5)
        buffer.update(True)
        buffer.update(True)
        result = buffer.update(False)
        self.assertEqual(result["state"], "WARNING")
        buffer.reset()
        self.assertEqual(buffer.update(False)["state"], "ALERT")


if __name__ == "__main__":
    unittest.main()