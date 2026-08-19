import unittest
import numpy as np
import torch

from models.retinaface_detector import RetinaFaceDetector
from models.llformer import LLFormer
from models.region_vit import RegionAwareViT
from models.flow_vit import OpticalFlowViT
from models.cross_attention_fusion import MultimodalFusionEngine
from models.temporal_transformer import TemporalSequenceTransformer
from models.drowsiness_pipeline import LowLightDrowsinessPipeline
from inference.adaptive_alarm import AdaptiveAlarmSystem
from xai.master_explainer import MasterXAIExplainer
from xai.landmark_explainer import LandmarkExplainer
from xai.alarm_explainer import ExplainableAlarmReasoner
from xai.temporal_explainer import TemporalAttentionExplainer


class TestLowLightPipelineAndXAI(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.batch_size = 2
        self.seq_len = 8
        self.embed_dim = 128

    def test_retinaface_roi_detector(self):
        detector = RetinaFaceDetector(eye_size=(64, 64), mouth_size=(64, 64))
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        rois = detector.detect_and_crop(dummy_frame)

        self.assertIn("face", rois)
        self.assertIn("left_eye", rois)
        self.assertIn("right_eye", rois)
        self.assertIn("mouth", rois)

    def test_llformer_enhancement(self):
        llformer = LLFormer(in_channels=3, out_channels=3, dim=16, num_blocks=1)
        x = torch.rand(2, 3, 112, 112)
        out = llformer(x)
        self.assertEqual(out.shape, (2, 3, 112, 112))

    def test_end_to_end_pipeline(self):
        model = LowLightDrowsinessPipeline(
            num_classes=5,
            embed_dim=self.embed_dim,
            sequence_length=self.seq_len,
            enable_llformer=True
        )
        video = torch.rand(self.batch_size, self.seq_len, 3, 224, 224)
        flow = torch.rand(self.batch_size, self.seq_len, 2, 112, 112)

        out = model(video, flow)
        self.assertIn("logits", out)
        self.assertIn("fatigue_score", out)
        self.assertEqual(out["logits"].shape, (self.batch_size, 5))

    def test_temporal_behavior_timeline(self):
        model = LowLightDrowsinessPipeline(
            num_classes=5,
            embed_dim=self.embed_dim,
            sequence_length=self.seq_len,
            enable_llformer=False
        )
        temp_exp = TemporalAttentionExplainer(model, fps=3.0)
        video = torch.rand(1, self.seq_len, 3, 224, 224)
        flow = torch.rand(1, self.seq_len, 2, 112, 112)

        res = temp_exp.explain_temporal_behavior(video, flow)
        self.assertIn("timeline_seconds", res)
        self.assertIn("drowsiness_probabilities", res)
        self.assertIn("narrative_explanation", res)
        self.assertEqual(len(res["drowsiness_probabilities"]), self.seq_len)

    def test_explainable_alarm_reasoner(self):
        reasoner = ExplainableAlarmReasoner()
        card = reasoner.generate_alarm_card(
            drowsy_prob=0.94,
            predicted_class=4,
            ear_value=0.14,
            mar_value=0.62,
            head_pitch=-22.0,
            perclos=0.38,
            closure_duration=3.8,
            alarm_level=3
        )
        self.assertIn("formatted_card", card)
        self.assertIn("prolonged eye closure", card["reason_summary"])
        self.assertEqual(card["yawn_status"], "DETECTED")
        self.assertEqual(card["tilt_status"], "DETECTED")


if __name__ == "__main__":
    unittest.main()
