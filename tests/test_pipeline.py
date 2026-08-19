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


class TestLowLightPipeline(unittest.TestCase):
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
        self.assertEqual(rois["face"].shape, (224, 224, 3))
        self.assertEqual(rois["left_eye"].shape, (64, 64, 3))
        self.assertEqual(rois["mouth"].shape, (64, 64, 3))

    def test_llformer_enhancement(self):
        llformer = LLFormer(in_channels=3, out_channels=3, dim=16, num_blocks=1)
        x = torch.rand(2, 3, 112, 112)
        out = llformer(x)

        self.assertEqual(out.shape, (2, 3, 112, 112))
        self.assertTrue((out >= 0.0).all() and (out <= 1.0).all())

    def test_region_vit(self):
        region_vit = RegionAwareViT(embed_dim=self.embed_dim, depth=2, num_heads=4)
        face = torch.rand(self.batch_size, 3, 224, 224)
        leye = torch.rand(self.batch_size, 3, 64, 64)
        reye = torch.rand(self.batch_size, 3, 64, 64)
        mouth = torch.rand(self.batch_size, 3, 64, 64)

        tokens = region_vit(face, leye, reye, mouth)
        self.assertEqual(tokens.shape[0], self.batch_size)
        self.assertEqual(tokens.shape[2], self.embed_dim)

    def test_flow_vit(self):
        flow_vit = OpticalFlowViT(img_size=112, patch_size=16, in_channels=2, embed_dim=self.embed_dim, depth=2, num_heads=4)
        flow = torch.rand(self.batch_size, 2, 112, 112)
        tokens = flow_vit(flow)

        self.assertEqual(tokens.shape[0], self.batch_size)
        self.assertEqual(tokens.shape[2], self.embed_dim)

    def test_fusion_and_temporal(self):
        fusion = MultimodalFusionEngine(embed_dim=self.embed_dim, num_heads=4)
        temporal = TemporalSequenceTransformer(embed_dim=self.embed_dim, depth=2, num_heads=4, num_classes=5)

        s_tokens = torch.rand(self.batch_size * self.seq_len, 245, self.embed_dim)
        f_tokens = torch.rand(self.batch_size * self.seq_len, 50, self.embed_dim)

        frame_emb = fusion(s_tokens, f_tokens)
        self.assertEqual(frame_emb.shape, (self.batch_size * self.seq_len, self.embed_dim))

        seq_emb = frame_emb.view(self.batch_size, self.seq_len, self.embed_dim)
        out = temporal(seq_emb)

        self.assertEqual(out["logits"].shape, (self.batch_size, 5))
        self.assertEqual(out["fatigue_score"].shape, (self.batch_size, 1))

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

    def test_adaptive_alarm_escalation(self):
        alarm = AdaptiveAlarmSystem(smoothing_window=3, level_1_thresh=0.4, level_2_thresh=0.6, level_3_thresh=0.8)

        # 1. Normal state -> Level 0
        res = alarm.update(raw_drowsy_prob=0.1, predicted_class=0)
        self.assertEqual(res["alarm_level"], 0)

        # 2. Repeated Yawning -> Level 1
        res = alarm.update(raw_drowsy_prob=0.45, predicted_class=2)
        self.assertGreaterEqual(res["alarm_level"], 1)

        # 3. High Drowsiness / Eye Closure -> Level 3
        for _ in range(5):
            res = alarm.update(raw_drowsy_prob=0.95, predicted_class=4)
        self.assertEqual(res["alarm_level"], 3)
        self.assertIn("CRITICAL", res["status_text"])


if __name__ == "__main__":
    unittest.main()
