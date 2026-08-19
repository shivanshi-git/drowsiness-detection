import unittest
import os
import shutil
import tempfile
import cv2
import numpy as np
import torch
from data.datasets.nthu_ddd import NTHUDDDDataset
from data.datasets.mrl_eye import MRLEyeDataset
from data.datasets.yawdd import YawDDDataset


class TestDatasets(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        # 1. Create mock NTHU-DDD sample video
        nthu_dir = os.path.join(self.test_dir, "nthu", "001", "Night")
        os.makedirs(nthu_dir, exist_ok=True)
        nthu_video = os.path.join(nthu_dir, "slow_blinking.avi")
        out = cv2.VideoWriter(nthu_video, cv2.VideoWriter_fourcc(*'XVID'), 10, (224, 224))
        for _ in range(20):
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

        # 2. Create mock MRL sample image
        mrl_dir = os.path.join(self.test_dir, "mrl")
        os.makedirs(mrl_dir, exist_ok=True)
        mrl_img = os.path.join(mrl_dir, "s0001_00001_0_0_0_0_0_01.png")
        cv2.imwrite(mrl_img, np.zeros((64, 64, 3), dtype=np.uint8))

        # 3. Create mock YawDD sample video
        yawdd_dir = os.path.join(self.test_dir, "yawdd")
        os.makedirs(yawdd_dir, exist_ok=True)
        yawdd_video = os.path.join(yawdd_dir, "yawning_driver.avi")
        out = cv2.VideoWriter(yawdd_video, cv2.VideoWriter_fourcc(*'XVID'), 10, (224, 224))
        for _ in range(20):
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_nthu_dataset_loading(self):
        ds = NTHUDDDDataset(root_dir=os.path.join(self.test_dir, "nthu"), sequence_length=16, is_train=True)
        item = ds[0]
        self.assertIn("video", item)
        self.assertIn("flow", item)
        self.assertEqual(item["video"].shape, (16, 3, 224, 224))
        self.assertEqual(item["flow"].shape, (16, 2, 112, 112))

    def test_mrl_eye_loading(self):
        ds = MRLEyeDataset(root_dir=os.path.join(self.test_dir, "mrl"), is_train=True)
        item = ds[0]
        self.assertIn("image", item)
        self.assertEqual(item["image"].shape, (3, 64, 64))

    def test_yawdd_loading(self):
        ds = YawDDDataset(root_dir=os.path.join(self.test_dir, "yawdd"), sequence_length=16)
        item = ds[0]
        self.assertIn("video", item)
        self.assertEqual(item["video"].shape, (16, 3, 224, 224))


if __name__ == "__main__":
    unittest.main()
