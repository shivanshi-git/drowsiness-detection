import unittest
import torch
from data.datasets.nthu_ddd import NTHUDDDDataset
from data.datasets.mrl_eye import MRLEyeDataset
from data.datasets.yawdd import YawDDDataset


class TestDatasets(unittest.TestCase):
    def test_nthu_dataset_loading(self):
        ds = NTHUDDDDataset(root_dir="", sequence_length=16, is_train=True)
        item = ds[0]
        self.assertIn("video", item)
        self.assertIn("flow", item)
        self.assertEqual(item["video"].shape, (16, 3, 224, 224))
        self.assertEqual(item["flow"].shape, (16, 2, 112, 112))

    def test_mrl_eye_loading(self):
        ds = MRLEyeDataset(root_dir="", is_train=True)
        item = ds[0]
        self.assertIn("image", item)
        self.assertEqual(item["image"].shape, (3, 64, 64))

    def test_yawdd_loading(self):
        ds = YawDDDataset(root_dir="", sequence_length=16)
        item = ds[0]
        self.assertIn("video", item)
        self.assertEqual(item["video"].shape, (16, 3, 224, 224))


if __name__ == "__main__":
    unittest.main()
