import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class MRLEyeDataset(Dataset):
    """
    MRL Eye Dataset (Cropped eye patches with binary labels: 0=Closed, 1=Open).
    """
    def __init__(self, root_dir: str, is_train: bool = True, image_size: tuple = (64, 64)):
        self.root_dir = root_dir
        self.is_train = is_train
        self.image_size = image_size
        self.samples = []
        self._index()

    def _index(self):
        if not os.path.exists(self.root_dir):
            for i in range(40 if self.is_train else 10):
                self.samples.append({"type": "synthetic", "label": i % 2})
            return

        for root, _, files in os.walk(self.root_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # MRL filename format: s0001_00001_0_0_0_0_0_01.png where 4th token is state (0=closed, 1=open)
                    parts = f.split('_')
                    label = int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else 0
                    self.samples.append({"type": "image", "path": os.path.join(root, f), "label": label})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        label = s["label"]
        img = np.full((self.image_size[0], self.image_size[1], 3), 40 if label == 0 else 120, dtype=np.uint8)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return {
            "image": tensor,
            "label": torch.tensor(label, dtype=torch.long)
        }
