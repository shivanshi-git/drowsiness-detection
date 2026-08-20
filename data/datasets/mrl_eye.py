import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class MRLEyeDataset(Dataset):
    """
    MRL Eye Dataset Loader.
    Strictly loads real cropped eye image patches from raw_dir.
    """
    def __init__(self, root_dir: str, is_train: bool = True, image_size: tuple = (64, 64)):
        self.root_dir = root_dir
        self.is_train = is_train
        self.image_size = image_size
        self.samples = []
        self._index()

    def _index(self):
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(
                f"[DATASET ERROR] MRL Eye root directory '{self.root_dir}' not found. "
                f"Please upload/place the MRL Eye dataset into '{self.root_dir}'."
            )

        for root, _, files in os.walk(self.root_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Format: s0001_00001_0_0_0_0_0_01.png where 4th token is state (0=closed, 1=open)
                    parts = f.split('_')
                    label = int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else 0
                    self.samples.append({"path": os.path.join(root, f), "label": label})

        if len(self.samples) == 0:
            raise ValueError(f"[DATASET ERROR] No image files (.png/.jpg) found in '{self.root_dir}'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(s["path"])
        if img is None:
            img = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, self.image_size)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        return {
            "image": tensor,
            "label": torch.tensor(s["label"], dtype=torch.long)
        }
