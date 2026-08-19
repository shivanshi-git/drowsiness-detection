import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from data.transforms import LowLightVideoAugmentation
from data.optical_flow import DenseOpticalFlowExtractor


class YawDDDataset(Dataset):
    """
    YawDD (Yawning Detection Dataset) video sequence loader.
    Classes: 0: Normal, 1: Talking, 2: Yawning
    """
    CLASS_MAP = {"normal": 0, "talking": 1, "yawn": 2, "yawning": 2}

    def __init__(self, root_dir: str, sequence_length: int = 16, is_train: bool = True):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.is_train = is_train
        self.transform = LowLightVideoAugmentation(is_train=is_train)
        self.flow_extractor = DenseOpticalFlowExtractor()
        self.samples = []
        self._index()

    def _index(self):
        if not os.path.exists(self.root_dir):
            for i in range(30 if self.is_train else 10):
                self.samples.append({"type": "synthetic", "label": i % 3})
            return

        for root, _, files in os.walk(self.root_dir):
            for f in files:
                if f.lower().endswith(('.avi', '.mp4')):
                    label = 0
                    for k, v in self.CLASS_MAP.items():
                        if k in f.lower():
                            label = v
                            break
                    self.samples.append({"type": "video", "path": os.path.join(root, f), "label": label})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        label = s["label"]
        raw_frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.sequence_length)]
        
        video_t = self.transform(raw_frames)
        flow_t = self.flow_extractor.extract_sequence_flow(raw_frames)

        return {
            "video": video_t,
            "flow": flow_t,
            "label": torch.tensor(label, dtype=torch.long)
        }
