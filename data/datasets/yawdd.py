import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from data.transforms import LowLightVideoAugmentation
from data.optical_flow import DenseOpticalFlowExtractor


class YawDDDataset(Dataset):
    """
    YawDD Dataset Loader.
    Strictly loads real driver video clips from raw_dir.
    """
    CLASS_MAP = {"normal": 0, "talking": 1, "yawn": 2, "yawning": 2}

    def __init__(self, root_dir: str, sequence_length: int = 16, frame_step: int = 2, is_train: bool = True):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.frame_step = frame_step
        self.is_train = is_train
        self.transform = LowLightVideoAugmentation(is_train=is_train)
        self.flow_extractor = DenseOpticalFlowExtractor()
        self.samples = []
        self._index()

    def _index(self):
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(
                f"[DATASET ERROR] YawDD root directory '{self.root_dir}' not found. "
                f"Please upload/place the YawDD dataset into '{self.root_dir}'."
            )

        for root, _, files in os.walk(self.root_dir):
            for f in files:
                if f.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')):
                    label = 0
                    for k, v in self.CLASS_MAP.items():
                        if k in f.lower():
                            label = v
                            break
                    self.samples.append({"path": os.path.join(root, f), "label": label})

        if len(self.samples) == 0:
            raise ValueError(f"[DATASET ERROR] No video files (.avi/.mp4) found in '{self.root_dir}'.")

    def __len__(self):
        return len(self.samples)

    def _load_video_frames(self, video_path: str) -> list:
        cap = cv2.VideoCapture(video_path)
        raw_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            raw_frames.append(frame)
            if len(raw_frames) > 300:
                break
        cap.release()

        if len(raw_frames) == 0:
            return [np.zeros((224, 224, 3), dtype=np.uint8)] * self.sequence_length

        stride = self.frame_step
        total_len = len(raw_frames)
        req_len = self.sequence_length * stride

        if total_len >= req_len:
            start_idx = np.random.randint(0, total_len - req_len + 1) if self.is_train else 0
            selected = raw_frames[start_idx:start_idx + req_len:stride]
        else:
            indices = np.linspace(0, total_len - 1, self.sequence_length).astype(int)
            selected = [raw_frames[idx] for idx in indices]

        return selected

    def __getitem__(self, idx):
        s = self.samples[idx]
        raw_frames = self._load_video_frames(s["path"])
        video_t = self.transform(raw_frames)
        flow_t = self.flow_extractor.extract_sequence_flow(raw_frames)

        return {
            "video": video_t,
            "flow": flow_t,
            "label": torch.tensor(s["label"], dtype=torch.long)
        }
