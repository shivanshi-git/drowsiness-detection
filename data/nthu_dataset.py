import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .transforms import LowLightVideoAugmentation
from .optical_flow import DenseOpticalFlowExtractor


class NTHUDriverDrowsinessDataset(Dataset):
    """
    NTHU-DDD Dataset Loader (Strict Real-Data Validator).
    """
    CLASS_MAP = {
        "normal": 0, "normal_driving": 0,
        "slow_blinking": 1, "slow_blink": 1, "blinking": 1,
        "yawning": 2, "yawn": 2,
        "nodding": 3, "head_nod": 3,
        "eye_closure": 4, "sleep": 4, "drowsy": 4
    }

    def __init__(
        self,
        root_dir: str,
        subjects: list = None,
        sequence_length: int = 16,
        frame_step: int = 2,
        is_train: bool = True
    ):
        self.root_dir = root_dir
        self.subjects = subjects or []
        self.sequence_length = sequence_length
        self.frame_step = frame_step
        self.is_train = is_train

        self.transform = LowLightVideoAugmentation(is_train=is_train)
        self.flow_extractor = DenseOpticalFlowExtractor()
        self.samples = []
        self._index_dataset()

    def _index_dataset(self):
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(
                f"[DATASET ERROR] Dataset directory '{self.root_dir}' does not exist. "
                f"Please upload/copy your dataset files to '{self.root_dir}' before running training."
            )

        for subj in os.listdir(self.root_dir):
            if self.subjects and (subj not in self.subjects):
                continue
            subj_path = os.path.join(self.root_dir, subj)
            if not os.path.isdir(subj_path):
                continue

            for root, _, files in os.walk(subj_path):
                for f in files:
                    if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        video_path = os.path.join(root, f)
                        label = self._infer_label_from_path(video_path)
                        self.samples.append({
                            "path": video_path,
                            "label": label,
                            "subject": subj
                        })

        if len(self.samples) == 0:
            raise ValueError(f"[DATASET ERROR] No video files (.mp4/.avi) found in '{self.root_dir}'.")

    def _infer_label_from_path(self, path: str) -> int:
        path_lower = path.lower()
        for keyword, label_idx in self.CLASS_MAP.items():
            if keyword in path_lower:
                return label_idx
        return 0

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

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        raw_frames = self._load_video_frames(sample["path"])
        video_tensor = self.transform(raw_frames)
        flow_tensor = self.flow_extractor.extract_sequence_flow(raw_frames)

        return {
            "video": video_tensor,
            "flow": flow_tensor,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "subject": sample.get("subject", "unknown")
        }


def build_nthu_dataloaders(
    root_dir: str,
    batch_size: int = 8,
    sequence_length: int = 16,
    num_workers: int = 0
):
    train_dataset = NTHUDriverDrowsinessDataset(root_dir=root_dir, sequence_length=sequence_length, is_train=True)
    val_dataset = NTHUDriverDrowsinessDataset(root_dir=root_dir, sequence_length=sequence_length, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
