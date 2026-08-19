import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .transforms import LowLightVideoAugmentation
from .optical_flow import DenseOpticalFlowExtractor


class NTHUDriverDrowsinessDataset(Dataset):
    """
    NTHU-DDD (National Tsing Hua University Driver Drowsiness Detection) Dataset Loader.
    Loads temporal sequences of frames, extracts optical flow, and returns multi-modal tensors.
    
    Classes:
        0: Normal Driving
        1: Slow Blinking
        2: Yawning
        3: Nodding
        4: Eye Closure
    """
    CLASS_MAP = {
        "normal": 0,
        "normal_driving": 0,
        "slow_blinking": 1,
        "slow_blink": 1,
        "blinking": 1,
        "yawning": 2,
        "yawn": 2,
        "nodding": 3,
        "head_nod": 3,
        "eye_closure": 4,
        "sleep": 4,
        "drowsy": 4
    }

    def __init__(
        self,
        root_dir: str,
        subjects: list = None,
        sequence_length: int = 16,
        frame_step: int = 2,
        is_train: bool = True,
        image_size: tuple = (224, 224),
        flow_size: tuple = (112, 112),
        generate_synthetic_if_empty: bool = True
    ):
        self.root_dir = root_dir
        self.subjects = subjects or []
        self.sequence_length = sequence_length
        self.frame_step = frame_step
        self.is_train = is_train
        self.image_size = image_size
        self.flow_size = flow_size

        self.transform = LowLightVideoAugmentation(is_train=is_train, target_size=image_size)
        self.flow_extractor = DenseOpticalFlowExtractor(target_size=flow_size)

        self.samples = []
        self._index_dataset()

        # If no real files were found on disk, build synthetic dataset samples for standalone execution
        if len(self.samples) == 0 and generate_synthetic_if_empty:
            self._create_synthetic_index()

    def _index_dataset(self):
        """
        Scans dataset directory for video clips and labels.
        Expected folder hierarchy:
          root_dir / <Subject_ID> / <Scenario: Night_Glasses etc.> / <Action_Name>.avi or mp4
        """
        if not os.path.exists(self.root_dir):
            return

        for subj in os.listdir(self.root_dir):
            subj_path = os.path.join(self.root_dir, subj)
            if not os.path.isdir(subj_path):
                continue
            if self.subjects and (subj not in self.subjects):
                continue

            for root, _, files in os.walk(subj_path):
                for f in files:
                    if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        video_path = os.path.join(root, f)
                        label = self._infer_label_from_path(video_path)
                        self.samples.append({
                            "type": "video",
                            "path": video_path,
                            "label": label,
                            "subject": subj
                        })

    def _infer_label_from_path(self, path: str) -> int:
        path_lower = path.lower()
        for keyword, label_idx in self.CLASS_MAP.items():
            if keyword in path_lower:
                return label_idx
        return 0  # Default Normal

    def _create_synthetic_index(self):
        """
        Generates simulated sample metadata for zero-setup end-to-end prototyping.
        """
        for i in range(40 if self.is_train else 10):
            self.samples.append({
                "type": "synthetic",
                "label": i % 5,
                "subject": f"sim_subj_{i % 5}"
            })

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
            if len(raw_frames) > 300: # Limit memory consumption
                break
        cap.release()

        if len(raw_frames) == 0:
            return [np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)] * self.sequence_length

        # Subsample / slice sliding window
        stride = self.frame_step
        total_len = len(raw_frames)
        req_len = self.sequence_length * stride

        if total_len >= req_len:
            start_idx = np.random.randint(0, total_len - req_len + 1) if self.is_train else 0
            selected = raw_frames[start_idx:start_idx + req_len:stride]
        else:
            # Repeat or pad
            indices = np.linspace(0, total_len - 1, self.sequence_length).astype(int)
            selected = [raw_frames[idx] for idx in indices]

        return selected

    def _generate_synthetic_frames(self, label: int) -> list:
        """
        Generates mock low-light video frames with simulated eye/mouth movement dynamics.
        """
        frames = []
        for t in range(self.sequence_length):
            # Create low-light dark background
            frame = np.full((self.image_size[0], self.image_size[1], 3), 20, dtype=np.uint8)
            # Draw synthetic face circle
            center = (self.image_size[1] // 2, self.image_size[0] // 2)
            cv2.circle(frame, center, 60, (50, 50, 60), -1)

            # Eye state simulation based on label & time
            eye_openness = 1.0
            if label == 4 or label == 1: # Eye closure or slow blinking
                eye_openness = max(0.1, np.sin(t / 2.0))
            eye_h = int(10 * eye_openness)

            # Left & Right eyes
            cv2.ellipse(frame, (center[0] - 25, center[1] - 15), (12, max(2, eye_h)), 0, 0, 360, (120, 120, 130), -1)
            cv2.ellipse(frame, (center[0] + 25, center[1] - 15), (12, max(2, eye_h)), 0, 0, 360, (120, 120, 130), -1)

            # Mouth simulation for yawning
            mouth_openness = 0.2
            if label == 2: # Yawning
                mouth_openness = max(0.2, (t / float(self.sequence_length)) * 1.5)
            mouth_h = int(15 * mouth_openness)
            cv2.ellipse(frame, (center[0], center[1] + 25), (18, max(4, mouth_h)), 0, 0, 360, (100, 80, 90), -1)

            frames.append(frame)
        return frames

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        label = sample["label"]

        if sample["type"] == "video":
            raw_frames = self._load_video_frames(sample["path"])
        else:
            raw_frames = self._generate_synthetic_frames(label)

        # 1. Spatial Video Tensor: (T, 3, H, W)
        video_tensor = self.transform(raw_frames)

        # 2. Dense Optical Flow Tensor: (T, 2, H_flow, W_flow)
        flow_tensor = self.flow_extractor.extract_sequence_flow(raw_frames)

        return {
            "video": video_tensor,
            "flow": flow_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "subject": sample.get("subject", "unknown")
        }


def build_nthu_dataloaders(
    root_dir: str,
    batch_size: int = 8,
    sequence_length: int = 16,
    num_workers: int = 0
):
    train_dataset = NTHUDriverDrowsinessDataset(
        root_dir=root_dir,
        sequence_length=sequence_length,
        is_train=True
    )
    val_dataset = NTHUDriverDrowsinessDataset(
        root_dir=root_dir,
        sequence_length=sequence_length,
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True if len(train_dataset) > batch_size else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader
