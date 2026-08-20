import os
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .transforms import LowLightVideoAugmentation
from .optical_flow import DenseOpticalFlowExtractor


def natural_sort_key(s: str):
    """Sort strings containing numbers in natural human order (e.g. frame_2 before frame_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


class NTHUDriverDrowsinessDataset(Dataset):
    """
    NTHU-DDD Dataset Loader supporting both:
      1. Video files (.mp4, .avi, .mov, .mkv, .webm)
      2. Extracted Image Frame directories (containing sequential .jpg/.png/.jpeg/.bmp/.webp frames).
    
    Extracts multi-frame temporal sequences (e.g. 16 frames per sample) for spatiotemporal training.
    """
    CLASS_MAP = {
        "normal": 0, "normal_driving": 0, "nonsleepy": 0, "non_sleepy": 0, "neutral": 0,
        "slow_blinking": 1, "slow_blink": 1, "blinking": 1, "blink": 1,
        "yawning": 2, "yawn": 2,
        "nodding": 3, "head_nod": 3, "head_nodding": 3, "nod": 3, "headnod": 3,
        "eye_closure": 4, "sleep": 4, "sleepy": 4, "drowsy": 4, "closed_eye": 4, "eyes_closed": 4,
        "sleepycombination": 4, "sleepy_combination": 4
    }

    VALID_IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    VALID_VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')

    def __init__(
        self,
        root_dir: str,
        subjects: list = None,
        sequence_length: int = 16,
        frame_step: int = 2,
        is_train: bool = True
    ):
        self.root_dir = root_dir
        self.subjects = [str(s).zfill(3) if str(s).isdigit() else str(s) for s in (subjects or [])]
        self.sequence_length = sequence_length
        self.frame_step = max(1, frame_step)
        self.is_train = is_train

        self.transform = LowLightVideoAugmentation(is_train=is_train)
        self.flow_extractor = DenseOpticalFlowExtractor()
        self.samples = []
        self._index_dataset()

    def _match_subject(self, path: str) -> str:
        """Extract subject identifier from path or verify against subject list."""
        path_norm = path.replace("\\", "/")
        for subj in self.subjects:
            # Check if subject ID appears as a discrete folder or token in the path
            pattern = rf"(^|/|_|-){re.escape(subj)}(/|_|-|$)"
            if re.search(pattern, path_norm):
                return subj
        # Try finding 3-digit subject code
        match = re.search(r'(?:subject[_\s]?|s)?(\d{3})', path_norm, re.IGNORECASE)
        if match:
            return match.group(1)
        return "unknown"

    def _infer_label_from_path(self, path: str) -> int:
        path_lower = path.replace("\\", "/").lower()
        # Check longest keywords first to prevent partial matches
        sorted_keywords = sorted(self.CLASS_MAP.keys(), key=lambda x: -len(x))
        for keyword in sorted_keywords:
            if keyword in path_lower:
                return self.CLASS_MAP[keyword]
        return 0

    def _index_dataset(self):
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(
                f"[DATASET ERROR] Dataset directory '{self.root_dir}' does not exist. "
                f"Please upload/copy your dataset files to '{self.root_dir}' before running training."
            )

        for root, dirs, files in os.walk(self.root_dir):
            # Check subject at file/clip level rather than skipping directory prematurely

            # 1. Check for video files in current directory
            video_files = [f for f in files if f.lower().endswith(self.VALID_VIDEO_EXTS)]
            for vf in video_files:
                video_path = os.path.join(root, vf)
                video_subj = self._match_subject(video_path)
                if self.subjects and (video_subj not in self.subjects) and not any(s in video_path for s in self.subjects):
                    continue
                label = self._infer_label_from_path(video_path)
                self.samples.append({
                    "type": "video",
                    "path": video_path,
                    "label": label,
                    "subject": video_subj
                })

            # 2. Check for image frame sequences in current directory
            img_files = [f for f in files if f.lower().endswith(self.VALID_IMAGE_EXTS)]
            if len(img_files) >= 2:  # Sequence of images representing a clip
                from collections import defaultdict
                clips = defaultdict(list)
                for f in img_files:
                    base = f.rsplit('_', 2)
                    clip_key = base[0] if len(base) >= 3 else root
                    clips[clip_key].append(f)

                stride = 12
                req_len = self.sequence_length * self.frame_step
                for clip_key, f_list in clips.items():
                    sorted_img_names = sorted(f_list, key=natural_sort_key)
                    n_frames = len(sorted_img_names)
                    sample_path = os.path.join(root, clip_key) if clip_key != root else root
                    clip_subj = self._match_subject(sample_path)
                    if clip_subj == "unknown":
                        clip_subj = self._match_subject(sorted_img_names[0])
                    if self.subjects and (clip_subj not in self.subjects) and not any(s in sample_path for s in self.subjects):
                        continue
                    label = self._infer_label_from_path(sorted_img_names[0]) if clip_key != root else self._infer_label_from_path(root)
                    if "drowsy" in root.lower() and "notdrowsy" not in root.lower() and label == 0:
                        label = 4  # Drowsy / Eye Closure

                    if n_frames <= req_len:
                        sorted_img_paths = [os.path.join(root, fn) for fn in sorted_img_names]
                        self.samples.append({
                            "type": "image_folder",
                            "path": sample_path,
                            "frames": sorted_img_paths,
                            "label": label,
                            "subject": clip_subj
                        })
                    else:
                        for start in range(0, n_frames - req_len + 1, stride):
                            sub_names = sorted_img_names[start : start + req_len]
                            sub_paths = [os.path.join(root, fn) for fn in sub_names]
                            self.samples.append({
                                "type": "image_folder",
                                "path": f"{sample_path}_sub_{start}",
                                "frames": sub_paths,
                                "label": label,
                                "subject": clip_subj
                            })

        if len(self.samples) == 0:
            raise ValueError(
                f"[DATASET ERROR] No video files (.mp4/.avi) or image frame sequences (.jpg/.png) found in '{self.root_dir}'."
            )

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, sample: dict) -> list:
        if sample["type"] == "image_folder":
            frame_paths = sample["frames"]
            total_len = len(frame_paths)
            if total_len == 0:
                return [np.zeros((224, 224, 3), dtype=np.uint8)] * self.sequence_length

            stride = self.frame_step
            req_len = self.sequence_length * stride

            if total_len >= req_len:
                start_idx = np.random.randint(0, total_len - req_len + 1) if self.is_train else 0
                selected_paths = frame_paths[start_idx : start_idx + req_len : stride]
            else:
                indices = np.linspace(0, total_len - 1, self.sequence_length).astype(int)
                selected_paths = [frame_paths[idx] for idx in indices]

            frames = []
            for p in selected_paths:
                img = cv2.imread(p)
                if img is None:
                    img = np.zeros((224, 224, 3), dtype=np.uint8)
                frames.append(img)
            return frames

        elif sample["type"] == "video":
            cap = cv2.VideoCapture(sample["path"])
            raw_frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                raw_frames.append(frame)
                if len(raw_frames) > 500:
                    break
            cap.release()

            if len(raw_frames) == 0:
                return [np.zeros((224, 224, 3), dtype=np.uint8)] * self.sequence_length

            stride = self.frame_step
            total_len = len(raw_frames)
            req_len = self.sequence_length * stride

            if total_len >= req_len:
                start_idx = np.random.randint(0, total_len - req_len + 1) if self.is_train else 0
                selected = raw_frames[start_idx : start_idx + req_len : stride]
            else:
                indices = np.linspace(0, total_len - 1, self.sequence_length).astype(int)
                selected = [raw_frames[idx] for idx in indices]

            return selected

        return [np.zeros((224, 224, 3), dtype=np.uint8)] * self.sequence_length

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        raw_frames = self._load_frames(sample)
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
    frame_step: int = 2,
    num_workers: int = 0
):
    train_dataset = NTHUDriverDrowsinessDataset(
        root_dir=root_dir,
        sequence_length=sequence_length,
        frame_step=frame_step,
        is_train=True
    )
    val_dataset = NTHUDriverDrowsinessDataset(
        root_dir=root_dir,
        sequence_length=sequence_length,
        frame_step=frame_step,
        is_train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
