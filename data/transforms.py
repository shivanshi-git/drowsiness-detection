import random
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class LowLightVideoAugmentation:
    """
    Photometric and geometric augmentations specialized for low-light / infrared
    in-cabin video sequences. Preserves temporal consistency across sequence frames.
    """
    def __init__(self, is_train: bool = True, target_size=(224, 224)):
        self.is_train = is_train
        self.target_size = target_size

    def __call__(self, frames: list) -> torch.Tensor:
        """
        Args:
            frames: List of BGR numpy images of shape (H, W, 3)
        Returns:
            torch.Tensor of shape (T, C, H, W) normalized to [0, 1]
        """
        # Determine temporal-consistent augmentation parameters
        do_hflip = self.is_train and (random.random() < 0.5)
        gamma = random.uniform(0.6, 1.8) if self.is_train else 1.0
        contrast_factor = random.uniform(0.7, 1.4) if self.is_train else 1.0
        brightness_delta = random.randint(-30, 30) if self.is_train else 0
        add_noise = self.is_train and (random.random() < 0.3)

        processed_frames = []
        for img in frames:
            if img is None or img.size == 0:
                img = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)

            # Resize to canonical target size
            if (img.shape[0], img.shape[1]) != self.target_size:
                img = cv2.resize(img, (self.target_size[1], self.target_size[0]))

            # Horizontal flip
            if do_hflip:
                img = cv2.flip(img, 1)

            # Low-light photometric shifts
            if self.is_train:
                # Gamma correction (simulating severe underexposure/headlights)
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                img = cv2.LUT(img, table)

                # Brightness & contrast
                img = cv2.convertScaleAbs(img, alpha=contrast_factor, beta=brightness_delta)

                # Sensor noise simulation
                if add_noise:
                    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
                    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Convert BGR -> RGB -> Tensor [C, H, W]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            processed_frames.append(tensor)

        # Stack into (T, C, H, W)
        return torch.stack(processed_frames, dim=0)


def apply_adaptive_histogram_equalization(image_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Applies CLAHE on the L channel of LAB color space to enhance low-light contrast.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
