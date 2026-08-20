import cv2
import numpy as np
import torch
import torch.nn.functional as F


class DenseOpticalFlowExtractor:
    """
    Computes dense optical flow fields across temporal video sequences.
    Outputs 2-channel motion velocity tensors (u_x, u_y) and magnitude normalized to [-1, 1].
    """
    def __init__(self, target_size: tuple = (112, 112), method: str = "farneback"):
        self.target_size = target_size
        self.method = method

    def compute_pair_flow(self, prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> np.ndarray:
        """
        Computes 2-channel flow (dx, dy) between two consecutive BGR frames.
        """
        prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

        if prev_gray.shape != self.target_size:
            prev_gray = cv2.resize(prev_gray, (self.target_size[1], self.target_size[0]))
            curr_gray = cv2.resize(curr_gray, (self.target_size[1], self.target_size[0]))

        # Dual TV-L1 or Farneback
        if self.method == "farneback":
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
        else:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

        # Normalize flow to [-1, 1] range
        flow = np.clip(flow / 20.0, -1.0, 1.0)
        return flow  # Shape: (H, W, 2)

    def extract_sequence_flow(self, frames_bgr: list) -> torch.Tensor:
        """
        Extracts optical flow across a sequence of T frames.
        Returns:
            torch.Tensor of shape (T, 2, H, W)
        """
        num_frames = len(frames_bgr)
        if num_frames == 0:
            return torch.zeros((1, 2, self.target_size[0], self.target_size[1]), dtype=torch.float32)

        flow_list = []
        # First frame flow is zero motion
        zero_flow = np.zeros((self.target_size[0], self.target_size[1], 2), dtype=np.float32)
        flow_list.append(zero_flow)

        for t in range(1, num_frames):
            prev = frames_bgr[t - 1]
            curr = frames_bgr[t]
            flow = self.compute_pair_flow(prev, curr)
            flow_list.append(flow)

        # Convert to Tensor: (T, H, W, 2) -> (T, 2, H, W)
        flows_np = np.stack(flow_list, axis=0)
        flows_tensor = torch.from_numpy(flows_np).permute(0, 3, 1, 2).float()
        return flows_tensor


def visualize_optical_flow_hsv(flow: np.ndarray) -> np.ndarray:
    """
    Visualizes 2D optical flow using HSV color representation.
    """
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr
