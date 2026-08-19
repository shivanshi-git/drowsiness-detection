import os
import torch


class CheckpointManager:
    """
    Saves and restores model checkpoints, optimizers, schedulers, and metrics.
    """
    def __init__(self, checkpoint_dir: str = "saved_models"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_best(self, model: torch.nn.Module, epoch: int, metric_val: float, filename: str = "best_model.pth"):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "epoch": epoch,
            "metric": metric_val,
            "state_dict": model.state_dict()
        }, path)
        print(f"[CHECKPOINT] Saved best model to {path} (Metric: {metric_val:.4f})")

    def load(self, model: torch.nn.Module, filename: str = "best_model.pth", device: str = "cpu"):
        path = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=device)
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            print(f"[CHECKPOINT] Successfully loaded weights from {path}")
            return ckpt
        return None
