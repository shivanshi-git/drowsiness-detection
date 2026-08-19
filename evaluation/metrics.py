import time
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score
)


def compute_comprehensive_metrics(y_true: list, y_pred: list, y_probs: list = None, class_names: list = None) -> dict:
    """
    Computes all standard SOTA classification metrics for driver drowsiness benchmarks.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weight),
        "confusion_matrix": cm.tolist()
    }

    if y_probs is not None:
        try:
            y_probs_np = np.array(y_probs)
            if y_probs_np.ndim == 2 and y_probs_np.shape[1] > 1:
                auc = roc_auc_score(y_true, y_probs_np, multi_class='ovr')
                metrics["roc_auc_ovr"] = float(auc)
        except Exception:
            pass

    return metrics


class LatencyProfiler:
    """
    Profiles execution runtime (ms) per component in the low-light pipeline.
    """
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.timings = {}

    def profile_pipeline(self, model: torch.nn.Module, video_sample: torch.Tensor, flow_sample: torch.Tensor, warmup: int = 3, runs: int = 10) -> dict:
        model.eval()
        video_sample = video_sample.to(self.device)
        flow_sample = flow_sample.to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(video_sample, flow_sample)

        # Measured runs
        times = []
        with torch.no_grad():
            for _ in range(runs):
                if self.device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                _ = model(video_sample, flow_sample)

                if self.device == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)  # milliseconds

        return {
            "mean_latency_ms": float(np.mean(times)),
            "std_latency_ms": float(np.std(times)),
            "fps_throughput": float(1000.0 / np.mean(times))
        }
