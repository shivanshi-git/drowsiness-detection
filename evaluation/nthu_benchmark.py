import argparse
import json
import torch
from torch.utils.data import DataLoader

from models.drowsiness_pipeline import LowLightDrowsinessPipeline
from data.nthu_dataset import NTHUDriverDrowsinessDataset
from evaluation.metrics import compute_comprehensive_metrics, LatencyProfiler


def run_nthu_benchmark(
    data_dir: str,
    checkpoint_path: str = None,
    batch_size: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"============================================================")
    print(f" NTHU-DDD Subject-Independent SOTA Benchmark Evaluation")
    print(f" Device: {device}")
    print(f"============================================================")

    test_dataset = NTHUDriverDrowsinessDataset(
        root_dir=data_dir,
        is_train=False,
        sequence_length=16
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LowLightDrowsinessPipeline(num_classes=5, sequence_length=16).to(device)
    if checkpoint_path and torch.os.path.exists(checkpoint_path):
        print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            video = batch["video"].to(device)
            flow = batch["flow"].to(device)
            labels = batch["label"].to(device)

            out = model(video, flow)
            logits = out["logits"]
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.tolist())

    metrics = compute_comprehensive_metrics(all_targets, all_preds, all_probs)
    print("\n--- BENCHMARK RESULTS ---")
    print(f"Accuracy:         {metrics['accuracy']*100:.2f}%")
    print(f"Balanced Acc:     {metrics['balanced_accuracy']*100:.2f}%")
    print(f"Macro F1-Score:   {metrics['macro_f1']*100:.2f}%")
    print(f"Weighted F1:      {metrics['weighted_f1']*100:.2f}%")
    if "roc_auc_ovr" in metrics:
        print(f"ROC-AUC (OvR):    {metrics['roc_auc_ovr']:.4f}")

    # Latency Profiling
    profiler = LatencyProfiler(device=device)
    dummy_video = torch.randn(1, 16, 3, 224, 224)
    dummy_flow = torch.randn(1, 16, 2, 112, 112)
    timing = profiler.profile_pipeline(model, dummy_video, dummy_flow)
    print(f"\n--- INFERENCE LATENCY ---")
    print(f"Latency per sequence: {timing['mean_latency_ms']:.2f} ms (+/- {timing['std_latency_ms']:.2f} ms)")
    print(f"Throughput:           {timing['fps_throughput']:.1f} sequences/sec")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/nthu_ddd_raw", help="Path to NTHU dataset")
    parser.add_argument("--checkpoint", default=None, help="Trained checkpoint path")
    args = parser.parse_args()

    run_nthu_benchmark(data_dir=args.data_dir, checkpoint_path=args.checkpoint)
