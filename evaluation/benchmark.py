import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from data.datasets.nthu_ddd import NTHUDDDDataset
from models.drowsiness_pipeline import LowLightDrowsinessPipeline
from evaluation.metrics import compute_comprehensive_metrics
from evaluation.confusion_matrix import plot_and_save_confusion_matrix


def run_benchmark(config_path: str = "configs/nthu_ddd.yaml", checkpoint_path: str = None):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[BENCHMARK] Evaluating on {cfg['dataset']['name']} using {device}")

    ds = NTHUDDDDataset(root_dir=cfg["dataset"].get("raw_dir", ""), is_train=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    model = LowLightDrowsinessPipeline(num_classes=cfg["dataset"].get("num_classes", 5)).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    preds, targets, probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            video = batch["video"].to(device)
            flow = batch["flow"].to(device)
            labels = batch["label"].to(device)

            out = model(video, flow)
            prob = torch.softmax(out["logits"], dim=1).cpu().numpy()
            pred = torch.argmax(out["logits"], dim=1).cpu().numpy()

            preds.extend(pred.tolist())
            targets.extend(labels.cpu().tolist())
            probs.extend(prob.tolist())

    metrics = compute_comprehensive_metrics(targets, preds, probs)
    print("\n--- BENCHMARK RESULTS ---")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k:20s}: {v}")

    os.makedirs("results", exist_ok=True)
    plot_and_save_confusion_matrix(targets, preds, cfg["dataset"].get("class_names", ["0", "1", "2", "3", "4"]), "results/benchmark_confusion_matrix.png")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nthu_ddd.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    run_benchmark(args.config, args.checkpoint)
