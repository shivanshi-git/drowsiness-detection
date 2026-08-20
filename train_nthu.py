import os
import time
import json
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt

from models.drowsiness_pipeline import LowLightDrowsinessPipeline
from data.nthu_dataset import build_nthu_dataloaders
from evaluation.metrics import compute_comprehensive_metrics
from generate_xai_samples import generate_xai_plots


def plot_and_save_artifacts(history: list, val_targets: list, val_preds: list, val_probs: list = None, results_dir: str = "results"):
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(history)

    # 1. Training & Validation Loss/Accuracy Curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss", marker="s")
    plt.title("Loss Progression Across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(df["epoch"], df["train_macro_f1"] * 100, label="Train Macro F1 (%)", marker="o")
    plt.plot(df["epoch"], df["val_accuracy"] * 100, label="Val Accuracy (%)", marker="s")
    plt.plot(df["epoch"], df["val_macro_f1"] * 100, label="Val Macro F1 (%)", marker="^")
    plt.title("Performance Metrics Across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Percentage (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    curves_path = os.path.join(results_dir, "training_curves.png")
    plt.savefig(curves_path, dpi=200)
    plt.close()

    # 2. Confusion Matrix Plot
    from sklearn.metrics import confusion_matrix
    class_names = ["Normal", "Slow Blink", "Yawn", "Nodding", "Eye Closure"]
    cm = confusion_matrix(val_targets, val_preds, labels=list(range(5)))

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("NTHU-DDD Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2. if cm.max() > 0 else 1.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # 3. ROC Curve Plot
    if val_probs is not None and len(val_probs) > 0:
        from evaluation.roc_auc import plot_and_save_roc_curve
        roc_path = os.path.join(results_dir, "roc_curve.png")
        plot_and_save_roc_curve(val_targets, val_probs, num_classes=5, output_path=roc_path)

    # 4. Export Metrics Summary Table (CSV & JSON)
    final_metrics = compute_comprehensive_metrics(val_targets, val_preds, val_probs)
    metrics_summary = {
        "Accuracy": float(final_metrics.get("accuracy", 0.0)),
        "Macro_Precision": float(final_metrics.get("macro_precision", 0.0)),
        "Macro_Recall": float(final_metrics.get("macro_recall", 0.0)),
        "Macro_F1": float(final_metrics.get("macro_f1", 0.0)),
        "Weighted_F1": float(final_metrics.get("weighted_f1", 0.0)),
    }

    metrics_df = pd.DataFrame([metrics_summary])
    metrics_csv_path = os.path.join(results_dir, "evaluation_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)

    metrics_json_path = os.path.join(results_dir, "evaluation_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)

    print(f"[ARTIFACTS SUCCESS] Saved plots & metric tables to '{results_dir}/'")


def train_sota_pipeline(config_path: str = "configs/nthu_ddd_config.yaml", epochs_override: int = None):
    # Enforce GPU Execution
    assert torch.cuda.is_available(), "[ERROR] CUDA GPU is required for training!"
    device = torch.device("cuda:0")
    print(f"[INFO] Enforced GPU Device: {torch.cuda.get_device_name(0)}")

    # Load configuration
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]
    data_cfg = cfg["dataset"]
    save_dir = train_cfg["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    num_epochs = epochs_override if epochs_override is not None else train_cfg["epochs"]

    # 1. Build DataLoaders
    train_loader, val_loader = build_nthu_dataloaders(
        root_dir=data_cfg["raw_dir"],
        batch_size=train_cfg["batch_size"],
        sequence_length=data_cfg["sequence_length"],
        frame_step=data_cfg.get("frame_step", 2),
        num_workers=train_cfg.get("num_workers", 0)
    )
    print(f"[INFO] Loaded {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples.")

    # 2. Build Pipeline Model
    model = LowLightDrowsinessPipeline(
        num_classes=data_cfg["num_classes"],
        embed_dim=cfg["pipeline"]["region_vit"]["embed_dim"],
        sequence_length=data_cfg["sequence_length"]
    ).to(device)

    # 3. Loss, Optimizer, Scheduler, Scaler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=float(train_cfg["min_lr"])
    )
    scaler = GradScaler(enabled=train_cfg["mixed_precision"])

    best_val_f1 = 0.0
    start_epoch = 1
    history = []

    # Auto-Resume Checkpoint Logic
    latest_ckpt_path = os.path.join(save_dir, "checkpoint_latest.pth")
    if os.path.exists(latest_ckpt_path):
        print(f"[INFO] Found existing checkpoint: '{latest_ckpt_path}'. Loading state to resume training...")
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint and checkpoint["scheduler_state"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "scaler_state" in checkpoint and checkpoint["scaler_state"] is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_f1 = checkpoint.get("best_val_f1", 0.0)
        history = checkpoint.get("history", [])
        print(f"[INFO] Resuming training from epoch {start_epoch}/{num_epochs} (Best Val F1: {best_val_f1*100:.2f}%)")

    print("\n--- Starting Training Loop on GPU ---")
    val_preds, val_targets, val_probs = [], [], []

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []

        for batch in train_loader:
            video = batch["video"].to(device)
            flow = batch["flow"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            with autocast(enabled=train_cfg["mixed_precision"]):
                out = model(video, flow)
                loss = criterion(out["logits"], labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip_norm"])
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * video.size(0)
            preds = torch.argmax(out["logits"], dim=1).detach().cpu().numpy()
            train_preds.extend(preds.tolist())
            train_targets.extend(labels.cpu().numpy().tolist())

        scheduler.step()
        train_loss = running_loss / max(1, len(train_loader.dataset))
        train_metrics = compute_comprehensive_metrics(train_targets, train_preds)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_preds, val_targets, val_probs = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                video = batch["video"].to(device)
                flow = batch["flow"].to(device)
                labels = batch["label"].to(device)

                with autocast(enabled=train_cfg["mixed_precision"]):
                    out = model(video, flow)
                    loss = criterion(out["logits"], labels)

                val_loss += loss.item() * video.size(0)
                probs = torch.softmax(out["logits"], dim=1).cpu().numpy()
                preds = torch.argmax(out["logits"], dim=1).cpu().numpy()

                val_preds.extend(preds.tolist())
                val_targets.extend(labels.cpu().numpy().tolist())
                val_probs.extend(probs.tolist())

        val_loss = val_loss / max(1, len(val_loader.dataset))
        val_metrics = compute_comprehensive_metrics(val_targets, val_preds, val_probs)
        epoch_time = time.time() - t0

        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Train F1: {train_metrics['macro_f1']*100:.1f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']*100:.1f}%, Val F1: {val_metrics['macro_f1']*100:.1f}%"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"]
        })

        # Save Resumable Latest Checkpoint
        latest_state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_f1": best_val_f1,
            "history": history
        }
        torch.save(latest_state, latest_ckpt_path)

        # Save Best Model Checkpoint
        if val_metrics["macro_f1"] >= best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_path = os.path.join(save_dir, "best_sota_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  [+] Saved new best model checkpoint -> {best_path} (Val F1: {best_val_f1*100:.2f}%)")

    # Save training history & plots
    pd.DataFrame(history).to_csv(os.path.join(save_dir, "training_history.csv"), index=False)
    plot_and_save_artifacts(history, val_targets, val_preds, val_probs=val_probs, results_dir="results")

    # Generate Explainability (XAI) Plots
    print("\n[INFO] Generating Explainability (XAI) Grad-CAM Visualizations...")
    generate_xai_plots(
        model_checkpoint=os.path.join(save_dir, "best_sota_model.pth"),
        output_dir="results/xai",
        num_samples=4
    )

    print(f"\n[SUCCESS] Training & Artifact Generation Complete! Best Validation F1: {best_val_f1*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nthu_ddd_config.yaml", help="Path to config yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    args = parser.parse_args()

    train_sota_pipeline(config_path=args.config, epochs_override=args.epochs)
