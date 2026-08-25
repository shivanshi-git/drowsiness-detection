import os
import copy
import math
import time
import json
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt

from models.drowsiness_pipeline import LowLightDrowsinessPipeline
from data.nthu_dataset import build_nthu_dataloaders
from training.losses import MultimodalDrowsinessLoss
from evaluation.metrics import compute_comprehensive_metrics


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


def _compute_class_weights(train_loader, num_classes=5):
    """Inverse-frequency class weights for loss weighting."""
    try:
        labels = [s["label"] for s in train_loader.dataset.samples]
        counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
        weights = 1.0 / counts.clamp(min=1)
        weights = weights / weights.sum() * num_classes
        print(f"[INFO] Class weights: {[f'{w:.3f}' for w in weights.tolist()]}")
        return weights
    except Exception as e:
        print(f"[INFO] Using uniform class weights ({e})")
        return None


def _mixup_batch(video, flow, labels, alpha=0.4, device="cuda"):
    """Temporal clip-level MixUp augmentation."""
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    idx = torch.randperm(video.size(0), device=device)
    return (lam * video + (1 - lam) * video[idx],
            lam * flow  + (1 - lam) * flow[idx],
            labels, labels[idx], lam)


def train_sota_pipeline(config_path: str = "configs/nthu_ddd_config.yaml", epochs_override: int = None):
    if not torch.cuda.is_available():
        raise RuntimeError("[STRICT GPU REQUIREMENT ERROR] CUDA GPU is not available for training.")
    device = torch.device("cuda:0")
    print(f"[INFO] Strictly using GPU: {torch.cuda.get_device_name(0)}")

    # Load configuration
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]
    data_cfg  = cfg["dataset"]
    save_dir  = train_cfg["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    num_epochs      = epochs_override if epochs_override is not None else train_cfg["epochs"]
    num_classes     = data_cfg["num_classes"]
    use_amp         = train_cfg.get("mixed_precision", True) and (device.type == "cuda")
    use_mixup       = train_cfg.get("use_mixup", True)
    mixup_alpha     = float(train_cfg.get("mixup_alpha", 0.4))
    ema_decay       = float(train_cfg.get("ema_decay", 0.9995))
    patience        = int(train_cfg.get("early_stop_patience", 12))
    warmup_epochs   = int(train_cfg.get("warmup_epochs", 5))
    grad_clip       = float(train_cfg.get("gradient_clip_norm", 1.0))
    base_lr         = float(train_cfg.get("learning_rate", train_cfg.get("lr", 1e-4)))
    wd              = float(train_cfg.get("weight_decay", 1e-4))

    # 1. Build DataLoaders
    train_loader, val_loader = build_nthu_dataloaders(
        root_dir=data_cfg["raw_dir"],
        batch_size=train_cfg["batch_size"],
        sequence_length=data_cfg["sequence_length"],
        frame_step=data_cfg.get("frame_step", 2),
        num_workers=train_cfg.get("num_workers", 0)
    )
    print(f"[INFO] Train: {len(train_loader.dataset)} samples | Val: {len(val_loader.dataset)} samples")

    # 2. Build Pipeline Model
    model = LowLightDrowsinessPipeline(
        num_classes=num_classes,
        embed_dim=cfg["pipeline"]["region_vit"]["embed_dim"],
        sequence_length=data_cfg["sequence_length"]
    ).to(device)

    # 3. Loss — Label Smoothing + Focal + binary fatigue BCE
    class_weights = _compute_class_weights(train_loader, num_classes)
    criterion = MultimodalDrowsinessLoss(
        num_classes=num_classes,
        focal_gamma=2.0,
        label_smoothing=0.1,
        focal_weight=0.5,
        class_weights=class_weights.to(device) if class_weights is not None else None
    )

    # 4. Optimizer with param groups: lower LR for LLFormer
    llf_params, other_params = [], []
    for name, p in model.named_parameters():
        (llf_params if "llformer" in name else other_params).append(p)
    optimizer = torch.optim.AdamW([
        {"params": llf_params,   "lr": base_lr * 0.1, "weight_decay": wd},
        {"params": other_params, "lr": base_lr,        "weight_decay": wd},
    ])

    # 5. Warmup + Cosine LR schedule (step-level)
    total_steps  = num_epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler('cuda', enabled=use_amp)

    # 6. EMA model
    ema_model = copy.deepcopy(model).to(device)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    best_val_acc = 0.0
    best_val_f1  = 0.0
    no_improve   = 0
    start_epoch  = 1
    history      = []
    global_step  = 0

    # Auto-Resume Checkpoint Logic
    latest_ckpt_path = os.path.join(save_dir, "checkpoint_latest.pth")
    sota_best_path = os.path.join(save_dir, "best_sota_model.pth")
    alt_sota_best_path = os.path.join(save_dir, "sota", "best_sota_model.pth")

    if os.path.exists(latest_ckpt_path):
        print(f"[INFO] Resuming training from full checkpoint: {latest_ckpt_path}")
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        ema_model.load_state_dict(ckpt.get("ema_state", ckpt["model_state"]))
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch   = ckpt.get("epoch", 0) + 1
        best_val_acc  = ckpt.get("best_val_acc", 0.0)
        best_val_f1   = ckpt.get("best_val_f1",  0.0)
        global_step   = ckpt.get("global_step",  0)
        history       = ckpt.get("history", [])
        no_improve    = ckpt.get("no_improve", 0)
        print(f"[INFO] Resumed epoch {start_epoch}/{num_epochs} | Best Acc={best_val_acc*100:.2f}%")
    else:
        preload_path = sota_best_path if os.path.exists(sota_best_path) else (alt_sota_best_path if os.path.exists(alt_sota_best_path) else None)
        if preload_path:
            print(f"[INFO] Pre-loading weights from existing best model: {preload_path}")
            ckpt_state = torch.load(preload_path, map_location=device)
            model.load_state_dict(ckpt_state)
            ema_model.load_state_dict(ckpt_state)

    print("\n--- Starting Training ---")
    val_preds, val_targets, val_probs = [], [], []

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []

        for batch in train_loader:
            video  = batch["video"].to(device)
            flow   = batch["flow"].to(device)
            labels = batch["label"].to(device)

            # MixUp augmentation
            do_mix = use_mixup and (torch.rand(1).item() < 0.5)
            if do_mix:
                video, flow, labels_a, labels_b, lam = _mixup_batch(
                    video, flow, labels, alpha=mixup_alpha, device=device
                )

            optimizer.zero_grad()
            with autocast('cuda', enabled=use_amp):
                out    = model(video, flow)
                logits = out["logits"]
                fatigue = out.get("fatigue_score", None)
                if do_mix:
                    loss = (lam * criterion(logits, labels_a, fatigue) +
                            (1 - lam) * criterion(logits, labels_b, fatigue))
                    batch_labels = labels_a  # for accuracy tracking
                else:
                    loss = criterion(logits, labels, fatigue)
                    batch_labels = labels

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            # EMA update
            for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
                ema_p.data.mul_(ema_decay).add_(model_p.data, alpha=1.0 - ema_decay)

            running_loss += loss.item() * video.size(0)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            train_preds.extend(preds.tolist())
            train_targets.extend(batch_labels.cpu().numpy().tolist())
        train_loss = running_loss / max(1, len(train_loader.dataset))
        train_metrics = compute_comprehensive_metrics(train_targets, train_preds)

        # Validation Phase — use EMA model for stable evaluation
        ema_model.eval()
        val_loss = 0.0
        val_preds, val_targets, val_probs = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                video  = batch["video"].to(device)
                flow   = batch["flow"].to(device)
                labels = batch["label"].to(device)

                with autocast('cuda', enabled=use_amp):
                    out  = ema_model(video, flow)
                    loss = criterion(out["logits"], labels)

                val_loss += loss.item() * video.size(0)
                probs = torch.softmax(out["logits"], dim=1).cpu().numpy()
                preds = torch.argmax(out["logits"], dim=1).cpu().numpy()

                val_preds.extend(preds.tolist())
                val_targets.extend(labels.cpu().numpy().tolist())
                val_probs.extend(probs.tolist())

        val_loss    = val_loss / max(1, len(val_loader.dataset))
        val_metrics = compute_comprehensive_metrics(val_targets, val_preds, val_probs)
        val_acc     = val_metrics["accuracy"]
        val_f1      = val_metrics["macro_f1"]
        cur_lr      = optimizer.param_groups[1]["lr"]
        epoch_time  = time.time() - t0

        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] ({epoch_time:.1f}s) LR={cur_lr:.2e} | "
            f"Train Loss: {train_loss:.4f}, Train F1: {train_metrics['macro_f1']*100:.1f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.1f}%, Val F1: {val_f1*100:.1f}%"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1
        })

        # Save resumable latest checkpoint (includes EMA + early stop state)
        torch.save({
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "ema_state":       ema_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state":    scaler.state_dict(),
            "best_val_acc":    best_val_acc,
            "best_val_f1":     best_val_f1,
            "global_step":     global_step,
            "no_improve":      no_improve,
            "history":         history
        }, latest_ckpt_path)

        # Save best model (by accuracy) — save EMA weights
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1  = val_f1
            no_improve   = 0
            best_path = os.path.join(save_dir, "best_sota_model.pth")
            torch.save(ema_model.state_dict(), best_path)
            print(f"  ✅ New best! Val Acc={best_val_acc*100:.2f}%  Val F1={best_val_f1*100:.2f}%  → {best_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n[EARLY STOP] No improvement for {patience} epochs. Stopping at epoch {epoch}.")
                break

    # Save training history & plots
    pd.DataFrame(history).to_csv(os.path.join(save_dir, "training_history.csv"), index=False)
    plot_and_save_artifacts(history, val_targets, val_preds, val_probs=val_probs, results_dir="results")
    print(f"\n[SUCCESS] Training complete! Best Val Accuracy: {best_val_acc*100:.2f}%  |  Best Val F1: {best_val_f1*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nthu_ddd_config.yaml", help="Path to config yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    args = parser.parse_args()

    train_sota_pipeline(config_path=args.config, epochs_override=args.epochs)
