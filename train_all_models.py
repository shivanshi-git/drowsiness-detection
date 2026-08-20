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

# Model Architectures
from models.drowsiness_pipeline import LowLightDrowsinessPipeline
from models.backbones.resnet50 import ResNet50Baseline
from models.backbones.vit_baseline import ViTBaseline
from models.backbones.swin_baseline import SwinTransformerBaseline
from models.backbones.inceptionv3 import InceptionV3Baseline

# Data & Evaluation
from data.nthu_dataset import build_nthu_dataloaders
from evaluation.metrics import compute_comprehensive_metrics
from generate_xai_samples import generate_xai_plots


def instantiate_model(model_name: str, num_classes: int = 5, embed_dim: int = 256, sequence_length: int = 16):
    """Instantiate model based on model keyword name."""
    name_clean = model_name.lower()
    if name_clean in ["sota", "sota_pipeline", "drowsiness_pipeline"]:
        return LowLightDrowsinessPipeline(num_classes=num_classes, embed_dim=embed_dim, sequence_length=sequence_length)
    elif name_clean in ["resnet", "resnet50", "resnet-50"]:
        return ResNet50Baseline(num_classes=num_classes, pretrained=True)
    elif name_clean in ["vit", "vit_base", "vit-base"]:
        return ViTBaseline(num_classes=num_classes, pretrained=False)
    elif name_clean in ["swin", "swin_tiny", "swin-tiny"]:
        return SwinTransformerBaseline(num_classes=num_classes, pretrained=False)
    elif name_clean in ["inception", "inceptionv3", "inception_v3"]:
        return InceptionV3Baseline(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Options: ['sota', 'resnet50', 'vit', 'swin', 'inception']")


def train_single_model(model_name: str, cfg: dict, epochs_override: int = None, device: torch.device = None):
    print(f"\n======================================================================")
    print(f"               STARTING TRAINING FOR MODEL: {model_name.upper()}")
    print(f"======================================================================")

    train_cfg = cfg["training"]
    data_cfg = cfg["dataset"]
    num_epochs = epochs_override if epochs_override is not None else train_cfg["epochs"]
    save_dir = os.path.join(train_cfg["checkpoint_dir"], model_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 1. Build Dataloaders
    splits = data_cfg.get("subject_splits", {})
    train_loader, val_loader = build_nthu_dataloaders(
        root_dir=data_cfg["raw_dir"],
        batch_size=train_cfg["batch_size"],
        sequence_length=data_cfg["sequence_length"],
        frame_step=data_cfg.get("frame_step", 2),
        num_workers=train_cfg.get("num_workers", 0),
        train_subjects=splits.get("train_subjects", None),
        val_subjects=splits.get("val_subjects", None)
    )

    # 2. Instantiate Model & Optimizer
    model = instantiate_model(
        model_name=model_name,
        num_classes=data_cfg["num_classes"],
        embed_dim=cfg["pipeline"]["region_vit"]["embed_dim"],
        sequence_length=data_cfg["sequence_length"]
    ).to(device)

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
    history = []
    is_sota = "sota" in model_name.lower() or "pipeline" in model_name.lower()

    for epoch in range(1, num_epochs + 1):
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
                if is_sota:
                    out = model(video, flow)
                else:
                    out = model(video)
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
                    if is_sota:
                        out = model(video, flow)
                    else:
                        out = model(video)
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
            f"[{model_name.upper()}] Epoch [{epoch:02d}/{num_epochs:02d}] ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Train F1: {train_metrics['macro_f1']*100:.1f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']*100:.1f}%, Val F1: {val_metrics['macro_f1']*100:.1f}%",
            flush=True
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"]
        })

        if val_metrics["macro_f1"] >= best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_path = os.path.join(save_dir, f"best_{model_name}_model.pth")
            torch.save(model.state_dict(), best_path)

    final_metrics = compute_comprehensive_metrics(val_targets, val_preds, val_probs)

    # Save per-model confusion matrix & ROC curve
    from evaluation.roc_auc import plot_and_save_roc_curve
    from sklearn.metrics import confusion_matrix

    class_names = ["Normal", "Slow Blink", "Yawn", "Nodding", "Eye Closure"]
    cm = confusion_matrix(val_targets, val_preds, labels=list(range(5)))

    plt.figure(figsize=(7, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix: {model_name.upper()}")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2. if cm.max() > 0 else 1.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"results/confusion_matrix_{model_name}.png", dpi=150)
    plt.close()

    if len(val_probs) > 0:
        plot_and_save_roc_curve(val_targets, val_probs, num_classes=5, output_path=f"results/roc_curve_{model_name}.png")

    print(f"[SUCCESS] {model_name.upper()} Training Completed! Best Val Macro F1: {best_val_f1*100:.2f}%")

    return {
        "model_name": model_name,
        "best_val_f1": best_val_f1,
        "history": history,
        "final_metrics": final_metrics
    }


def train_all_models_pipeline(config_path: str = "configs/nthu_ddd_config.yaml", target_models: list = None, epochs_override: int = None):
    assert torch.cuda.is_available(), "[ERROR] CUDA GPU is required for training!"
    device = torch.device("cuda:0")
    print(f"[INFO] Using Accelerated GPU Device: {torch.cuda.get_device_name(0)}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if target_models is None or "all" in target_models:
        model_list = ["sota", "resnet50", "vit", "swin", "inception"]
    else:
        model_list = target_models

    all_results = []

    for model_name in model_list:
        res = train_single_model(model_name=model_name, cfg=cfg, epochs_override=epochs_override, device=device)
        all_results.append(res)

    # Export Comparative Benchmark Table
    summary_data = []
    for r in all_results:
        m = r["final_metrics"]
        summary_data.append({
            "Model Architecture": r["model_name"].upper(),
            "Accuracy (%)": round(m.get("accuracy", 0.0) * 100, 2),
            "Macro Precision (%)": round(m.get("macro_precision", 0.0) * 100, 2),
            "Macro Recall (%)": round(m.get("macro_recall", 0.0) * 100, 2),
            "Macro F1 (%)": round(m.get("macro_f1", 0.0) * 100, 2),
            "Weighted F1 (%)": round(m.get("weighted_f1", 0.0) * 100, 2),
        })

    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = "results/benchmark_comparison.csv"
    summary_json_path = "results/benchmark_comparison.json"
    summary_df.to_csv(summary_csv_path, index=False)

    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=4)

    # Plot Comparative Curves Across All Models
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for r in all_results:
        df_hist = pd.DataFrame(r["history"])
        plt.plot(df_hist["epoch"], df_hist["val_loss"], label=f"{r['model_name'].upper()} Val Loss", marker="o")
    plt.title("Comparative Validation Loss Across Models")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    for r in all_results:
        df_hist = pd.DataFrame(r["history"])
        plt.plot(df_hist["epoch"], df_hist["val_macro_f1"] * 100, label=f"{r['model_name'].upper()} Val Macro F1", marker="s")
    plt.title("Comparative Validation Macro F1 Score (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1 (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/comparative_training_curves.png", dpi=200)
    plt.close()

    # Generate Explainability XAI plots for the SOTA model
    sota_ckpt = "saved_models/sota/best_sota_model.pth"
    if os.path.exists(sota_ckpt):
        print("\n[INFO] Generating Explainability (XAI) Grad-CAM Visualizations for SOTA Model...")
        generate_xai_plots(
            model_checkpoint=sota_ckpt,
            output_dir="results/xai",
            num_samples=4
        )

    print("\n======================================================================")
    print("         BENCHMARK COMPARISON TABLE Across Models (NTHU-DDD)")
    print("======================================================================")
    print(summary_df.to_string(index=False))
    print(f"\n[BENCHMARK SUCCESS] Saved comparative results to '{summary_csv_path}' & plots to 'results/'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Benchmark All Models on NTHU-DDD")
    parser.add_argument("--config", default="configs/nthu_ddd_config.yaml", help="Path to config yaml")
    parser.add_argument("--models", nargs="+", default=["all"], help="Models to train: sota, resnet50, vit, swin, inception or all")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    args = parser.parse_args()

    train_all_models_pipeline(config_path=args.config, target_models=args.models, epochs_override=args.epochs)
