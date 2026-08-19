import os
import time
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from models.drowsiness_pipeline import LowLightDrowsinessPipeline
from data.nthu_dataset import build_nthu_dataloaders
from evaluation.metrics import compute_comprehensive_metrics


def train_sota_pipeline(config_path: str = "configs/nthu_ddd_config.yaml"):
    # Load configuration
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Initializing Training on Device: {device}")

    train_cfg = cfg["training"]
    data_cfg = cfg["dataset"]
    save_dir = train_cfg["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # 1. Build DataLoaders
    train_loader, val_loader = build_nthu_dataloaders(
        root_dir=data_cfg["raw_dir"],
        batch_size=train_cfg["batch_size"],
        sequence_length=data_cfg["sequence_length"],
        num_workers=train_cfg["num_workers"]
    )
    print(f"[INFO] Loaded {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples.")

    # 2. Build Pipeline Model
    model = LowLightDrowsinessPipeline(
        num_classes=data_cfg["num_classes"],
        embed_dim=cfg["pipeline"]["region_vit"]["embed_dim"],
        sequence_length=data_cfg["sequence_length"]
    ).to(device)

    # 3. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["epochs"],
        eta_min=float(train_cfg["min_lr"])
    )
    scaler = GradScaler(enabled=train_cfg["mixed_precision"] and (device == "cuda"))

    best_val_f1 = 0.0
    history = []

    print("\n--- Starting Training Loop ---")
    for epoch in range(1, train_cfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []

        for batch in train_loader:
            video = batch["video"].to(device)
            flow = batch["flow"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            with autocast(enabled=train_cfg["mixed_precision"] and (device == "cuda")):
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

                with autocast(enabled=train_cfg["mixed_precision"] and (device == "cuda")):
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
            f"Epoch [{epoch:02d}/{train_cfg['epochs']:02d}] ({epoch_time:.1f}s) | "
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

        # Save Best Model Checkpoint
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_path = os.path.join(save_dir, "best_sota_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  [+] Saved new best model checkpoint -> {best_path} (Val F1: {best_val_f1*100:.2f}%)")

    # Save training logs
    pd.DataFrame(history).to_csv(os.path.join(save_dir, "training_history.csv"), index=False)
    print(f"\n[DONE] Training complete. Best Validation F1: {best_val_f1*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nthu_ddd_config.yaml", help="Path to config yaml")
    args = parser.parse_args()

    train_sota_pipeline(config_path=args.config)
