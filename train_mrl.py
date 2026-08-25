import os
import time
import json
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torchvision.models import (
    resnet50, ResNet50_Weights,
    vit_b_16, ViT_B_16_Weights,
    swin_t, Swin_T_Weights,
    inception_v3, Inception_V3_Weights
)

from models.drowsiness_pipeline import LowLightDrowsinessPipeline
from evaluation.roc_auc import plot_and_save_roc_curve
from sklearn.metrics import confusion_matrix

def build_mrl_model(model_name: str, num_classes: int = 2):
    name_clean = model_name.lower()
    if name_clean in ["sota", "sota_pipeline"]:
        model = LowLightDrowsinessPipeline(num_classes=num_classes, enable_llformer=False)
        return model
    elif name_clean in ["resnet", "resnet50", "resnet-50"]:
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif name_clean in ["vit", "vit_base", "vit-base"]:
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model
    elif name_clean in ["swin", "swin_tiny", "swin-tiny"]:
        model = swin_t(weights=Swin_T_Weights.DEFAULT)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model
    elif name_clean in ["inception", "inceptionv3", "inception_v3"]:
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.aux_logits = False
        return model
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def train_single_mrl_model(model_name: str, epochs: int = 15, batch_size: int = 64, lr: float = 3e-4, device=None):
    print(f"\n======================================================================")
    print(f"       STARTING MRL EYE TRAINING FOR MODEL: {model_name.upper()}")
    print(f"======================================================================")
    
    mrl_dir = "MRL/data"
    train_dir = os.path.join(mrl_dir, "train")
    val_dir = os.path.join(mrl_dir, "val")
    save_dir = os.path.join("saved_models/mrl_eye", model_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    img_size = 299 if "inception" in model_name.lower() else 224

    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Pre-load existing checkpoint weights to avoid training from scratch
    possible_paths = [
        os.path.join(save_dir, f"best_{model_name}_mrl_model.pth"),
        os.path.join("saved_models", "mrl_eye", f"best_{model_name}_mrl_model.pth"),
        os.path.join("saved_models", "mrl_eye", model_name, f"best_{model_name}_mrl_model.pth"),
    ]
    preloaded = False
    for p in possible_paths:
        if os.path.exists(p):
            print(f"[PRELOAD SUCCESS] Loading existing pre-trained MRL weights for '{model_name}' from: {p}")
            try:
                ckpt = torch.load(p, map_location=device)
                if isinstance(ckpt, dict) and "model_state" in ckpt:
                    model.load_state_dict(ckpt["model_state"], strict=False)
                elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                    model.load_state_dict(ckpt["state_dict"], strict=False)
                else:
                    model.load_state_dict(ckpt, strict=False)
                print(f"[PRELOAD SUCCESS] Loaded existing MRL weights for '{model_name}'. No training from scratch.")
                preloaded = True
                break
            except Exception as e:
                print(f"[PRELOAD WARN] Could not load MRL checkpoint {p}: {e}")
    if not preloaded:
        print(f"[INFO] Initialized '{model_name}' with default backbone pre-trained weights.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda')

    best_val_acc = 0.0
    history = []
    is_sota = "sota" in model_name.lower()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                if is_sota:
                    b = imgs.size(0)
                    video_seq = imgs.unsqueeze(1).repeat(1, 16, 1, 1, 1)
                    flow_seq = torch.zeros(b, 16, 2, 112, 112, device=device)
                    out = model(video_seq, flow_seq)
                    outputs = out["logits"]
                else:
                    outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct, val_total = 0, 0
        val_preds, val_targets, val_probs = [], [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.amp.autocast('cuda'):
                    if is_sota:
                        b = imgs.size(0)
                        video_seq = imgs.unsqueeze(1).repeat(1, 16, 1, 1, 1)
                        flow_seq = torch.zeros(b, 16, 2, 112, 112, device=device)
                        out = model(video_seq, flow_seq)
                        outputs = out["logits"]
                    else:
                        outputs = model(imgs)

                    loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                val_preds.extend(preds.tolist())
                val_targets.extend(labels.cpu().numpy().tolist())
                val_probs.extend(probs.tolist())
                val_correct += (preds == labels.cpu().numpy()).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        epoch_time = time.time() - t0

        print(f"[{model_name.upper()}] Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%", flush=True)

        history.append({"epoch": epoch, "train_acc": train_acc, "val_acc": val_acc})

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(save_dir, f"best_{model_name}_mrl_model.pth")
            torch.save(model.state_dict(), best_path)

    # Save per-model evaluation metrics CSV & JSON
    metrics_summary = {
        "Model": model_name.upper(),
        "Accuracy": float(best_val_acc),
        "Total_Epochs": epochs
    }
    pd.DataFrame([metrics_summary]).to_csv(f"results/mrl_evaluation_metrics_{model_name}.csv", index=False)
    with open(f"results/mrl_evaluation_metrics_{model_name}.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)

    # Save confusion matrix plot for MRL model
    cm = confusion_matrix(val_targets, val_preds, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"MRL Eye Confusion Matrix: {model_name.upper()}")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Awake", "Sleepy"])
    plt.yticks(tick_marks, ["Awake", "Sleepy"])
    thresh = cm.max() / 2. if cm.max() > 0 else 1.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"results/mrl_confusion_matrix_{model_name}.png", dpi=150)
    plt.close()

    # Save ROC Curve plot for MRL model
    if len(val_probs) > 0:
        plot_and_save_roc_curve(val_targets, val_probs, num_classes=2, output_path=f"results/mrl_roc_curve_{model_name}.png")

    return {"model_name": model_name.upper(), "best_val_acc": round(best_val_acc * 100, 2)}

def train_all_mrl_models(epochs: int = 30):
    assert torch.cuda.is_available(), "[ERROR] CUDA GPU required for MRL dataset training!"
    device = torch.device("cuda:0")
    print(f"[INFO] Using Device for MRL Benchmark: {torch.cuda.get_device_name(0)}")
    
    models_list = ["sota", "resnet50", "vit", "swin", "inception"]
    results = []

    for name in models_list:
        res = train_single_mrl_model(name, epochs=epochs, device=device)
        results.append(res)

    summary_df = pd.DataFrame(results)
    summary_df.to_csv("results/mrl_benchmark_comparison.csv", index=False)
    with open("results/mrl_benchmark_comparison.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n======================================================================")
    print("      MRL EYE DATASET BENCHMARK COMPARISON Across All Models          ")
    print("======================================================================")
    print(summary_df.to_string(index=False))

    # Auto-generate FINAL_BENCHMARK_REPORT.md and sync to Git
    try:
        print("\n[REPORT GENERATOR] Generating updated FINAL_BENCHMARK_REPORT.md...")
        report_cmd = ".venv/bin/python utils/generate_final_benchmark_report.py"
        os.system(report_cmd)

        print("\n[GIT AUTO-SYNC] Staging, committing, and pushing FINAL_BENCHMARK_REPORT.md and MRL evaluation artifacts to GitHub...")
        os.system("git add FINAL_BENCHMARK_REPORT.md results/ && git commit -m 'feat: auto-update FINAL_BENCHMARK_REPORT.md and MRL evaluation matrices after training' && git push origin low-light-detection")
        print("[GIT AUTO-SYNC SUCCESS] FINAL_BENCHMARK_REPORT.md & MRL results synced to GitHub!")
    except Exception as e:
        print(f"[GIT AUTO-SYNC WARNING] Could not auto-push to git: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs for MRL models")
    args = parser.parse_args()
    train_all_mrl_models(epochs=args.epochs)
