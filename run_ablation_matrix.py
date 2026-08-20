import os
import json
import pandas as pd
from train import train_model

import os
import json
import argparse
import torch
import pandas as pd
from train import train_model

def run_ablation_matrix(epochs=25, batch_size=64, device="cuda"):
    """
    Executes the systematic 6-experiment ablation matrix on GPU:
      - Baseline: Cross-Entropy loss, static single crop ResNet-18, basic transforms
      - Exp A:    Focal Loss (gamma=2.0)
      - Exp B:    Dual-Branch ResNet (Cross-Entropy)
      - Exp C:    Focal Loss + Augmentations (Phase 1)
      - Exp D:    Dual-Branch Feature Fusion + Focal Loss (Phase 2)
    - Temporal sequence training is intentionally excluded until a sequence dataset is available.
    """
    if device == 'cuda' and not torch.cuda.is_available():
        print("[!] Warning: CUDA requested but PyTorch cannot access GPU. Falling back to CPU.")
        device = 'cpu'

    print(f"[*] Executing Ablation Matrix on Target Hardware: {device.upper()}")
    if device == 'cuda':
        print(f"[*] GPU Acceleration: {torch.cuda.get_device_name(0)}")

    # Calculate and print dataset disk size and image count before starting training
    dataset_dir = "processed_dataset"
    if os.path.exists(dataset_dir):
        total_bytes = sum(os.path.getsize(os.path.join(r, f)) for r, _, files in os.walk(dataset_dir) for f in files if os.path.isfile(os.path.join(r, f)))
        total_files = sum(len(files) for _, _, files in os.walk(dataset_dir))
        size_mb = total_bytes / (1024 * 1024)
        size_gb = total_bytes / (1024 * 1024 * 1024)
        size_str = f"{size_gb:.2f} GB" if size_gb >= 1.0 else f"{size_mb:.2f} MB"

        print("==================================================")
        print("          TRAINING DATASET METRICS                ")
        print("==================================================")
        print(f"Dataset Directory:   {dataset_dir}")
        print(f"Total Disk Size:     {size_str} ({total_bytes:,} bytes)")
        print(f"Total Image Files:   {total_files:,} images")
        print("==================================================\n")

    experiments = [
        {"name": "Baseline", "model": "resnet18", "loss": "cross_entropy"},
        {"name": "Exp_A_FocalLoss", "model": "resnet18", "loss": "focal"},
        {"name": "Exp_B_DualBranch", "model": "dual_branch_resnet18", "loss": "cross_entropy"},
        {"name": "Exp_C_Phase1", "model": "resnet18", "loss": "focal"},
        {"name": "Exp_D_Phase2", "model": "dual_branch_resnet18", "loss": "focal"},
    ]

    results_table = []
    print("==================================================")
    print("      DRIVER DROWSINESS ABLATION MATRIX          ")
    print("==================================================")

    for exp in experiments:
        print(f"\n[*] Running Experiment: {exp['name']} (Model: {exp['model']} | Loss: {exp['loss']} | Device: {device})")
        summary = train_model(
            model_name=exp['model'],
            dataset_dir="processed_dataset",
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            loss_type=exp['loss']
        )
        
        results_table.append({
            "Experiment": exp['name'],
            "Model": exp['model'],
            "Loss": exp['loss'].upper(),
            "Accuracy": f"{summary.get('accuracy', 0.0)*100:.2f}%",
            "Precision": f"{summary.get('precision', 0.0):.4f}",
            "Recall (Sensitivity)": f"{summary.get('recall', 0.0):.4f}",
            "Specificity": f"{summary.get('specificity', 0.0):.4f}",
            "F1-Score": f"{summary.get('f1_score', 0.0):.4f}",
            "ROC-AUC": f"{summary.get('roc_auc', 0.0):.4f}"
        })

    df_ablation = pd.DataFrame(results_table)
    print("\n\n==================================================")
    print("             FINAL ABLATION RESULTS               ")
    print("==================================================")
    print(df_ablation.to_string(index=False))

    os.makedirs("results", exist_ok=True)
    df_ablation.to_csv("results/ablation_matrix_summary.csv", index=False)
    print("\n[✓] Ablation Matrix saved to 'results/ablation_matrix_summary.csv'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Driver Drowsiness System Ablation Matrix on GPU.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs per ablation experiment.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation.")
    parser.add_argument("--device", type=str, default="cuda", help="Target computing device ('cuda' or 'cpu').")
    args = parser.parse_args()

    run_ablation_matrix(epochs=args.epochs, batch_size=args.batch_size, device=args.device)
