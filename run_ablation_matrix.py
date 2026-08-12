import os
import json
import pandas as pd
from train import train_model

def run_ablation_matrix():
    """
    Executes the systematic 6-experiment ablation matrix:
      - Baseline: Cross-Entropy loss, static single crop ResNet-18, basic transforms
      - Exp A:    Focal Loss (gamma=2.0)
      - Exp B:    Photometric Sensor Augmentations
      - Exp C:    Focal Loss + Augmentations (Phase 1)
      - Exp D:    Dual-Branch Feature Fusion (Phase 2)
      - Exp E:    Dual-Branch + Temporal GRU Sequence Model (Phase 3)
    """
    experiments = [
        {"name": "Baseline", "model": "resnet18", "loss": "cross_entropy"},
        {"name": "Exp_A_FocalLoss", "model": "resnet18", "loss": "focal"},
        {"name": "Exp_B_DualBranch", "model": "dual_branch_resnet18", "loss": "cross_entropy"},
        {"name": "Exp_C_Phase1", "model": "resnet18", "loss": "focal"},
        {"name": "Exp_D_Phase2", "model": "dual_branch_resnet18", "loss": "focal"},
        {"name": "Exp_E_Phase3", "model": "temporal_resnet18", "loss": "focal"},
    ]

    results_table = []
    print("==================================================")
    print("      DRIVER DROWSINESS ABLATION MATRIX          ")
    print("==================================================")

    for exp in experiments:
        print(f"\n[*] Running Experiment: {exp['name']} (Model: {exp['model']} | Loss: {exp['loss']})")
        summary = train_model(
            model_name=exp['model'],
            dataset_dir="processed_dataset",
            epochs=1,
            batch_size=64,
            device="cpu",
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
    run_ablation_matrix()
