import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from train import train_model

MODELS = ['resnet18']

def train_and_compare_all_models(dataset_dir='processed_dataset', epochs=5, batch_size=32, lr=1e-4, device='cuda'):
    """
    Trains the canonical ResNet18 model and saves its evaluation artifacts.
    """
    overall_results = {}
    results_base_dir = "results"
    os.makedirs(results_base_dir, exist_ok=True)

    print("==========================================================")
    print("      DROWSINESS DETECTION MULTI-MODEL TRAINING PIPELINE  ")
    print("==========================================================")
    print(f"[*] Total Models to Train: {len(MODELS)}")
    print(f"[*] Models: {', '.join(MODELS)}")
    print(f"[*] Dataset Directory: {dataset_dir}")
    print(f"[*] Training Epochs: {epochs} | Batch Size: {batch_size}")
    print("==========================================================")

    for idx, model_name in enumerate(MODELS, 1):
        print(f"\n\n>>> [{idx}/{len(MODELS)}] STARTING TRAINING FOR MODEL: '{model_name}' <<<")
        try:
            summary_metrics = train_model(
                model_name=model_name,
                dataset_dir=dataset_dir,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=device
            )
            overall_results[model_name] = summary_metrics
        except Exception as e:
            print(f"[!] Error training model '{model_name}': {e}")
            overall_results[model_name] = {"error": str(e)}

    # Save aggregated overall summary JSON
    summary_json_path = os.path.join(results_base_dir, "all_models_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(overall_results, f, indent=4)
    print(f"\n[✓] Aggregated multi-model summary saved to: {summary_json_path}")

    # Generate Comparative Leaderboard Chart
    generate_leaderboard_chart(overall_results, save_path=os.path.join(results_base_dir, "model_comparison_leaderboard.png"))
    generate_comparison_markdown(overall_results, save_path=os.path.join(results_base_dir, "model_comparison_report.md"))

def generate_leaderboard_chart(results, save_path):
    """
    Plots a multi-bar metric comparison across all successfully trained models.
    """
    valid_data = []
    for model_name, metrics in results.items():
        if "accuracy" in metrics:
            valid_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'] * 100,
                'F1-Score': metrics['f1_score'] * 100,
                'ROC-AUC': metrics['roc_auc'] * 100,
                'FPS': metrics['fps']
            })

    if not valid_data:
        print("[!] No valid metrics collected for leaderboard chart generation.")
        return

    df = pd.DataFrame(valid_data)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Bar plot for Accuracy, F1, ROC-AUC
    df.plot(x='Model', y=['Accuracy', 'F1-Score', 'ROC-AUC'], kind='bar', ax=ax1, colormap='viridis', width=0.75)
    ax1.set_title('Multi-Model Performance Comparison (Accuracy, F1, ROC-AUC %)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    # Bar plot for Throughput (FPS)
    df.plot(x='Model', y='FPS', kind='bar', ax=ax2, color='crimson', width=0.4)
    ax2.set_title('Inference Speed Throughput (FPS)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Frames Per Second (FPS)', fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] Multi-Model Leaderboard Chart saved to: {save_path}")

def generate_comparison_markdown(results, save_path):
    """
    Writes a Markdown comparison report table for all trained models.
    """
    lines = [
        "# Drowsiness Detection Multi-Model Evaluation Leaderboard",
        "",
        "| Model Architecture | Accuracy (%) | Precision | Recall | F1-Score | ROC-AUC | Inference Latency (ms) | FPS Throughput |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for model_name, metrics in results.items():
        if "accuracy" in metrics:
            acc = f"{metrics['accuracy']*100:.2f}%"
            prec = f"{metrics['precision']:.4f}"
            rec = f"{metrics['recall']:.4f}"
            f1 = f"{metrics['f1_score']:.4f}"
            auc = f"{metrics['roc_auc']:.4f}"
            lat = f"{metrics['latency_ms']:.2f}"
            fps = f"{metrics['fps']:.1f}"
            lines.append(f"| **{model_name}** | {acc} | {prec} | {rec} | {f1} | {auc} | {lat} | {fps} |")
        else:
            lines.append(f"| **{model_name}** | Error | N/A | N/A | N/A | N/A | N/A | N/A |")

    with open(save_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[✓] Multi-Model Comparison Report written to: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train and compare all drowsiness detection models.")
    parser.add_argument("--dataset_dir", type=str, default="processed_dataset")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    train_and_compare_all_models(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )
