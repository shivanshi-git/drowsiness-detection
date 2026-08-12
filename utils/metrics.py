import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

def evaluate_model_performance(model, dataloader, device='cpu'):
    """
    Evaluates model on dataloader and returns:
    - accuracy, precision, recall, f1, auc
    - average latency per image (ms)
    - FPS throughput
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []
    all_probs = []

    start_time = time.time()
    total_images = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            total_images += inputs.size(0)

    elapsed = time.time() - start_time
    fps = total_images / elapsed if elapsed > 0 else 0
    latency_ms = (elapsed / total_images) * 1000 if total_images > 0 else 0

    acc = accuracy_score(all_targets, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary', zero_division=0)
    
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.5

    cm = confusion_matrix(all_targets, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'specificity': specificity,
        'f1_score': f1,
        'roc_auc': auc,
        'confusion_matrix': cm,
        'all_targets': all_targets,
        'all_preds': all_preds,
        'all_probs': all_probs,
        'fps': fps,
        'latency_ms': latency_ms
    }

def save_evaluation_matrix(metrics, history=None, class_names=['alert', 'drowsy'], output_dir='results'):
    """
    Saves complete evaluation matrix:
    - Confusion Matrix (PNG)
    - ROC Curve (PNG)
    - Training/Validation Loss & Accuracy curves (PNG if history provided)
    - Detailed Evaluation Report (JSON & TXT)
    """
    import os
    import json
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, roc_curve, auc

    os.makedirs(output_dir, exist_ok=True)

    # 1. Plot Confusion Matrix
    plt.figure(figsize=(7, 6))
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Evaluation Matrix - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 2. Plot ROC Curve
    plt.figure(figsize=(7, 6))
    fpr, tpr, _ = roc_curve(metrics['all_targets'], metrics['all_probs'])
    roc_auc_val = metrics['roc_auc']
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc_val:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Evaluation Matrix - Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # 3. Plot Training Curves & Export Per-Epoch CSV if history available
    epoch_details = []
    if history and 'train_loss' in history and len(history['train_loss']) > 0:
        import pandas as pd
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        epochs = list(range(1, len(history['train_loss']) + 1))

        ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
        ax1.plot(epochs, history['val_loss'], 'r-s', label='Val Loss')
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, [a * 100 for a in history['train_acc']], 'b-o', label='Train Accuracy (%)')
        ax2.plot(epochs, [a * 100 for a in history['val_acc']], 'r-s', label='Val Accuracy (%)')
        ax2.set_title('Accuracy Curves (%)')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        curves_path = os.path.join(output_dir, 'training_curves.png')
        plt.savefig(curves_path, dpi=300)
        plt.close()

        # Build epoch details list
        for ep_idx in range(len(history['train_loss'])):
            epoch_details.append({
                'epoch': ep_idx + 1,
                'train_loss': round(float(history['train_loss'][ep_idx]), 4),
                'train_acc_pct': round(float(history['train_acc'][ep_idx]) * 100, 2),
                'val_loss': round(float(history['val_loss'][ep_idx]), 4),
                'val_acc_pct': round(float(history['val_acc'][ep_idx]) * 100, 2)
            })

        # Save CSV Spreadsheet of per-epoch details
        csv_path = os.path.join(output_dir, 'epoch_training_history.csv')
        df_history = pd.DataFrame(epoch_details)
        df_history.to_csv(csv_path, index=False)

    # 4. Generate Classification Report & JSON Summary
    clf_report_str = classification_report(metrics['all_targets'], metrics['all_preds'], target_names=class_names)
    
    summary_dict = {
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'specificity': float(metrics.get('specificity', 0.0)),
        'f1_score': float(metrics['f1_score']),
        'roc_auc': float(metrics['roc_auc']),
        'fps': float(metrics['fps']),
        'latency_ms': float(metrics['latency_ms']),
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'epoch_history': epoch_details
    }

    json_path = os.path.join(output_dir, 'evaluation_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary_dict, f, indent=4)

    txt_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(txt_path, 'w') as f:
        f.write("==================================================\n")
        f.write("         MODEL EVALUATION MATRIX REPORT           \n")
        f.write("==================================================\n\n")
        f.write(f"Overall Accuracy:  {metrics['accuracy']*100:.2f}%\n")
        f.write(f"Precision:         {metrics['precision']:.4f}\n")
        f.write(f"Recall:            {metrics['recall']:.4f}\n")
        f.write(f"Specificity:       {metrics.get('specificity', 0.0):.4f}\n")
        f.write(f"F1-Score:          {metrics['f1_score']:.4f}\n")
        f.write(f"ROC-AUC:           {metrics['roc_auc']:.4f}\n")
        f.write(f"Inference Latency: {metrics['latency_ms']:.2f} ms/sample\n")
        f.write(f"FPS Throughput:    {metrics['fps']:.1f} FPS\n\n")
        if epoch_details:
            f.write("Epoch-by-Epoch Training Details:\n")
            f.write("Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%)\n")
            f.write("-----------------------------------------------------------\n")
            for ep in epoch_details:
                f.write(f" {ep['epoch']:02d}   |   {ep['train_loss']:.4f}   |    {ep['train_acc_pct']:6.2f}%   |  {ep['val_loss']:.4f}  |   {ep['val_acc_pct']:6.2f}%\n")
            f.write("\n")
        f.write("Classification Report:\n")
        f.write(clf_report_str)

    print(f"\n[✓] Evaluation Matrix artifacts generated successfully in '{output_dir}/':")
    print(f"    - {cm_path}")
    print(f"    - {roc_path}")
    if history:
        print(f"    - {os.path.join(output_dir, 'training_curves.png')}")
        print(f"    - {os.path.join(output_dir, 'epoch_training_history.csv')}")
    print(f"    - {json_path}")
    print(f"    - {txt_path}\n")

    return summary_dict

