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

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': auc,
        'confusion_matrix': cm,
        'fps': fps,
        'latency_ms': latency_ms
    }
