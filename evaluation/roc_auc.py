import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def plot_and_save_roc_curve(y_true: list, y_probs: list, num_classes: int = 5, output_path: str = "roc_curve.png"):
    """
    Computes One-vs-Rest or Binary ROC curve and AUC.
    """
    y_true_np = np.array(y_true)
    y_probs_np = np.array(y_probs)

    plt.figure(figsize=(6, 5))

    if num_classes == 2:
        if y_probs_np.ndim > 1 and y_probs_np.shape[1] > 1:
            scores = y_probs_np[:, 1]
        else:
            scores = y_probs_np.ravel()
        
        fpr, tpr, _ = roc_curve(y_true_np, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
        plt.title("MRL Eye ROC Curve")
    else:
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        if y_true_bin.shape[1] == 1:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs_np[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"Class {i} (AUC = {roc_auc:.4f})")
        plt.title("Multi-Class ROC Curves (OvR)")

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[EVAL] Saved ROC curve to {output_path}")

