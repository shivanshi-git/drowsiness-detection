# 🧠 Explainable AI (XAI) Rigor & Evaluation Strategy

> **Project:** Driver Drowsiness Detection System with Explainable AI (XAI)  
> **Document Purpose:** Technical strategy to eliminate 4x4 spatial grid degradation on 128x128 crops, replace qualitative confirmation bias with quantitative XAI metrics, and unify cross-paradigm explanations (CNN Grad-CAM vs. ViT Attention).

---

## 📌 Executive Summary

Explainable AI (XAI) visual heatmaps are frequently presented purely as qualitative "eye tests" (e.g., *it looks like the heatmap is near the eye*). In tight 128x128 crop domains, standard XAI methods suffer from three severe technical flaws:

1. **Spatial Resolution Collapse:** Deep conv layers downsample 128x128 inputs down to a coarse 4x4 spatial grid, creating uninformative, blurry "blob" heatmaps upon upsampling.
2. **Qualitative Confirmation Bias:** No quantitative metrics (Pointing Game, Deletion AUC, Mask IoU) are used to empirically validate heatmap accuracy against ground-truth facial landmarks.
3. **Cross-Paradigm Incompatibility:** Comparing CNN **Grad-CAM** (feature map gradient activations) side-by-side with Vision Transformer **Attention Rollout** (self-attention matrix products) compares mathematically incompatible signals.

This document establishes a **High-Resolution XAI Pipeline**, introduces **Quantitative XAI Benchmark Metrics**, and defines a **Unified Cross-Paradigm Explanation Framework**.

---

## 🚨 Diagnosis of XAI Flaws

### 1. Spatial Feature Map Degradation (4x4 Grid Problem)
* **The Physics:** Standard CNNs (ResNet50, MobileNetV3, VGG16) downsample input images by 32x through successive pooling and stride-2 convolutions.
* **The Impact on 128x128 Crops:** 128 / 32 = 4x4 spatial feature map.
  Upsampling a tiny 4x4 grid back to a 128x128 image creates giant, round, featureless blobs. Fine micro-structures (eyelid boundaries, pupil dilation, or upper lip yawn curvature) are completely erased in a 4x4 representation.

### 2. Qualitative Confirmation Bias (Eyeballing vs. Science)
* **The Pitfall:** Displaying heatmaps in a dashboard and subjectively declaring "the model looks at the eye."
* **The Risk:** Neural networks often focus on spurious background pixels or crop borders while still outputting a blob that overlaps part of the eye by coincidence. Without quantitative metrics, there is zero empirical proof that heatmaps reflect true feature attribution.

### 3. Incompatible Cross-Paradigm Signals (Grad-CAM vs. Attention Rollout)
* **Grad-CAM (CNNs):** Measures the gradient of the class output score with respect to feature activation maps.
* **Attention Rollout (ViTs):** Multiplies raw self-attention weight matrices across transformer layers, measuring information flow through QKV attention heads.
* **The Flaw:** Attention Rollout ignores class gradients (it shows where the model *looks*, not what drove the *drowsy class decision*). Comparing CNN Grad-CAM to ViT Attention Rollout is comparing apples to oranges.

---

## 🛠️ The 3-Step Engineering Solutions

### 1. High-Resolution XAI Architecture (Grad-CAM++ & LayerCAM)

To eliminate 4x4 spatial degradation on 128x128 crops, we implement two targeted fixes:

1. **Mid-Level Layer Targeting:** Instead of probing only the final layer (`layer4` / 4x4), we extract heatmaps from `layer3` (8x8 grid) and fuse them with `layer4`.
2. **LayerCAM / Grad-CAM++:** **LayerCAM** computes element-wise positive weights rather than global spatial averages, yielding fine-grained spatial attribution maps that preserve sharp eyelid boundaries.

---

### 2. Quantitative XAI Benchmark Suite

We validate heatmap attribution empirically using three quantitative metrics calculated against **MediaPipe landmark polygon masks**:

#### A. Pointing Game Accuracy
Checks if the maximum attribution peak falls inside the ground-truth eye mask:
Returns 1 if peak is inside landmark mask, 0 otherwise. Pointing Game Accuracy is the hit rate across all test samples.

#### B. Deletion & Insertion AUC (Area Under Curve)
- **Deletion AUC:** Progressively mask out top-attributed pixels (in 10% steps) and measure how fast class probability drops. A steeper drop = higher quality heatmap.
- **Insertion AUC:** Progressively insert top-attributed pixels into a blank canvas and measure how fast probability recovers.

#### C. Bounding Box & Landmark Mask IoU
Binarize the normalized heatmap at threshold tau = 0.5 and compute Intersection over Union (IoU) with the MediaPipe eye landmark mask.

---

### 3. Unified Cross-Paradigm XAI (ViT-Grad-CAM & Integrated Gradients)

To enable mathematically valid cross-paradigm comparisons between CNNs and Vision Transformers:

1. **ViT-Grad-CAM:** Compute gradients of the class score with respect to the output token embeddings of the last Transformer block (`blocks[-1].norm1`). Reshape patch tokens back to a 2D spatial grid.
2. **Integrated Gradients (Model-Agnostic):** Integrates gradients along the straight path from a blank baseline to the input crop. Identical mathematical formulation applied to both CNNs and ViTs.

---

## 💻 Code Implementation

### Quantitative XAI Auditor (`xai/quantitative_eval.py`)

```python
import torch
import numpy as np
import cv2

class XAIEvaluator:
    """
    Computes Pointing Game Accuracy, Deletion AUC, and Mask IoU 
    to quantitatively validate Explainable AI heatmaps.
    """
    @staticmethod
    def pointing_game_hit(heatmap: np.ndarray, landmark_mask: np.ndarray) -> bool:
        """
        Returns True if highest activation peak falls within ground-truth landmark mask.
        """
        y_max, x_max = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        return bool(landmark_mask[y_max, x_max] > 0)

    @staticmethod
    def compute_mask_iou(heatmap: np.ndarray, landmark_mask: np.ndarray, threshold: float = 0.5) -> float:
        """
        Computes IoU between binarized heatmap (> threshold) and landmark polygon mask.
        """
        binarized_h = (heatmap >= threshold).astype(np.uint8)
        binarized_m = (landmark_mask > 0).astype(np.uint8)
        
        intersection = np.logical_and(binarized_h, binarized_m).sum()
        union = np.logical_or(binarized_h, binarized_m).sum()
        
        if union == 0:
            return 0.0
        return float(intersection / union)

    @staticmethod
    def deletion_auc(model, image_tensor: torch.Tensor, heatmap: np.ndarray, target_class: int, steps: int = 10) -> float:
        """
        Progressively removes top-attributed pixels and measures probability drop.
        """
        model.eval()
        h, w = heatmap.shape
        flat_indices = np.argsort(heatmap.flatten())[::-1]
        
        probabilities = []
        img_copy = image_tensor.clone().squeeze(0)
        step_size = len(flat_indices) // steps
        
        with torch.no_grad():
            for s in range(steps + 1):
                out = torch.softmax(model(img_copy.unsqueeze(0)), dim=1)
                prob = out[0, target_class].item()
                probabilities.append(prob)
                
                if s < steps:
                    idx_to_mask = flat_indices[s * step_size : (s + 1) * step_size]
                    for idx in idx_to_mask:
                        r, c = divmod(idx, w)
                        img_copy[:, r, c] = 0.0
                        
        auc = np.trapz(probabilities, dx=1.0 / steps)
        return float(auc)
```

---

## 📊 XAI Benchmark Protocol Matrix

| XAI Method | Target Layer / Node | Spatial Grid | Quantitative Metrics | Paradigm Compatibility |
|---|---|---|---|---|
| **LayerCAM** | `layer3` + `layer4` Fusion | 8x8 -> 128x128 | Pointing Game, Deletion AUC, IoU | CNNs (ResNet, MobileNet, EfficientNet) |
| **Grad-CAM++** | `layer4` | 4x4 -> 128x128 | Pointing Game, Deletion AUC, IoU | CNNs |
| **ViT-Grad-CAM** | `blocks[-1].norm1` | 14x14 -> 128x128 | Pointing Game, Deletion AUC, IoU | Vision Transformers (ViT, Swin) |
| **Integrated Gradients** | Input Layer (128x128) | Full 128x128 Pixel Resolution | Pointing Game, Deletion AUC | Model-Agnostic (CNN + ViT) |

---

## 📈 Summary & Action Plan

1. **Fix Spatial Grid Collapse:** Probe `layer3` (8x8) and apply **LayerCAM** to prevent 4x4 blob blurriness on 128x128 crops.
2. **Replace Eyeballing with Science:** Evaluate heatmaps quantitatively using **Pointing Game Accuracy**, **Deletion AUC**, and **Landmark Mask IoU**.
3. **Unify Cross-Paradigm XAI:** Replace ungrounded Attention Rollout with **ViT-Grad-CAM** or **Integrated Gradients** for mathematically valid CNN vs. Transformer comparisons.
