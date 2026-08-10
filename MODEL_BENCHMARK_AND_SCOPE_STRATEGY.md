# 🏛️ Model Architecture Scope & Benchmarking Strategy

> **Project:** Driver Drowsiness Detection System with Explainable AI (XAI)  
> **Document Purpose:** Detailed technical strategy to streamline model architecture breadth, eliminate hyperparameter conflation, resolve Vision Transformer dataset scale mismatches, and establish a fair, empirical benchmarking protocol.

---

## 📌 Executive Summary

Attempting to train, tune, and evaluate **15+ model architecture variants across 5 distinct paradigms** (VGG, ResNet, MobileNet, EfficientNet, ViT, Swin, ConvNeXt) introduces severe scope inflation and benchmarking invalidity. 

Different neural network paradigms require fundamentally different optimization regimes (e.g., Vision Transformers require weight decay, warmup schedules, and strong regularization, whereas CNNs train stably with standard Adam/SGD). Applying a single uniform training schedule across all architectures creates an **apples-to-oranges benchmark** where performance reflects tuning attention rather than architectural merit.

This document establishes a **Streamlined 4-Tier Benchmark Suite**, defines **Paradigm-Specific Training Protocols**, and re-frames performance claims into an **Empirical Hypothesis Validation Framework**.

---

## 🚨 Diagnosis of Model Scope Risks

### 1. Scope Inflation & Compute Overhead
* **The Pitfall:** Training 15+ models (VGG16, ResNet18/50, ResNeXt50, DenseNet121, MobileNetV2/V3, ShuffleNetV2, SqueezeNet, EfficientNet-B0/B2, ConvNeXt-Tiny, ViT-Tiny/Base, Swin-Tiny) requires exponential compute and maintenance.
* **The Reality:** Most variants within the same family (e.g., MobileNetV2 vs V3, ResNet18 vs 50) yield incremental insights while diluting tuning quality across all models.

### 2. Hyperparameter Conflation (Apples-to-Oranges Benchmarks)
* **The Pitfall:** Using a single uniform training setup (e.g., Adam with fixed `lr=1e-3` for 25 epochs) across all models.
* **The Reality:** 
  - **CNNs** (ResNet, MobileNet) have strong inductive biases (translation equivariance) and train robustly with Adam/SGD and modest weight decay (`1e-4`).
  - **Transformers** (ViT, Swin) lack spatial inductive biases, requiring AdamW (`weight_decay=0.05`), Cosine Annealing with linear warmup, and specialized data augmentations (Mixup / CutMix).
  - *Result:* ViT will underperform drastically under a standard CNN schedule, producing a misleading benchmark result.

### 3. Model-Dataset Scale Mismatch (ViT-Base 86M Parameters)
* **The Pitfall:** Deploying heavyweight Vision Transformers (ViT-Base with ~86 Million parameters) on narrow 128x128 eye/mouth crop datasets.
* **The Reality:** Vision Transformers are notoriously data-hungry. Without hundreds of thousands of images or fine-grained pretraining, ViT-Base severely overfits small eye crop datasets. 
* **The Solution:** Downscale to lightweight Transformer variants (**ViT-Tiny** with ~5.7M params or **Swin-Tiny** with ~28M params) initialized with ImageNet-1K pretrained weights.

### 4. Unverified Claims Stated as Fact
* **The Pitfall:** Presenting hard numbers ("Accuracy: 98.2%", "FPS: 60") in project documentation *before* training has completed.
* **The Reality:** Literature expectations must be framed explicitly as **Hypotheses to Validate Empirical Run Results**, preventing unverified claims from passing as deliverables.

---

## 🛠️ The 4-Tier Streamlined Benchmark Blueprint

Rather than evaluating 15 redundant models, we select **4 representative model archetypes**, each serving a distinct architectural role in real-world deployment trade-offs:

| Benchmark Tier | Model Architecture | Parameters | Target Role | Architectural Paradigm |
|---|---|---|---|---|
| **Tier 1: Edge Baseline** | **Custom CNN** / **MobileNetV3-Small** | ~1.5M - 2.5M | Ultra-low latency edge deployment (webcam/mobile) | Lightweight Depthwise Separable CNN |
| **Tier 2: Deep Residual** | **ResNet50** | ~23.5M | Industry standard baseline for feature extraction | Deep Residual Network (Skip Connections) |
| **Tier 3: Compound Scaled** | **EfficientNet-B0** | ~4.0M | Optimal parameter-to-accuracy trade-off | Compound Coefficient Scaled CNN |
| **Tier 4: Attention Transformer** | **ViT-Tiny** (or **Swin-Tiny**) | ~5.7M - 28M | Global self-attention modeling | Vision Transformer (Patch Embeddings) |

---

## ⚙️ Paradigm-Specific Training Protocols

To ensure a fair, rigorous benchmark, each architecture family receives an optimized training configuration:

### 1. CNN Protocol (Custom CNN, ResNet50, MobileNetV3, EfficientNet-B0)
```python
CNN_CONFIG = {
    "optimizer": "AdamW",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "scheduler": "ReduceLROnPlateau",
    "scheduler_params": {"mode": "min", "factor": 0.5, "patience": 3},
    "augmentation": ["RandomHorizontalFlip", "RandomRotation(10)", "ColorJitter(0.2, 0.2)"],
    "warmup_epochs": 0
}
```

### 2. Transformer Protocol (ViT-Tiny, Swin-Tiny)
```python
TRANSFORMER_CONFIG = {
    "optimizer": "AdamW",
    "lr": 3e-4,                      # Lower learning rate for self-attention
    "weight_decay": 0.05,            # Higher weight decay to prevent overfitting
    "scheduler": "CosineAnnealingLR",
    "scheduler_params": {"T_max": 25, "eta_min": 1e-6},
    "warmup_epochs": 5,              # 5-epoch linear warmup essential for ViT stability
    "augmentation": ["RandomHorizontalFlip", "RandAugment(n=2, m=9)", "Mixup(alpha=0.2)"],
    "pretrained": True               # Always load ImageNet-1K pretrained weights
}
```

---

## 📊 Empirical Benchmark Hypothesis Table

*Note: All metric columns represent empirical hypotheses to be validated through systematic training runs.*

| Model Archetype | Params (M) | FLOPs (G) | Top-1 Val Acc (%) | F1-Score | Inference Latency (ms) | CPU FPS | GPU FPS | Benchmark Status |
|---|---|---|---|---|---|---|---|---|
| **Custom CNN** | 1.5M | 0.05 | *Hypothesis: ~91%* | *TBD* | *Target: < 5ms* | *Target: > 120* | *Target: > 300* | ⏳ Pending Run |
| **MobileNetV3-Small** | 2.5M | 0.06 | *Hypothesis: ~94%* | *TBD* | *Target: < 8ms* | *Target: > 90* | *Target: > 250* | ⏳ Pending Run |
| **ResNet50** | 23.5M | 1.30 | *Hypothesis: ~96%* | *TBD* | *Target: < 18ms* | *Target: > 35* | *Target: > 140* | ⏳ Pending Run |
| **EfficientNet-B0** | 4.0M | 0.39 | *Hypothesis: ~97%* | *TBD* | *Target: < 12ms* | *Target: > 50* | *Target: > 180* | ⏳ Pending Run |
| **ViT-Tiny** | 5.7M | 1.10 | *Hypothesis: ~95%* | *TBD* | *Target: < 22ms* | *Target: > 25* | *Target: > 110* | ⏳ Pending Run |

---

## 💻 Refactored Paradigm-Aware Model Factory (`models/model_factory.py`)

Here is how `models/model_factory.py` is structured to cleanly decouple models and return paradigm-specific training hyperparameters:

```python
import torch
import torch.nn as nn
from torchvision.models import resnet50, mobilenet_v3_small, efficientnet_b0, ResNet50_Weights, MobileNet_V3_Small_Weights, EfficientNet_B0_Weights

def get_model_and_config(model_name: str, num_classes: int = 2):
    """
    Returns the initialized model architecture along with its paradigm-specific
    training protocol (optimizer type, learning rate, weight decay, scheduler).
    """
    model_name = model_name.lower()
    
    if model_name in ["custom_cnn", "custom"]:
        from models.custom_cnn import CustomCNN
        model = CustomCNN(num_classes=num_classes)
        config = {
            "opt_type": "AdamW", "lr": 1e-3, "weight_decay": 1e-4,
            "scheduler": "plateau", "warmup_epochs": 0
        }
        
    elif model_name == "mobilenet_v3":
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        config = {
            "opt_type": "AdamW", "lr": 1e-3, "weight_decay": 1e-4,
            "scheduler": "plateau", "warmup_epochs": 0
        }
        
    elif model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        config = {
            "opt_type": "AdamW", "lr": 5e-4, "weight_decay": 1e-4,
            "scheduler": "plateau", "warmup_epochs": 0
        }
        
    elif model_name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        config = {
            "opt_type": "AdamW", "lr": 1e-3, "weight_decay": 1e-4,
            "scheduler": "cosine", "warmup_epochs": 2
        }
        
    elif model_name in ["vit_tiny", "vit"]:
        import timm
        # Using timm ViT-Tiny pretrained on ImageNet-1k (5.7M parameters)
        model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_classes)
        config = {
            "opt_type": "AdamW", "lr": 3e-4, "weight_decay": 0.05,
            "scheduler": "cosine", "warmup_epochs": 5
        }
    else:
        raise ValueError(f"Unknown model architecture: '{model_name}'")
        
    return model, config
```

---

## 📈 Summary & Action Plan

1. **Streamline Model Scope:** Trim from 15 redundant models down to **4 core archetypes** (Custom/MobileNetV3, ResNet50, EfficientNet-B0, ViT-Tiny).
2. **Eliminate Apples-to-Oranges Benchmarks:** Use paradigm-specific hyperparameters (AdamW + Cosine Warmup for Transformers; AdamW + Plateau for CNNs).
3. **Prevent ViT Scale Mismatch:** Replace heavy ViT-Base (86M params) with lightweight **ViT-Tiny** (5.7M params) pretrained on ImageNet.
4. **Hypothesis-Driven Reporting:** Frame performance benchmarks as empirical hypotheses to be validated through systematic training runs.
