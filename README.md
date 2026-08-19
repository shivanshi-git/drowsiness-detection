# Low-Light Driver Drowsiness Detection: SOTA Transformer + XAI Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![NTHU-DDD](https://img.shields.io/badge/Benchmark-NTHU--DDD-green.svg)](https://cv.cs.nthu.edu.tw/)
[![XAI](https://img.shields.io/badge/Explainability-GradCAM%20%7C%20IG%20%7C%20SHAP%20%7C%20Temporal-orange.svg)](xai/)

A State-of-the-Art (SOTA) Deep Learning pipeline for driver drowsiness and microsleep detection designed specifically for **low-light, infrared (IR), and challenging in-cabin nighttime driving scenarios** on the **NTHU Driver Drowsiness Detection (NTHU-DDD)** dataset, with a dedicated **Multi-Modal Explainability (XAI)** suite.

---

## 🏛️ End-to-End Pipeline Architecture

```
Camera (Raw Low-Light / IR Video Stream)
  │
  ▼
RetinaFace (Face & 5-point Landmark Localization)
  │  ├── Face Bounding Box & RoI Decomposition
  │  └── RoIs: Face (224x224), Left Eye (64x64), Right Eye (64x64), Mouth (64x64)
  │
  ▼
LLFormer (Low-Light Enhancement Transformer)
  │  ├── Multi-Dconv Head Transposed Attention (MDTA)
  │  └── Gated-Dconv Feed-Forward Network (GDFN)
  │
  ├───► Region-Aware ViT (Spatial Stream)
  │         ├── Patch Embedding + Region Type Embeddings
  │         └── Multi-layer Spatial Self-Attention
  │
  └───► Optical Flow ViT (Motion Stream)
            ├── Dense Motion Flow (dx, dy)
            └── Dynamic Velocity Tokenization
  │
  ▼
Cross-Attention Fusion + Spatial/Channel Attention
  │  └── Fuses Spatial Semantics with Motion Velocity Fields
  │
  ▼
Temporal Sequence Transformer
  │  └── Models temporal dynamics across sliding sequence windows (16-32 frames)
  │
  ▼
Drowsiness Classification Head
  │  └── 5 States: Normal, Slow Blinking, Yawning, Nodding, Eye Closure
  │
  ▼
┌──────────────────────────────────────────────────────────────┐
│                  EXPLAINABILITY (XAI) LAYER                  │
├──────────────────────────────────────────────────────────────┤
│  • Grad-CAM & ViT Attention Maps (Spatial eye/mouth saliency)│
│  • Integrated Gradients (Axiomatic pixel/motion attributions)│
│  • Regional SHAP (Quantified Shapley values per facial RoI)  │
│  • Temporal Explainer (Frame-level attention trigger curve)  │
│  • Facial Landmark Explainer (Geometric EAR, MAR, Head Pose) │
└──────────────────────────────────────────────────────────────┘
  │
  ▼
Adaptive Real-Time Alarm Engine
     ├── Tier 0: Attentive (Green HUD)
     ├── Tier 1: Visual Notice (Yellow HUD) - P(drowsy) >= 0.45
     ├── Tier 2: Caution Chime (Orange HUD) - P(drowsy) >= 0.65 or PERCLOS > 15%
     └── Tier 3: Critical Siren (Red Flashing HUD) - P(drowsy) >= 0.85 or Closure > 1.5s
```

---

## 📁 Repository Structure

```
.
├── configs/
│   └── nthu_ddd_config.yaml         # Dataset, model, training & alarm config
├── data/
│   ├── nthu_dataset.py              # NTHU-DDD temporal window dataset loader
│   ├── optical_flow.py              # Dense Farneback optical flow extractor
│   └── transforms.py                # Low-light photometric transforms & augmentations
├── models/
│   ├── retinaface_detector.py       # RetinaFace & Multi-RoI facial extractor
│   ├── llformer.py                  # LLFormer low-light enhancement transformer
│   ├── region_vit.py                # Region-Aware ViT for Eye/Mouth/Face RoIs
│   ├── flow_vit.py                  # Optical Flow ViT for motion dynamics
│   ├── cross_attention_fusion.py    # Cross-attention & Spatial/Channel attention
│   ├── temporal_transformer.py      # Temporal sequence transformer (16-32 frames)
│   └── drowsiness_pipeline.py       # End-to-end orchestrated pipeline network
├── xai/
│   ├── grad_cam.py                  # Grad-CAM & ViT Attention Rollout Maps
│   ├── integrated_gradients.py      # Integrated Gradients multi-modal attribution
│   ├── shap_explainer.py            # Regional SHAP feature contribution analysis
│   ├── temporal_explainer.py        # Temporal sequence attention weight profiling
│   ├── landmark_explainer.py        # Geometric EAR, MAR, and Head Pose explanations
│   └── master_explainer.py          # Unified XAI engine and multi-panel dashboard
├── inference/
│   ├── adaptive_alarm.py            # Multi-tier dynamic alerting & PERCLOS engine
│   └── realtime_stream.py           # Live video / webcam HUD runner with XAI view
├── evaluation/
│   ├── metrics.py                   # Multi-class metrics, PERCLOS, ROC-AUC, latency profiler
│   └── nthu_benchmark.py            # NTHU-DDD subject-independent benchmark script
├── tests/
│   └── test_pipeline.py             # Full unit test suite with XAI validation
├── train_nthu.py                    # PyTorch AMP mixed-precision training script
├── requirements.txt                 # Pinned dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Run Unit & XAI Test Suite
```bash
python tests/test_pipeline.py
```

### 2. Train Model on NTHU-DDD
```bash
python train_nthu.py --config configs/nthu_ddd_config.yaml
```

### 3. Run Real-Time Stream with XAI Dashboard
```bash
# Toggle XAI dashboard view with 'x', quit with 'q'
python inference/realtime_stream.py --source 0
```
