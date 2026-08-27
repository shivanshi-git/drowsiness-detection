# Driver Drowsiness Detection: SOTA Transformer & Multi-Modal XAI

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![NTHU-DDD](https://img.shields.io/badge/Benchmark-NTHU--DDD-green.svg)](https://cv.cs.nthu.edu.tw/)
[![XAI](https://img.shields.io/badge/Explainability-GradCAM%20%7C%20IG%20%7C%20SHAP%20%7C%20Temporal-orange.svg)](xai/)

A State-of-the-Art (SOTA) Deep Learning pipeline for driver drowsiness and microsleep detection designed specifically for **low-light, infrared (IR), and challenging in-cabin driving scenarios** across **NTHU-DDD, YawDD, and MRL Eye** datasets.

---

## 🏛️ Pipeline Architecture

```
Camera (Raw Low-Light Video Stream)
  │
  ▼
RetinaFace (Face & Landmark Localization)
  │
  ▼
LLFormer (Low-Light Enhancement Transformer)
  │
  ├───► Region-Aware ViT (Spatial Stream)
  │
  └───► Optical Flow ViT (Motion Stream)
  │
  ▼
Cross-Attention Fusion + Spatial/Channel Attention
  │
  ▼
Temporal Sequence Transformer
  │
  ▼
Drowsiness Classification Head
  │
  ▼
┌──────────────────────────────────────────────────────────────┐
│                  EXPLAINABILITY (XAI) LAYER                  │
├──────────────────────────────────────────────────────────────┤
│  • Grad-CAM & Attention Maps (Spatial eye/mouth saliency)   │
│  • Integrated Gradients (Axiomatic pixel/motion attributions)│
│  • Regional SHAP (Quantified Shapley values per facial RoI)  │
│  • Temporal Explainer (Frame-level confidence timeline)      │
│  • Facial Landmark Explainer (Geometric EAR, MAR, Head Pose) │
└──────────────────────────────────────────────────────────────┘
  │
  ▼
Adaptive Real-Time Alarm & Explainable Alert Engine
```

### 🌙 Why are the raw images so dark? (The Role of LLFormer)
This pipeline is specifically designed for **Low-Light Driver Drowsiness Detection** (e.g., driving at night, in a tunnel, or with poor cabin lighting). Because ambient light is extremely low in these scenarios, the raw camera frames appear very dark or "black". 

To solve this, the pipeline includes the **LLFormer (Low-Light Enhancement Transformer)**, which artificially brightens the frames, enhances contrast, and restores facial details *before* they are processed by the spatial and motion streams.

### 📈 Why is MRL-Eye performance higher than NTHU-DDD?
The final benchmark reports often show >98% accuracy for MRL-Eye, but much lower Macro F1 scores (e.g. ~46-70%) for NTHU-DDD. This massive difference in performance comes down to the fundamental nature of the two datasets:
1. **Task Complexity**: MRL-Eye is a simple **binary classification** task (Open vs. Closed eye) on closely cropped images. NTHU-DDD is a highly complex **5-class behavioral classification** task (Normal, Slow Blinking, Yawning, Nodding, Eye Closure) on full-body/face frames.
2. **Static vs. Temporal**: MRL-Eye relies purely on static spatial features. NTHU-DDD requires tracking temporal motion across a sequence of frames (e.g. tracking how long an eye is closed to classify a "slow blink" vs. a normal blink).
3. **Lighting**: MRL-Eye images are generally clear or neatly captured with IR. NTHU-DDD focuses on extreme low-light/nighttime driving, forcing the model to work incredibly hard just to extract facial features.
4. **Class Imbalance (The F1 Drop)**: In NTHU-DDD, you will see a massive gap between **Accuracy** (~85%) and **Macro F1** (~46-70%). In a real driving video, the driver is in a "Normal" state 90% of the time. The model achieves high accuracy by guessing "Normal", but the low Macro F1 score reveals the difficulty of correctly identifying rare, critical classes (like Nodding).

---

## 📁 Project Structure

```
driver_drowsiness/
│
├── configs/
│   ├── nthu_ddd.yaml
│   ├── mrl_eye.yaml
│   ├── yawdd.yaml
│   ├── cross_dataset.yaml
│   └── alarm.yaml
│
├── data/
│   ├── datasets/
│   │   ├── nthu_ddd.py
│   │   ├── mrl_eye.py
│   │   └── yawdd.py
│   │
│   ├── splits/
│   │   ├── nthu_subject_split.py
│   │   ├── mrl_subject_split.py
│   │   └── yawdd_subject_split.py
│   │
│   ├── optical_flow.py
│   ├── temporal_sampler.py
│   └── transforms.py
│
├── models/
│   ├── backbones/
│   │   ├── resnet50.py
│   │   ├── inceptionv3.py
│   │   ├── vit_baseline.py
│   │   └── swin_baseline.py
│   │
│   ├── retinaface_detector.py
│   ├── llformer.py
│   ├── region_vit.py
│   ├── flow_vit.py
│   ├── cross_attention_fusion.py
│   ├── temporal_transformer.py
│   └── drowsiness_pipeline.py
│
├── training/
│   ├── train.py
│   ├── trainer.py
│   ├── losses.py
│   └── checkpoint.py
│
├── inference/
│   ├── adaptive_alarm.py
│   ├── realtime_stream.py
│   └── visualization.py
│
├── evaluation/
│   ├── metrics.py
│   ├── confusion_matrix.py
│   ├── roc_auc.py
│   ├── benchmark.py
│   ├── cross_dataset.py
│   └── robustness.py
│
├── experiments/
│   ├── baseline_vit/
│   ├── improved_vit/
│   ├── ablation/
│   └── cross_dataset/
│
├── xai/
│   ├── grad_cam.py
│   ├── integrated_gradients.py
│   ├── shap_explainer.py
│   ├── temporal_explainer.py
│   ├── landmark_explainer.py
│   ├── alarm_explainer.py
│   └── master_explainer.py
│
├── tests/
│   ├── test_dataset.py
│   ├── test_models.py
│   ├── test_pipeline.py
│   └── test_alarm.py
│
├── train.py
├── evaluate.py
├── inference.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Run Complete Test Suite
```bash
python -m unittest discover tests
```

### 2. Train Model
```bash
python train.py --config configs/nthu_ddd.yaml
```

### 3. Evaluate & Benchmark
```bash
python evaluate.py --config configs/nthu_ddd.yaml --checkpoint saved_models/nthu_ddd/best_model.pth
```

### 4. Real-Time HUD with Explainability (XAI)
```bash
python inference.py --source 0
```
