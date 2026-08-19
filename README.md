# Low-Light Driver Drowsiness Detection: SOTA Transformer Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![NTHU-DDD](https://img.shields.io/badge/Benchmark-NTHU--DDD-green.svg)](https://cv.cs.nthu.edu.tw/)

A State-of-the-Art (SOTA) Deep Learning pipeline for driver drowsiness and microsleep detection designed specifically for **low-light, infrared (IR), and challenging in-cabin nighttime driving scenarios** evaluated on the **NTHU Driver Drowsiness Detection (NTHU-DDD)** dataset.

---

## 🏛️ Pipeline Architecture

```
Raw Low-Light / IR Video Stream
  │
  ▼
[ 1. RetinaFace Face & Landmark Localization ]
  │  ├── Face Bounding Box & 5-point Keypoints
  │  └── Dynamic RoI Extractor (Face, Left Eye, Right Eye, Mouth)
  │
  ▼
[ 2. LLFormer (Low-Light Enhancement Transformer) ]
  │  ├── Multi-Dconv Head Transposed Attention (MDTA)
  │  └── Gated-Dconv Feed-Forward Network (GDFN)
  │
  ├───► [ 3A. Region-Aware ViT (Spatial Stream) ]
  │         ├── Patch Embedding + Region Type Embeddings (Face/Eye/Mouth)
  │         └── Multi-layer Self-Attention
  │
  └───► [ 3B. Optical Flow ViT (Motion Stream) ]
            ├── Dense Velocity Flow (dx, dy)
            └── Dynamic Motion Tokenization
  │
  ▼
[ 4. Bidirectional Cross-Attention Fusion ]
  │  └── Fuses Spatial Semantics with Motion Velocity Fields
  │
  ▼
[ 5. Spatial & Channel Attention Module (CBAM-like) ]
  │  └── Filters cabin noise and highlights fatigue-critical regions
  │
  ▼
[ 6. Temporal Sequence Transformer ]
  │  └── Models sequence-level dynamics across sliding temporal windows (16-32 frames)
  │
  ▼
[ 7. Multi-Class Fatigue Classification Head ]
  │  └── 5 States: Normal, Slow Blinking, Yawning, Nodding, Eye Closure
  │
  ▼
[ 8. Adaptive Real-Time Alarm Engine ]
     ├── Tier 0: Attentive (Green)
     ├── Tier 1: Visual Notice (Yellow) - $P(\text{drowsy}) \ge 0.45$
     ├── Tier 2: Caution Chime (Orange) - $P(\text{drowsy}) \ge 0.65$ or PERCLOS $> 15\%$
     └── Tier 3: Critical Siren (Red) - $P(\text{drowsy}) \ge 0.85$ or Closure $> 1.5\text{s}$
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/your-org/drowsiness-detection.git
cd "drowsiness detection"
pip install -r requirements.txt
```

### 2. Run Test Suite

Verify all pipeline modules and forward passes:

```bash
python tests/test_pipeline.py
```

### 3. Training on NTHU-DDD Dataset

Configure parameters in [`configs/nthu_ddd_config.yaml`](configs/nthu_ddd_config.yaml) and start training:

```bash
python train_nthu.py --config configs/nthu_ddd_config.yaml
```

### 4. Subject-Independent Benchmark Evaluation

```bash
python evaluation/nthu_benchmark.py --data_dir data/nthu_ddd_raw --checkpoint saved_models/low_light_sota/best_sota_model.pth
```

### 5. Real-Time Video / Webcam Inference with HUD

```bash
# Webcam
python inference/realtime_stream.py --source 0

# Video file
python inference/realtime_stream.py --source path/to/night_drive.mp4 --checkpoint saved_models/low_light_sota/best_sota_model.pth
```

---

## 📁 Repository Structure

```
├── configs/
│   └── nthu_ddd_config.yaml       # Hyperparameters, paths, sequence length, alarm thresholds
├── data/
│   ├── nthu_dataset.py            # NTHU-DDD dataset loader, windowing, and synthetic fallback
│   ├── optical_flow.py            # Dense Farneback optical flow & motion velocity extractor
│   └── transforms.py              # Low-light photometric transforms & data augmentation
├── models/
│   ├── retinaface_detector.py     # RetinaFace wrapper and multi-RoI facial extractor
│   ├── llformer.py                # LLFormer low-light enhancement transformer
│   ├── region_vit.py              # Region-Aware ViT for Face/Eye/Mouth tokenization
│   ├── flow_vit.py                # Optical Flow ViT for motion dynamic tokenization
│   ├── cross_attention_fusion.py  # Cross-attention & Spatial/Channel attention fusion
│   ├── temporal_transformer.py    # Temporal sequence modeling across video frames
│   └── drowsiness_pipeline.py     # End-to-end orchestrated low-light detection pipeline
├── evaluation/
│   ├── metrics.py                 # Multi-class accuracy, F1, PERCLOS error & latency profiler
│   └── nthu_benchmark.py          # NTHU-DDD subject-independent benchmarking script
├── inference/
│   ├── adaptive_alarm.py          # Multi-tier dynamic alerting & fatigue engine
│   └── realtime_stream.py         # Real-time video stream runner with HUD overlay
├── tests/
│   └── test_pipeline.py           # Unit and integration test suite
├── train_nthu.py                  # PyTorch mixed-precision training script
├── requirements.txt               # Pinned dependencies
└── README.md
```
