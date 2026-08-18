# Drowsiness Detection Dataset Overview

> **Status:** The counts below are a historical dataset snapshot, not a guarantee produced by the current default run. Rebuild the dataset and run `audit_leakage.py` before using these numbers in a report. The current repository does not include the raw archives or `processed_dataset/`.

This document provides a detailed breakdown of the datasets utilized in the Driver Drowsiness Detection project. The project utilizes a massive raw dataset pipeline which is unified, preprocessed, and distilled into a highly efficient and balanced training dataset.

## 1. Raw Datasets (The Archives)

The raw data is stored across four main archive directories, totaling approximately **6.5 GB** and containing **174,772 raw image files**. These directories are ignored by Git (`.gitignore`) due to their massive size.

| Directory Name | Size on Disk | Source Type | Description |
| :--- | :--- | :--- | :--- |
| **`archive/`** | 458 MB | Image/Video | Primary Kaggle Drowsiness Detection dataset containing full-face images (alert and drowsy). |
| **`archive(1)/`** | 118 MB | Image | MRL Eye Dataset featuring tightly cropped images of human eyes (open vs. closed) under varying lighting conditions (IR and visible). |
| **`archive(2)/`** | 3.0 GB | Video/Frames | High-resolution frames from the NTHU-DDD (National Tsing Hua University Driver Drowsiness Detection) dataset. Contains simulated driving scenarios. |
| **`archive(3)/`** | 2.9 GB | Image/Frames | Additional frames from the UTA-RLDD (Real-Life Drowsiness Dataset) capturing multi-stage drowsiness progression. |

> [!NOTE] 
> The raw datasets contain a massive imbalance of files, varying resolutions (from 80x80 up to 1920x1080), and differing framing (some are full faces, some are just eyes).

---

## 2. Dataset Preprocessing (`preprocess_mixed_data.py`)

Training deep neural networks on 174,000+ unstandardized, imbalanced, high-resolution images is slow and prone to data leakage and lighting bias. Therefore, we utilize the enhanced `preprocess_mixed_data.py` script.

### Preprocessing Pipeline:
1. **Intelligent Keyword Inference**: The script recursively scans raw archives and infers labels by analyzing folder and file names (e.g., `no_yawn`, `awake`, `open` -> **Alert**; `closed`, `fatigue`, `yawn` -> **Drowsy**).
2. **Group-Aware Subject Isolation (Leakage Prevention)**: Extracts unique Subject / Group IDs (`nthu_001`, `mrl_s0013`, `subject01`, etc.) and performs group-level stratified splitting. 100% of frames/videos from any subject stay strictly in `train` or `val` with **0.00% data leakage**.
3. **Physical 50/50 Class Balancing**: Enforces exact 1:1 physical class balancing in training splits (`60,314 Alert` vs `60,314 Drowsy`) to eliminate class representation bias during neural network optimization.
4. **Cascaded 3-Tier Eye Extractor**:
   - **Tier 1 (MediaPipe Face Mesh)**: Extracts exact left & right eye ROIs with 40% bounding box padding.
   - **Tier 2 (OpenCV Cascade Face Detection)**: Fallback for detecting upper-half eye regions if MediaPipe landmarks fail.
   - **Tier 3 (Anatomical Upper-Third Crop)**: Fallback for video frames where face detection fails on extreme downward head nodding, eliminating full-frame scale artifacts.
5. **CLAHE Illumination Equalization**: Applies Contrast Limited Adaptive Histogram Equalization (`clipLimit=2.0`, `tileGridSize=(8, 8)`) on L-channel (LAB space) to equalize contrast across IR night-vision and daylight RGB driving frames.
6. **Standardization**: All extracted crops are resized to a uniform `128x128` resolution.

---

## 3. The Final `processed_dataset`

After group-aware preprocessing and CLAHE contrast equalization, the raw 6.5 GB dataset is distilled into a hyper-efficient **1.0 GB** folder containing exactly **155,388 standardized 128x128 images**.

This `processed_dataset` is mapped to an **80/20** Training and Validation split with **zero subject overlap** between splits.

### Training Split (80% - 120,628 images) — Strictly 50/50 Balanced
- **`0_alert`**: 60,314 images (50.0%)
- **`1_drowsy`**: 60,314 images (50.0%)

### Validation Split (20% - 34,760 images)
- **`0_alert`**: 17,572 images (50.6%)
- **`1_drowsy`**: 17,188 images (49.4%)

### Per-Dataset Contribution Breakdown

The processed **155,388 image dataset** combines crops extracted across all 4 raw benchmark dataset archives:

| Dataset Name | Source Directory | Generated Crops | Contribution (%) | Key Characteristics & Features |
| :--- | :--- | :--- | :--- | :--- |
| **MRL Eye Dataset** | `archive(1)/` | **80,194 crops** | **51.61%** | High-precision IR & RGB eye crops across 37 unique human subjects under varying lighting. |
| **NTHU-DDD** (National Tsing Hua University) | `archive(2)/` | **59,516 crops** | **38.30%** | Simulated driving scenario video frames capturing glasses, night vision IR, nodding, and yawning. |
| **UTA-RLDD** (Real-Life Drowsiness Dataset) | `archive(3)/` | **8,686 crops** | **5.59%** | Multi-stage drowsiness progression video frames across human participants. |
| **Kaggle Driver Drowsiness Dataset** | `archive/` | **6,992 crops** | **4.50%** | Full-face and eye state images under varied vehicle interior lighting conditions. |
| **TOTAL** | **All Archives** | **155,388 crops** | **100.00%** | **Unified 128x128 CLAHE crop dataset with 0.00% data leakage.** |

### Final Structure Example:
```text
processed_dataset/
├── train/
│   ├── 0_alert/  └── ... (60,314 images - 50.0%)
│   └── 1_drowsy/ └── ... (60,314 images - 50.0%)
└── val/
    ├── 0_alert/  └── ... (17,572 images - 50.6%)
    └── 1_drowsy/ └── ... (17,188 images - 49.4%)
```

> [!TIP]
> By standardizing the dataset to exactly 128x128 CLAHE eye crops, enforcing 50/50 physical class balance, and isolating subject splits, models can train fast while guaranteeing valid, leak-free evaluation metrics on real-world driver drowsiness CUES.

---

## 4. Model Architecture & Generalizability Strategy (ResNet-18 vs. ResNet-50)

To achieve >90% validation accuracy on unseen human driver subjects without overfitting, choosing the right deep learning backbone capacity is critical:

| Feature / Metric | **ResNet-18 (Recommended 🏆)** | **ResNet-50** |
| :--- | :--- | :--- |
| **Trainable Parameters** | **11.18 Million** | **23.51 Million** (2.1x larger) |
| **Overfitting Risk on 128x128 Crops** | **Low / Optimal** | **Moderate** (2048 channel bottleneck width can overfit fine crop details) |
| **GPU Inference Speed** | **>140 FPS** (Ultra Fast) | **~70 FPS** |
| **Real-World Edge Latency** | **< 8 ms** | **~18 ms** |
| **Generalizability to Unseen Drivers** | **Highest** (Prevents subject identity memorization) | Good (Requires stronger weight decay) |

### Key Takeaways:
- **ResNet-18** provides the optimal parameter capacity (~11.18M) to capture fine eyelid state, iris position, and pupil dilation on 128x128 crops without memorizing driver-specific features (skin tone, lighting background).
- **Inference Efficiency**: Runs at double the frame rate (>140 FPS), making it ideal for real-world webcam deployment (`app.py`).

---

## 5. Model Checkpoint Management Strategy

During training, two distinct `.pth` checkpoint files are maintained under `saved_models/` for each model architecture:

| Checkpoint Filename | Update Frequency | Primary Purpose |
| :--- | :--- | :--- |
| **`<model_name>_drowsiness_model.pth`** | End of **Every Epoch** | Saves current epoch, optimizer state, and loss history so training can be **paused and resumed seamlessly**. |
| **`<model_name>_best_model.pth`** | Only on **New Highest Record Val F1-Score** | Guarantees that the **highest-performing model weights** on unseen validation drivers are preserved for real-time deployment (`app.py` & `predict.py`). |
