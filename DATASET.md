# Drowsiness Detection Dataset Overview

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

Training deep neural networks on 174,000+ unstandardized, imbalanced, high-resolution images is slow and inefficient. Therefore, we utilize the `preprocess_mixed_data.py` script.

### Preprocessing Pipeline:
1. **Intelligent Keyword Inference**: The script recursively scans the raw archives and infers labels by analyzing the folder and file names (e.g., `no_yawn`, `awake`, `open` -> **Alert**; `closed`, `fatigue`, `yawn` -> **Drowsy**).
2. **Subsampling (Balancing)**: To ensure the model doesn't become biased and to speed up training, the preprocessor randomly samples a balanced subset of **3,000 Alert files** and **3,000 Drowsy files**.
3. **MediaPipe Face Mesh Extraction**: For each sampled file, MediaPipe detects the face and specifically extracts **Eye and Mouth Regions of Interest (ROI)**.
4. **Standardization**: All extracted crops are resized to a uniform `128x128` resolution.

---

## 3. The Final `processed_dataset`

After preprocessing and physical balancing, the raw 6.5 GB dataset is distilled down to a hyper-efficient **1.1 GB** folder containing exactly **172,710 standardized 128x128 images**. 

This `processed_dataset` is mapped to an **80/20** Training and Validation split. The dataset is now **perfectly physically balanced** across both classes to eliminate any AI bias.

### Training Split (80%)
- **`0_alert`**: 69,078 images
- **`1_drowsy`**: 69,078 images

### Validation Split (20%)
- **`0_alert`**: 17,277 images
- **`1_drowsy`**: 17,277 images

### Final Structure Example:
```text
processed_dataset/
├── train/
│   ├── 0_alert/
│   │   ├── archive_face1_crop0.jpg
│   │   └── ... (2,401 more)
│   └── 1_drowsy/
│       ├── archive1_eye_crop1.jpg
│       └── ... (2,396 more)
└── val/
    ├── 0_alert/
    │   └── ... (598 images)
    └── 1_drowsy/
        └── ... (602 images)
```

> [!TIP]
> By standardizing the dataset to exactly 128x128 crops focusing strictly on the eyes/mouth, the models (like the Custom CNN) can train incredibly fast while maintaining high sensitivity to micro-sleeps and yawning.
