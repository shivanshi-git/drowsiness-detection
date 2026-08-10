# Driver Drowsiness Detection System with Explainable AI (XAI)

A comprehensive Deep Learning framework for real-time **Driver Drowsiness Detection**, featuring **12 candidate model architectures** (VGG16, ResNet, MobileNet, EfficientNet, ViT, Custom CNN) integrated with **Grad-CAM Explainable AI** and an interactive **Streamlit Dashboard**.

---

## 🌟 Key Features

1. **Mixed Data Preprocessor (`data/preprocess_mixed_data.py`)**:
   - Handles mixed datasets of **Images (`.jpg`, `.png`)** and **Videos (`.mp4`, `.avi`, `.mov`)**.
   - Uses MediaPipe Face Mesh to extract eye and mouth ROIs, normalizing crops to $128 \times 128$.
   - Fuses MRL Eye Dataset, Kaggle Drowsiness, NTHU-DDD, and UTA-RLDD into a unified balanced dataset.

2. **Modular Model Suite (`models/model_factory.py`)**:
   - Includes **VGG16/19**, **ResNet18/50**, **MobileNetV2/V3**, **EfficientNet-B0/B2**, **Custom CNN**, and **ViT-Tiny**.

3. **Explainable AI Engine (`xai/grad_cam.py`)**:
   - Computes gradient activation heatmaps to visually explain *why* the model predicted a driver as drowsy (highlighting eye closure, eyelids, or yawning).

4. **Hybrid Geometric Metric (`utils/face_mesh.py`)**:
   - Real-time **Eye Aspect Ratio (EAR)** and **Mouth Aspect Ratio (MAR)** tracking.

5. **Interactive Web Dashboard (`app.py`)**:
   - Streamlit interface for image diagnostic upload, live camera stream monitoring, and model benchmark comparisons.

---

## ⚡ Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Custom Dataset (Ignored by Git)
Place raw images/videos in `raw_data/` and preprocess them, or directly structure your processed dataset in `processed_dataset/`:
```text
processed_dataset/
├── train/
│   ├── alert/
│   └── drowsy/
└── val/
    ├── alert/
    └── drowsy/
```
*(Note: `raw_data/` and `processed_dataset/` are listed in `.gitignore` to ensure dataset files remain local and are NOT pushed to GitHub).*

To preprocess mixed images/videos from `raw_data/`:
```bash
python data/preprocess_mixed_data.py --raw_dir raw_data --out_dir processed_dataset
```

### 3. Train Model & Generate Evaluation Matrix
Train any model (e.g. `vgg16`, `mobilenet_v2`, `resnet18`, `custom_cnn`):
```bash
python train.py --model vgg16 --dataset_dir processed_dataset --epochs 10 --batch_size 32
```
This automatically:
- Saves the best trained PyTorch model checkpoint to `saved_models/vgg16_drowsiness_model.pth`.
- Computes and exports the complete **Evaluation Matrix** to `results/`:
  - `confusion_matrix.png` (Confusion Matrix Heatmap)
  - `roc_curve.png` (ROC Curve & AUC Score)
  - `training_curves.png` (Loss & Accuracy Epoch History)
  - `evaluation_summary.json` & `evaluation_report.txt` (Detailed Precision, Recall, F1, Accuracy, Latency & FPS)
  - `xai_verification_sample.png` (Grad-CAM Heatmap verification sample)

### 4. Commit and Push Trained Model & Evaluation Matrix to Repository
```bash
git add saved_models/ results/ train.py utils/metrics.py .gitignore README.md
git commit -m "Add trained drowsiness model, evaluation matrix, and pipeline updates"
git push origin main
```

### 5. Single Image Prediction & Grad-CAM Heatmap
```bash
python predict.py --image test_driver.jpg --model vgg16 --out output_xai.jpg
```

### 6. Launch Streamlit Interactive Web Dashboard
```bash
streamlit run app.py
```
---
