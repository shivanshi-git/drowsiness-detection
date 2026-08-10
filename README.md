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

### 2. Preprocess Raw Datasets (Mixed Images & Videos)
Place your downloaded MRL, Kaggle, NTHU, or UTA-RLDD folders inside a `raw_data` folder, then run:
```bash
python data/preprocess_mixed_data.py --raw_dir raw_data --out_dir processed_dataset
```

### 3. Train Model Architecture
Train any model (e.g. `vgg16`, `mobilenet_v2`, `resnet18`, `custom_cnn`):
```bash
python train.py --model vgg16 --dataset_dir processed_dataset --epochs 10 --batch_size 32
```

### 4. Single Image Prediction & Grad-CAM Heatmap
```bash
python predict.py --image test_driver.jpg --model vgg16 --out output_xai.jpg
```

### 5. Launch Streamlit Interactive Web Application
```bash
streamlit run app.py
```
---
