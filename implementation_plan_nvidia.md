# Implementation Plan - Multi-Model Drowsiness Detection Pipeline

Add dataset archive folders to `.gitignore`, build a unified multi-dataset preprocessor for `archive`, `archive(1)`, `archive(2)`, and `archive(3)`, train each deep learning model sequentially, and save evaluation matrices, loss/accuracy curves, ROC plots, Grad-CAM heatmaps, and metric reports per model.

## User Review Required

> [!IMPORTANT]
> - **Dataset Ingestion**: The dataset archives (`archive`, `archive(1)`, `archive(2)`, `archive(3)`) contain over 174,000 images across different label naming schemes (e.g. `awake`/`sleepy`, `Open_Eyes`/`Closed_Eyes`, `notdrowsy`/`drowsy`, `active`/`fatigue`). We will unify all 4 source archives into `processed_dataset/` (`train/0_alert`, `train/1_drowsy`, `val/0_alert`, `val/1_drowsy`).
> - **Execution Hardware**: Hardware will execute on CPU/GPU. We will calibrate training parameters (epochs, batch size, and sample size) to ensure complete, thorough model convergence while completing training for all models efficiently.
> - **Isolated Results Output**: Each model's artifacts (Confusion Matrix, ROC plot, Loss/Accuracy curves, Grad-CAM heatmap, JSON/TXT reports) will be saved in `results/{model_name}/` to prevent overwriting.

## Proposed Changes

---

### Environment & Repository Hygiene

#### [MODIFY] [.gitignore](file:///home/altos/.gemini/antigravity-ide/scratch/drowsiness-detection/.gitignore)
- Add explicit ignore patterns for dataset archives:
  - `archive/`
  - `archive(1)/`
  - `archive(2)/`
  - `archive(3)/`
  - `archive*/`
- Verify git status so no dataset archive files are tracked or pushed to GitHub.

---

### Data Preprocessing & Unification

#### [MODIFY] [preprocess_mixed_data.py](file:///home/altos/.gemini/antigravity-ide/scratch/drowsiness-detection/data/preprocess_mixed_data.py)
- Update data preprocessor to handle multi-archive ingestion:
  - Support automatic label mapping:
    - Alert (`0_alert`): `awake`, `Open_Eyes`, `No_yawn`, `notdrowsy`, `active`
    - Drowsy (`1_drowsy`): `sleepy`, `Closed_Eyes`, `Yawn`, `drowsy`, `fatigue`
  - Process images & video clips from all 4 archive folders (`archive`, `archive(1)`, `archive(2)`, `archive(3)`).
  - Crop and normalize faces/eyes to target size (128x128).
  - Split cleanly into `processed_dataset/train` and `processed_dataset/val`.

---

### Training & Evaluation Engine

#### [MODIFY] [utils/metrics.py](file:///home/altos/.gemini/antigravity-ide/scratch/drowsiness-detection/utils/metrics.py)
- Enhance `save_evaluation_matrix()` to accept `model_name` and custom output paths.
- Ensure output files are saved cleanly per model:
  - `{model_name}_confusion_matrix.png`
  - `{model_name}_roc_curve.png`
  - `{model_name}_training_curves.png`
  - `{model_name}_evaluation_summary.json`
  - `{model_name}_evaluation_report.txt`

#### [MODIFY] [train.py](file:///home/altos/.gemini/antigravity-ide/scratch/drowsiness-detection/train.py)
- Update `train_model()` to save artifacts under `results/{model_name}/`.
- Ensure Grad-CAM XAI visualizer saves `{model_name}_xai_heatmap.png`.
- Save model checkpoints to `saved_models/{model_name}_drowsiness_model.pth`.

#### [NEW] [train_all_models.py](file:///home/altos/.gemini/antigravity-ide/scratch/drowsiness-detection/train_all_models.py)
- Batch script to sequentially train all supported model architectures:
  1. `custom_cnn`
  2. `vgg16`
  3. `vgg19`
  4. `resnet18`
  5. `resnet50`
  6. `mobilenet_v2`
  7. `mobilenet_v3`
  8. `efficientnet_b0`
  9. `vit_tiny`
- Generate a overall comparative evaluation report & leaderboard chart (`model_comparison_leaderboard.png` and `all_models_summary.json`).

---

## Verification Plan

### Automated Tests & Pipeline Verification
1. **Gitignore Verification**:
   - Run `git status` to verify `archive`, `archive(1)`, `archive(2)`, `archive(3)` are completely ignored by Git.
2. **Environment & Dependency Setup**:
   - Setup `.venv` virtual environment and install `requirements.txt`.
   - Test PyTorch, OpenCV, MediaPipe, scikit-learn, matplotlib imports.
3. **Dataset Preprocessing Test**:
   - Run unified preprocessor script on `archive`, `archive(1)`, `archive(2)`, `archive(3)`.
   - Verify non-empty `processed_dataset/train/0_alert`, `processed_dataset/train/1_drowsy`, `processed_dataset/val/0_alert`, `processed_dataset/val/1_drowsy`.
4. **Multi-Model Sequential Training & Evaluation**:
   - Run `train_all_models.py`.
   - Verify model checkpoints in `saved_models/`:
     - `custom_cnn_drowsiness_model.pth`
     - `vgg16_drowsiness_model.pth`
     - `vgg19_drowsiness_model.pth`
     - `resnet18_drowsiness_model.pth`
     - `resnet50_drowsiness_model.pth`
     - `mobilenet_v2_drowsiness_model.pth`
     - `mobilenet_v3_drowsiness_model.pth`
     - `efficientnet_b0_drowsiness_model.pth`
     - `vit_tiny_drowsiness_model.pth`
   - Verify per-model results in `results/{model_name}/`:
     - Evaluation Matrix (Confusion Matrix plot)
     - Training & Validation Loss/Accuracy curves
     - ROC Curve
     - Grad-CAM heatmap visualization
     - JSON & TXT evaluation report
   - Verify overall multi-model leaderboard summary.
