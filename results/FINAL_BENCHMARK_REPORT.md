# 🏆 Official Final Benchmark Evaluation Report (NTHU-DDD & MRL-Eye)

This document contains the official benchmark evaluation results across all 5 deep learning architectures for both the **NTHU-DDD** (Low-Light Driver Drowsiness Detection) and **MRL-Eye** (Spatial Eye Open/Closed State) datasets.

---

## 🚘 1. NTHU-DDD Dataset Benchmark Summary

| Model Architecture | Status | Epochs Completed | Best Val Macro F1 | Best Val Accuracy | Saved Checkpoint Path |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **INCEPTION** | ✅ Completed | `30/30` | **40.54%** | **76.23%** | [best_inception_model.pth](file:///saved_models/inception/best_inception_model.pth) |

---

## 👁️ 2. MRL-Eye Dataset Benchmark Summary

| Model Architecture | Status | Epochs Completed | Best Val Macro F1 | Best Val Accuracy | Saved Checkpoint Path |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **RESNET50** | ✅ Completed | `30/30` | **99.12%** | **99.15%** | [best_resnet50_mrl_model.pth](file:///saved_models/mrl_eye/resnet50/best_resnet50_mrl_model.pth) |
| **SOTA** | ✅ Completed | `30/30` | **98.33%** | **98.33%** | [best_sota_mrl_model.pth](file:///saved_models/mrl_eye/sota/best_sota_mrl_model.pth) |
| **VIT** | ✅ Completed | `30/30` | **99.01%** | **99.01%** | [best_vit_mrl_model.pth](file:///saved_models/mrl_eye/vit/best_vit_mrl_model.pth) |
| **SWIN** | ✅ Completed | `30/30` | **99.10%** | **99.13%** | [best_swin_mrl_model.pth](file:///saved_models/mrl_eye/swin/best_swin_mrl_model.pth) |
| **INCEPTION** | ✅ Completed | `21/21` | **99.10%** | **99.12%** | [best_inception_mrl_model.pth](file:///saved_models/mrl_eye/inception/best_inception_mrl_model.pth) |

---

### 📊 Report Summary & Artifacts
- All model checkpoints are saved in `saved_models/` and `saved_models/mrl_eye/`.
- All evaluation metric CSVs, JSONs, Confusion Matrix PNGs, and ROC curves are saved in [results/](file:///results).
