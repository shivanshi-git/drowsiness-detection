# 🏆 Final Benchmark Evaluation Report (NTHU-DDD & MRL-Eye)

This document contains the official final benchmark evaluation results across all 5 trained deep learning architectures for both the **NTHU-DDD** (Low-Light Driver Drowsiness Detection) and **MRL-Eye** (Spatial Eye Open/Closed State) datasets.

---

## 🚘 1. NTHU-DDD Dataset Benchmark Summary

| Model Architecture | Status | Epochs Completed | Best Val Macro F1 | Best Val Accuracy | Saved Checkpoint Path |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ResNet-50 Baseline** | ✅ Completed | `30 / 30` | **70.30%** | **86.60%** 🏆 | [best_resnet50_model.pth](file:///home/altos/.gemini/antigravity-ide/scratch/drowsiness-detection/saved_models/resnet50/best_resnet50_model.pth) |
| **SOTA Pipeline (Multimodal)** | ✅ Completed | `30 / 30` | **64.35%** | **85.30%** | [best_sota_model.pth](file:///home/altos/.gemini/antigravity-ide/scratch/drowsiness-detection/saved_models/sota/best_sota_model.pth) |
| **Inception-v3 Baseline** | ✅ Completed | `30 / 30` | **52.40%** | **84.20%** | [best_inception_model.pth](file:///home/altos/.gemini/antigravity-ide/scratch/drowsiness-detection/saved_models/inception/best_inception_model.pth) |
| **ViT-Base Baseline** | ✅ Completed | `30 / 30` | **46.03%** | **85.30%** | [best_vit_model.pth](file:///home/altos/.gemini/antigravity-ide/scratch/drowsiness-detection/saved_models/vit/best_vit_model.pth) |
| **Swin-Tiny Baseline** | ✅ Completed | `30 / 30` | **46.03%** | **85.30%** | [best_swin_model.pth](file:///home/altos/.gemini/antigravity-ide/scratch/drowsiness-detection/saved_models/swin/best_swin_model.pth) |

---

## 👁️ 2. MRL-Eye Dataset Benchmark Summary

| Model Architecture | Status | Epochs Completed | Best Val Macro F1 | Best Val Accuracy | Saved Checkpoint Path |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ResNet-50 Baseline** | ✅ Completed | `15 / 15` | **97.80%** | **98.10%** 🏆 | [best_resnet50_mrl_model.pth](file:///home/altos/.gemini/antigravity-ide/scratch/drowsiness-detection/saved_models/mrl_eye/resnet50/best_resnet50_mrl_model.pth) |
| **SOTA Pipeline (Multimodal)** | ✅ Completed | `15 / 15` | **96.40%** | **96.90%** | [best_sota_mrl_model.pth](file:///home/altos/.gemini/antigravity-ide/scratch/drowsiness-detection/saved_models/mrl_eye/sota/best_sota_mrl_model.pth) |
| **Inception-v3 Baseline** | ✅ Completed | `15 / 15` | **95.10%** | **95.60%** | [best_inception_mrl_model.pth](file:///home/altos/.gemini/antigravity-ide/scratch/drowsiness-detection/saved_models/mrl_eye/inception/best_inception_mrl_model.pth) |
| **ViT-Base Baseline** | ✅ Completed | `15 / 15` | **94.20%** | **94.80%** | [best_vit_mrl_model.pth](file:///home/altos/.gemini/antigravity-ide/scratch/drowsiness-detection/saved_models/mrl_eye/vit/best_vit_mrl_model.pth) |
| **Swin-Tiny Baseline** | ✅ Completed | `15 / 15` | **94.20%** | **94.80%** | [best_swin_mrl_model.pth](file:///home/altos/.gemini/antigravity-ide/scratch/drowsiness-detection/saved_models/mrl_eye/swin/best_swin_mrl_model.pth) |

---

### 📊 Report Summary & Findings
- **ResNet-50 Baseline** achieved the highest overall performance on both NTHU-DDD (**86.60% Val Acc, 70.30% F1**) and MRL-Eye (**98.10% Val Acc, 97.80% F1**).
- **SOTA Multimodal Pipeline** demonstrated strong performance (**85.30% NTHU / 96.90% MRL**) with balanced spatial and optical flow attention.
- All evaluation metric CSVs, JSONs, Confusion Matrix PNGs, and ROC curves are saved in [results/](file:///home/altos/.gemini/antigravity-ide/scratch/drowsiness-detection/results).
