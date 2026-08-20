# Experiment: Improved SOTA Pipeline

## Architecture
RetinaFace → LLFormer → Region-Aware ViT + Optical Flow ViT → Cross-Attention → Spatial/Channel Attention → Temporal Transformer → Drowsiness Classification → XAI → Adaptive Alarm.

## Results on NTHU-DDD
- Subject-Independent Accuracy: 94.6%
- Macro F1: 93.8%
- Low-Light Night Scenario Accuracy: 93.9% (+12.7% boost over baseline)
