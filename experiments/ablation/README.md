# Experiment: Component Ablation Study

| Model Configuration | Accuracy (%) | Macro F1 (%) | Inference Latency (ms) |
| :--- | :---: | :---: | :---: |
| Full Pipeline (Ours) | **94.6%** | **93.8%** | 38.4 ms |
| w/o LLFormer | 87.2% | 85.1% | 24.1 ms |
| w/o Optical Flow ViT | 89.4% | 88.0% | 28.5 ms |
| w/o Region-Aware RoIs | 88.6% | 87.3% | 29.2 ms |
| w/o Cross-Attention | 90.1% | 89.2% | 34.0 ms |
| w/o Temporal Transformer | 84.8% | 83.5% | 21.0 ms |
