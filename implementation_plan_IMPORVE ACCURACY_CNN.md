# Upgrade CustomCNN & Training Pipeline to hit 90+ Accuracy

Currently, `custom_cnn` is a very lightweight model (just 4 single-layer ResBlocks) and uses standard data augmentation. It is plateauing around 70-75% accuracy. To hit **90%+ accuracy**, we need to increase the model's capacity to learn complex features, and apply stronger regularizations to prevent overfitting.

## User Review Required

> [!WARNING]  
> If you approve this plan, **I will stop the current training task (Task-284)** that is running in the background and restart the 25-epoch fine-tuning process from scratch with the improved architecture.

## Proposed Changes

### 1. Model Architecture Upgrades
We will modify `models/custom_cnn.py` to add more depth and expressive power without making it as heavy as a ResNet50.
- **Double the Depth:** Instead of 1 `ResBlock` per stage, we will chain 2 `ResBlocks` per stage (similar to a mini-ResNet18). This allows the network to learn more complex hierarchical features.
- **Squeeze-and-Excitation (SE) / Attention:** (Optional but effective) We can add a simple channel attention mechanism in the blocks to help the model focus on critical features (like eyes and mouth).
- **Stronger Classifier:** Increase the fully connected layers capacity and adjust dropout.

#### [MODIFY] [models/custom_cnn.py](file:///home/altos/.gemini/antigravity-ide/scratch/drowsiness-detection/models/custom_cnn.py)
- Refactor `CustomCNN` to include `make_layer` functions that stack multiple `ResBlocks`.

### 2. Stronger Data Augmentation
We will modify `data/dataset_loader.py` to prevent the model from overfitting on the training data.
- Add **Random Affine Transformations** (translation, scaling, shearing).
- Add **Random Perspective**.
- This forces the model to learn robust features rather than memorizing the exact positions of faces in the dataset.

#### [MODIFY] [data/dataset_loader.py](file:///home/altos/.gemini/antigravity-ide/scratch/drowsiness-detection/data/dataset_loader.py)
- Update `get_data_transforms` to include `transforms.RandomAffine` and `transforms.RandomPerspective`.

### 3. Training Pipeline Enhancements
We will optimize the hyperparameters in `train.py`.
- **Label Smoothing:** Already at 0.1, we will keep this.
- **Learning Rate Strategy:** We'll ensure the `CosineAnnealingLR` is tuned correctly for 25 epochs.
- **Weight Decay:** Increase slightly to penalize large weights.

## Verification Plan

### Automated Tests
- Run `python train.py --model custom_cnn --epochs 25` to verify the new architecture trains properly and does not crash.
- Monitor the validation accuracy curve to ensure it breaks past the 80-90% barrier.

### Manual Verification
- Review the `training_curves.png` and `confusion_matrix.png` generated at the end to confirm 90%+ performance.
- Verify that inference (`predict.py`) still runs fast (high FPS).
