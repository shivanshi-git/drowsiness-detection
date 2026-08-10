import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_data_transforms(img_size=(128, 128)):
    """
    Data Augmentations for Drowsiness Detection.
    - Train: Random horizontal flips, small rotations, color jitter, normalization.
    - Val/Test: Resize and normalization only.
    """
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def create_dataloaders(dataset_dir="processed_dataset", batch_size=32, num_workers=4, img_size=(128, 128)):
    """
    Constructs PyTorch DataLoaders for train and validation splits.
    """
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")

    train_transform, val_transform = get_data_transforms(img_size=img_size)

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' must contain 'train' and 'val' subfolders.")

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)

    import numpy as np
    from torch.utils.data import WeightedRandomSampler

    # Calculate class weights to handle imbalanced datasets (e.g. 120k vs 86k)
    targets = train_dataset.targets
    class_counts = np.bincount(targets)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"[DataLoader] Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"[DataLoader] Classes: {train_dataset.classes}")

    return train_loader, val_loader, train_dataset.classes
