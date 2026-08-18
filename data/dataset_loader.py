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
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=10),
        transforms.RandomPerspective(distortion_scale=0.25, p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), value='random'),
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

def create_dataloaders(dataset_dir="processed_dataset", batch_size=32, num_workers=4, img_size=None, include_test=False):
    """
    Constructs PyTorch DataLoaders for train and validation splits.
    """
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    test_dir = os.path.join(dataset_dir, "test")

    if img_size is None:
        if "dual" in dataset_dir.lower():
            img_size = (128, 256)
        else:
            img_size = (128, 128)

    train_transform, val_transform = get_data_transforms(img_size=img_size)

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' must contain 'train' and 'val' subfolders.")
    if include_test and not os.path.exists(test_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' must contain a held-out 'test' subfolder.")

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
    if val_dataset.classes != train_dataset.classes:
        raise ValueError("Train and validation class folders do not match")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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

    if include_test:
        test_dataset = datasets.ImageFolder(root=test_dir, transform=val_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        if test_dataset.classes != train_dataset.classes:
            raise ValueError("Train and test class folders do not match")
        print(f"[DataLoader] Test samples: {len(test_dataset)}")
        return train_loader, val_loader, test_loader, train_dataset.classes

    print(f"[DataLoader] Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"[DataLoader] Classes: {train_dataset.classes}")

    return train_loader, val_loader, train_dataset.classes
