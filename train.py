import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.model_factory import get_model_and_config, count_parameters
from data.dataset_loader import create_dataloaders
from utils.metrics import evaluate_model_performance, save_evaluation_matrix
from utils.losses import FocalLoss
from xai.grad_cam import GradCAM
from xai.visualizer import overlay_heatmap, plot_xai_comparison
import numpy as np

def train_model(model_name='custom_cnn', dataset_dir='processed_dataset', epochs=10, batch_size=32, lr=None, device='cuda', loss_type='focal'):
    """
    Main training and validation loop for Driver Drowsiness Detection.
    Applies paradigm-specific optimization protocols (AdamW, weight decay, learning rates) automatically.
    """
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                "[!] CUDA device requested, but PyTorch cannot access NVIDIA GPU!\n"
                "    Reason: NVIDIA kernel driver module is not loaded in the running Linux kernel (6.17.0-1026-nvidia).\n"
                "    To fix: Run 'sudo dkms autoinstall && sudo modprobe nvidia' with OS admin privileges, then reinstall CUDA PyTorch."
            )
        device = torch.device('cuda')
    else:
        device = torch.device(device)
    print(f"[*] Training Model: '{model_name}' on Device: {device}")

    # Create DataLoaders
    train_loader, val_loader, class_names = create_dataloaders(dataset_dir=dataset_dir, batch_size=batch_size)

    # Instantiate Model, Target Layer, and Paradigm-Specific Optimization Config
    model, target_layer, config = get_model_and_config(model_name=model_name, num_classes=len(class_names), pretrained=True)
    model = model.to(device)

    effective_lr = lr if lr is not None else config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 1e-4)

    print(f"[*] Paradigm Protocol: {config.get('opt_type', 'AdamW')} | LR: {effective_lr} | Weight Decay: {weight_decay} | Loss: {loss_type.upper()}")
    print(f"[*] Trainable Parameters: {count_parameters(model):.2f} Million")

    if loss_type.lower() == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=weight_decay)

    if config.get('scheduler') == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val_f1 = 0.0
    checkpoint_dir = os.path.join("saved_models")
    model_results_dir = os.path.join("results", model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(model_results_dir, exist_ok=True)

    start_epoch = 1
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_drowsiness_model.pth")
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None and hasattr(scheduler, 'load_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'history' in checkpoint and checkpoint['history'] is not None:
                history = checkpoint['history']
            else:
                history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_f1 = checkpoint.get('val_f1', 0.0)
            print(f"[✓] Resumed training from checkpoint at Epoch {start_epoch-1} (Best Val F1: {best_val_f1:.4f})")
        except Exception as e:
            print(f"[!] Note: Could not load checkpoint ({e}). Starting fresh training.")
            history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    else:
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Evaluate on validation set
        val_metrics = evaluate_model_performance(model, val_loader, device=device)
        print(f"[Epoch {epoch:02d}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Acc: {val_metrics['accuracy']*100:.2f}% | Val F1: {val_metrics['f1_score']:.4f} | FPS: {val_metrics['fps']:.1f}")

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics['f1_score'])
        else:
            scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_loss'].append(1.0 - val_metrics['accuracy'])

        # Save Latest Checkpoint after every epoch for seamless pause/resume
        latest_checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_drowsiness_model.pth")
        torch.save({
            'epoch': epoch,
            'model_name': model_name,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
            'val_f1': val_metrics['f1_score'],
            'history': history,
            'class_names': class_names
        }, latest_checkpoint_path)

        # Save Best Model Checkpoint
        if val_metrics['f1_score'] >= best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            best_checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_name': model_name,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'class_names': class_names
            }, best_checkpoint_path)
            print(f"  [✓] Saved Best Model Checkpoint (Val F1: {best_val_f1:.4f}) to: {best_checkpoint_path}")

    # Final Evaluation & Evaluation Matrix Artifact Generation
    print(f"\n[*] Computing Final Evaluation Matrix and Generating Artifacts for '{model_name}'...")
    final_val_metrics = evaluate_model_performance(model, val_loader, device=device)
    summary_dict = save_evaluation_matrix(final_val_metrics, history=history, class_names=class_names, output_dir=model_results_dir)

    # Generate 5 XAI Verification Heatmaps per model
    print(f"\n[*] Generating 5 Grad-CAM Heatmaps for '{model_name}'...")
    try:
        grad_cam = GradCAM(model, target_layer)
        val_samples_gen = iter(val_loader)
        sample_inputs, sample_targets = next(val_samples_gen)
        
        num_xai_samples = min(5, sample_inputs.size(0))
        for idx in range(num_xai_samples):
            sample_img_tensor = sample_inputs[idx:idx+1].to(device)
            true_label = class_names[sample_targets[idx].item()]

            heatmap, pred_class_idx, confidence = grad_cam.generate_heatmap(sample_img_tensor)
            pred_label = class_names[pred_class_idx]

            orig_np = sample_inputs[idx].permute(1, 2, 0).numpy()
            orig_np = (orig_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
            orig_np = np.clip(orig_np, 0, 1)
            orig_bgr = (orig_np[:, :, ::-1] * 255).astype(np.uint8)

            blended_bgr, _ = overlay_heatmap(orig_bgr, heatmap)
            blended_rgb = blended_bgr[:, :, ::-1]
            orig_rgb = (orig_np * 255).astype(np.uint8)

            xai_sample_path = os.path.join(model_results_dir, f"xai_sample_{idx+1}.png")
            plot_xai_comparison(
                orig_rgb, heatmap, blended_rgb,
                f"Pred: {pred_label.upper()} (True: {true_label})",
                confidence, model_name=model_name, save_path=xai_sample_path
            )
            print(f"  [✓] XAI Grad-CAM Heatmap {idx+1}/5 Saved to: {xai_sample_path}")

        grad_cam.remove_hooks()
    except Exception as e:
        print(f"[!] Warning: Grad-CAM generation skipped for {model_name}: {e}")

    print(f"[✓] Training & Evaluation Matrix Generation Complete for '{model_name}'!")
    return summary_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Drowsiness Detection Deep Learning Models with XAI.")
    parser.add_argument("--model", type=str, default="resnet18", choices=['custom_cnn', 'vgg16', 'vgg19', 'resnet18', 'resnet50', 'dual_branch_resnet18', 'temporal_resnet18', 'temporal_resnet50', 'mobilenet_v2', 'mobilenet_v3', 'efficientnet_b0', 'vit_tiny'])
    parser.add_argument("--dataset_dir", type=str, default="processed_dataset")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss", type=str, default="focal", choices=['focal', 'cross_entropy'])
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    if os.path.exists(args.dataset_dir):
        train_model(model_name=args.model, dataset_dir=args.dataset_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device, loss_type=args.loss)
    else:
        print(f"[!] Dataset directory '{args.dataset_dir}' not found. Please run 'python data/preprocess_mixed_data.py --raw_dir <your_raw_folder>' first.")
