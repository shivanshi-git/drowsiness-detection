import os
import torch
import numpy as np
import cv2
from PIL import Image

from models.model_factory import get_model
from xai.grad_cam import GradCAM
from xai.visualizer import overlay_heatmap, plot_xai_comparison
from data.dataset_loader import create_dataloaders

def generate_5_xai_heatmaps(model_name='custom_cnn', checkpoint_path=None, output_dir=None, num_samples=5, device='cuda'):
    """
    Generates at least 5 distinct Grad-CAM heatmaps for validation samples and saves them to results/<model_name>/.
    """
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model, target_layer = get_model(model_name=model_name, num_classes=2, pretrained=False)

    if not checkpoint_path:
        checkpoint_path = f"saved_models/{model_name}_drowsiness_model.pth"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    class_names = checkpoint.get('class_names', ['0_alert', '1_drowsy'])
    model = model.to(device)
    model.eval()

    if not output_dir:
        output_dir = os.path.join("results", model_name)
    os.makedirs(output_dir, exist_ok=True)

    _, val_loader, _ = create_dataloaders(dataset_dir="processed_dataset", batch_size=32)

    grad_cam = GradCAM(model, target_layer)
    saved_count = 0

    print(f"[*] Generating {num_samples} Grad-CAM Heatmaps for '{model_name}' on Device: {device}...")

    for sample_inputs, sample_targets in val_loader:
        for idx in range(sample_inputs.size(0)):
            if saved_count >= num_samples:
                break

            img_tensor = sample_inputs[idx:idx+1].to(device)
            true_label = class_names[sample_targets[idx].item()]

            heatmap, pred_class_idx, confidence = grad_cam.generate_heatmap(img_tensor)
            pred_label = class_names[pred_class_idx]

            orig_np = sample_inputs[idx].permute(1, 2, 0).numpy()
            orig_np = (orig_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
            orig_np = np.clip(orig_np, 0, 1)
            orig_bgr = (orig_np[:, :, ::-1] * 255).astype(np.uint8)

            blended_bgr, _ = overlay_heatmap(orig_bgr, heatmap)
            blended_rgb = blended_bgr[:, :, ::-1]
            orig_rgb = (orig_np * 255).astype(np.uint8)

            sample_save_path = os.path.join(output_dir, f"xai_sample_{saved_count+1}.png")
            plot_xai_comparison(
                orig_rgb, heatmap, blended_rgb,
                f"Pred: {pred_label.upper()} (True: {true_label})",
                confidence, model_name=model_name, save_path=sample_save_path
            )
            saved_count += 1
            print(f"  [✓] XAI Heatmap {saved_count}/{num_samples} Saved -> {sample_save_path}")

    grad_cam.remove_hooks()
    print(f"\n[✓] Successfully generated {saved_count} Grad-CAM XAI Heatmaps in '{output_dir}/'!\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate 5 Grad-CAM heatmaps for a trained model checkpoint.")
    parser.add_argument("--model", type=str, default="custom_cnn")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    generate_5_xai_heatmaps(model_name=args.model, checkpoint_path=args.checkpoint, num_samples=args.num_samples, device=args.device)
