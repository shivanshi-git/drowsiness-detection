import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from models.drowsiness_pipeline import LowLightDrowsinessPipeline
from data.nthu_dataset import build_nthu_dataloaders
from xai.grad_cam import GradCAMExplainer


def generate_xai_plots(
    model_checkpoint: str = "saved_models/low_light_sota/best_sota_model.pth",
    output_dir: str = "results/xai",
    num_samples: int = 4
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[XAI] Using device: {device}")

    # Build Dataloader
    _, val_loader = build_nthu_dataloaders(
        root_dir="archive(2)/NTHU DDD",
        batch_size=1,
        sequence_length=16,
        frame_step=2,
        num_workers=0
    )

    # Initialize Model
    model = LowLightDrowsinessPipeline(num_classes=5, embed_dim=256, sequence_length=16).to(device)
    if os.path.exists(model_checkpoint):
        state_dict = torch.load(model_checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[XAI] Loaded checkpoint: {model_checkpoint}")
    else:
        print(f"[XAI WARNING] Checkpoint {model_checkpoint} not found. Running with initialized weights.")

    model.eval()
    explainer = GradCAMExplainer(model)

    class_names = ["Normal", "Slow Blinking", "Yawning", "Nodding", "Eye Closure"]

    sample_count = 0
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, batch in enumerate(val_loader):
        if sample_count >= num_samples:
            break

        video = batch["video"].to(device)
        flow = batch["flow"].to(device)
        label = batch["label"].item()

        with torch.enable_grad():
            cam = explainer.generate_cam(video, flow, target_class=label)

        # Convert first frame to numpy image
        frame_tensor = video[0, 0].cpu().numpy().transpose(1, 2, 0)
        frame_img = (frame_tensor * 255.0).clip(0, 255).astype(np.uint8)

        # Resize CAM to frame image size
        cam_resized = cv2.resize(cam, (frame_img.shape[1], frame_img.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(frame_img, 0.6, heatmap_rgb, 0.4, 0)

        # Save individual sample image
        sample_path = os.path.join(output_dir, f"xai_sample_{sample_count+1}.png")
        plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.title(f"Sample {sample_count+1} | Label: {class_names[label if label < len(class_names) else 0]}")
        plt.axis("off")
        plt.savefig(sample_path, bbox_inches="tight", dpi=150)
        plt.close()

        # Plot in grid
        axes[sample_count, 0].imshow(frame_img)
        axes[sample_count, 0].set_title(f"Input Frame (Label: {class_names[label if label < len(class_names) else 0]})")
        axes[sample_count, 0].axis("off")

        axes[sample_count, 1].imshow(cam_resized, cmap="jet")
        axes[sample_count, 1].set_title("Grad-CAM Saliency")
        axes[sample_count, 1].axis("off")

        axes[sample_count, 2].imshow(overlay)
        axes[sample_count, 2].set_title("Visual Overlay")
        axes[sample_count, 2].axis("off")

        sample_count += 1

    summary_grid_path = os.path.join(output_dir, "xai_summary_grid.png")
    fig.tight_layout()
    fig.savefig(summary_grid_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[XAI SUCCESS] Saved {sample_count} XAI visual heatmaps to {output_dir}")


if __name__ == "__main__":
    generate_xai_plots()
