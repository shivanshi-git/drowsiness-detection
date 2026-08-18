import os
import cv2
import torch
import numpy as np
import argparse
from PIL import Image
from torchvision import transforms

from models.model_factory import get_model
from xai.grad_cam import GradCAM
from xai.visualizer import overlay_heatmap

def run_prediction(image_path, model_name='resnet18', checkpoint_path=None, device='cpu'):
    """
    Infers drowsiness prediction on an input image and returns blended Grad-CAM heatmap.
    """
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model, target_layer = get_model(model_name=model_name, num_classes=2, pretrained=False)

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        class_names = checkpoint.get('class_names', ['0_alert', '1_drowsy'])
        print(f"[*] Loaded Checkpoint from: {checkpoint_path}")
    else:
        print("[!] No checkpoint provided or found. Using initialized model weights.")
        class_names = ['0_alert', '1_drowsy']

    model = model.to(device)
    model.eval()

    # Load and preprocess image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Compute Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    heatmap, pred_class_idx, confidence = grad_cam.generate_heatmap(input_tensor)
    grad_cam.remove_hooks()

    pred_label = class_names[pred_class_idx]
    
    # Overlay heatmap over original BGR image
    blended_bgr, _ = overlay_heatmap(image_bgr, heatmap)

    # Draw label text
    status_text = f"{pred_label.upper()} ({confidence*100:.1f}%)"
    text_color = (0, 0, 255) if 'drowsy' in pred_label.lower() else (0, 255, 0)
    cv2.putText(blended_bgr, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2, cv2.LINE_AA)

    return pred_label, confidence, blended_bgr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict drowsiness & generate Grad-CAM for a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out", type=str, default="prediction_xai_output.jpg")
    args = parser.parse_args()

    label, conf, result_img = run_prediction(args.image, model_name=args.model, checkpoint_path=args.checkpoint)
    cv2.imwrite(args.out, result_img)
    print(f"[✓] Prediction: {label} ({conf*100:.2f}% confidence)")
    print(f"[✓] Grad-CAM Output saved to: {args.out}")
