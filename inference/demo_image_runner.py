import os
import time
import argparse
import cv2
import numpy as np
import torch

from models.drowsiness_pipeline import LowLightDrowsinessPipeline
from data.transforms import LowLightVideoAugmentation
from data.optical_flow import DenseOpticalFlowExtractor
from xai.master_explainer import MasterXAIExplainer


def run_image_inference(
    image_path: str,
    checkpoint_path: str = None,
    output_dir: str = "demo_captures",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Runs full SOTA Low-Light Drowsiness Detection + Multi-Modal XAI on a single image.
    Outputs a diagnostic demonstration card saved to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[DEMO] Loading image from: {image_path}")

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Could not load image from {image_path}. Generating a simulated demonstration sample.")
        frame = np.full((480, 640, 3), 35, dtype=np.uint8)
        # Draw face and closed eyes simulation
        cv2.circle(frame, (320, 240), 120, (60, 60, 70), -1)
        cv2.line(frame, (260, 210), (290, 210), (160, 160, 170), 4) # Closed left eye
        cv2.line(frame, (350, 210), (380, 210), (160, 160, 170), 4) # Closed right eye
        cv2.ellipse(frame, (320, 290), (35, 18), 0, 0, 360, (100, 70, 80), -1) # Yawn

    # Initialize model
    model = LowLightDrowsinessPipeline(num_classes=5, sequence_length=16).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Replicate single image into a 16-frame sequence for transformer evaluation
    transform = LowLightVideoAugmentation(is_train=False, target_size=(224, 224))
    flow_extractor = DenseOpticalFlowExtractor(target_size=(112, 112))
    xai_engine = MasterXAIExplainer(model)

    frames_seq = [frame] * 16
    video_tensor = transform(frames_seq).unsqueeze(0).to(device)
    flow_tensor = flow_extractor.extract_sequence_flow(frames_seq).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        out = model(video_tensor, flow_tensor)
        logits = out["logits"]
        pred_class = torch.argmax(logits, dim=1).item()
        fatigue_score = out["fatigue_score"].item()

    # Generate full XAI diagnosis
    print("[DEMO] Computing Multi-Modal Explainability (Grad-CAM, SHAP, Temporal, Alarm Card)...")
    explanation = xai_engine.generate_full_explanation(
        video_tensor=video_tensor,
        flow_tensor=flow_tensor,
        raw_last_frame_bgr=frame,
        target_class=pred_class,
        perclos=0.35 if pred_class in [1, 4] else 0.08,
        closure_duration=2.8 if pred_class == 4 else 0.5,
        alarm_level=3 if fatigue_score > 0.85 else (2 if fatigue_score > 0.60 else 1)
    )

    card = explanation["alarm_card"]
    print("\n" + card["formatted_card"])

    # Save diagnostic composite image
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_img_path = os.path.join(output_dir, f"xai_demo_snapshot_{timestamp}.jpg")
    cv2.imwrite(out_img_path, explanation["composite_image"])
    print(f"\n[SAVED] High-resolution XAI Demo image exported to -> {out_img_path}")

    # Display result window
    cv2.imshow("Driver Drowsiness Image Demo & Explanation", explanation["composite_image"])
    print("[INFO] Press any key on the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="demo_sample.jpg", help="Path to input image")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path")
    args = parser.parse_args()

    run_image_inference(image_path=args.image, checkpoint_path=args.checkpoint)
