import time
import collections
import argparse
import cv2
import numpy as np
import torch

from models.drowsiness_pipeline import LowLightDrowsinessPipeline
from data.transforms import LowLightVideoAugmentation
from data.optical_flow import DenseOpticalFlowExtractor
from inference.adaptive_alarm import AdaptiveAlarmSystem


def run_realtime_inference(
    video_source=0,
    checkpoint_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    seq_len=16
):
    print(f"[INFO] Initializing SOTA Low-Light Drowsiness Detection Pipeline on {device}...")
    
    # Instantiate pipeline model
    model = LowLightDrowsinessPipeline(num_classes=5, sequence_length=seq_len).to(device)
    if checkpoint_path and torch.os.path.exists(checkpoint_path):
        print(f"[INFO] Loading trained weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Preprocessing & Flow extractors
    transform = LowLightVideoAugmentation(is_train=False, target_size=(224, 224))
    flow_extractor = DenseOpticalFlowExtractor(target_size=(112, 112))
    alarm_system = AdaptiveAlarmSystem()

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video stream: {video_source}")
        return

    frame_buffer = collections.deque(maxlen=seq_len)
    fps_start_time = time.time()
    frame_count = 0
    fps = 0.0

    CLASS_NAMES = ["Normal", "Slow Blinking", "Yawning", "Nodding", "Eye Closure"]

    print("[INFO] Starting real-time stream. Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 == 0:
            fps = 10.0 / max(1e-5, (time.time() - fps_start_time))
            fps_start_time = time.time()

        frame_buffer.append(frame)
        h_orig, w_orig = frame.shape[:2]

        alarm_data = {
            "alarm_level": 0,
            "status_text": "Buffering Sequence...",
            "hud_color": (150, 150, 150),
            "smoothed_fatigue_score": 0.0,
            "perclos": 0.0,
            "closure_duration": 0.0,
            "predicted_class": 0
        }

        # Run inference once buffer is filled
        if len(frame_buffer) == seq_len:
            buffer_list = list(frame_buffer)
            video_tensor = transform(buffer_list).unsqueeze(0).to(device) # (1, T, 3, H, W)
            flow_tensor = flow_extractor.extract_sequence_flow(buffer_list).unsqueeze(0).to(device) # (1, T, 2, H, W)

            with torch.no_grad():
                out = model(video_tensor, flow_tensor)
                logits = out["logits"]
                fatigue_score = out["fatigue_score"].item()
                pred_class = torch.argmax(logits, dim=1).item()

            alarm_data = alarm_system.update(fatigue_score, pred_class, fps=fps)

        # Render HUD Overlay
        display_frame = frame.copy()
        hud_color = alarm_data["hud_color"]

        # Top Banner
        cv2.rectangle(display_frame, (0, 0), (w_orig, 70), (20, 20, 20), -1)
        cv2.putText(display_frame, f"STATUS: {alarm_data['status_text']}", (20, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, hud_color, 2)
        cv2.putText(display_frame, f"CLASS: {CLASS_NAMES[alarm_data['predicted_class']]}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Fatigue Gauge & PERCLOS
        score = alarm_data["smoothed_fatigue_score"]
        gauge_w = int(200 * score)
        cv2.rectangle(display_frame, (w_orig - 240, 15), (w_orig - 20, 35), (60, 60, 60), 1)
        cv2.rectangle(display_frame, (w_orig - 240, 15), (w_orig - 240 + gauge_w, 35), hud_color, -1)
        cv2.putText(display_frame, f"Fatigue: {score*100:.1f}%", (w_orig - 240, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.putText(display_frame, f"PERCLOS: {alarm_data['perclos']*100:.1f}% | FPS: {fps:.1f}", (w_orig - 240, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # Red Border Warning for Level 3
        if alarm_data["alarm_level"] == 3:
            cv2.rectangle(display_frame, (0, 0), (w_orig, h_orig), (0, 0, 255), 8)

        cv2.imshow("Low-Light Driver Drowsiness Monitor", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Camera index or video filepath")
    parser.add_argument("--checkpoint", default=None, help="Trained model checkpoint path")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    run_realtime_inference(video_source=src, checkpoint_path=args.checkpoint)
