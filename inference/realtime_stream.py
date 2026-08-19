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
from xai.master_explainer import MasterXAIExplainer


def run_realtime_inference(
    video_source=0,
    checkpoint_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    seq_len=16,
    show_xai_dashboard=True
):
    print(f"[INFO] Initializing SOTA Low-Light Drowsiness Detection + XAI Pipeline on {device}...")
    
    model = LowLightDrowsinessPipeline(num_classes=5, sequence_length=seq_len).to(device)
    if checkpoint_path and torch.os.path.exists(checkpoint_path):
        print(f"[INFO] Loading trained weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Preprocessing, flow, alarm & XAI modules
    transform = LowLightVideoAugmentation(is_train=False, target_size=(224, 224))
    flow_extractor = DenseOpticalFlowExtractor(target_size=(112, 112))
    alarm_system = AdaptiveAlarmSystem()
    xai_engine = MasterXAIExplainer(model)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video stream: {video_source}")
        return

    frame_buffer = collections.deque(maxlen=seq_len)
    fps_start_time = time.time()
    frame_count = 0
    fps = 0.0

    CLASS_NAMES = ["Normal", "Slow Blinking", "Yawning", "Nodding", "Eye Closure"]

    print("[INFO] Starting real-time stream with XAI HUD. Press 'x' to toggle XAI view, 'q' to exit.")

    xai_composite = None

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
            video_tensor = transform(buffer_list).unsqueeze(0).to(device)
            flow_tensor = flow_extractor.extract_sequence_flow(buffer_list).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(video_tensor, flow_tensor)
                logits = out["logits"]
                fatigue_score = out["fatigue_score"].item()
                pred_class = torch.argmax(logits, dim=1).item()

            alarm_data = alarm_system.update(fatigue_score, pred_class, fps=fps)

            # Generate XAI visual explanation periodically or during alerts
            if show_xai_dashboard and (frame_count % 6 == 0 or alarm_data["alarm_level"] >= 1):
                try:
                    explanation = xai_engine.generate_full_explanation(
                        video_tensor,
                        flow_tensor,
                        raw_last_frame_bgr=frame,
                        target_class=pred_class
                    )
                    xai_composite = explanation["composite_image"]
                except Exception:
                    pass

        # Decide display frame
        if show_xai_dashboard and xai_composite is not None:
            display_frame = xai_composite.copy()
            cur_w = display_frame.shape[1]
        else:
            display_frame = frame.copy()
            cur_w = w_orig

        hud_color = alarm_data["hud_color"]

        # Top Banner
        cv2.rectangle(display_frame, (0, 0), (cur_w, 65), (20, 20, 20), -1)
        cv2.putText(display_frame, f"STATUS: {alarm_data['status_text']}", (20, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, hud_color, 2)
        cv2.putText(display_frame, f"CLASS: {CLASS_NAMES[alarm_data['predicted_class']]}", (20, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Fatigue & PERCLOS HUD
        score = alarm_data["smoothed_fatigue_score"]
        gauge_w = int(180 * score)
        cv2.rectangle(display_frame, (cur_w - 220, 12), (cur_w - 20, 30), (60, 60, 60), 1)
        cv2.rectangle(display_frame, (cur_w - 220, 12), (cur_w - 220 + gauge_w, 30), hud_color, -1)
        cv2.putText(display_frame, f"Fatigue: {score*100:.1f}% | PERCLOS: {alarm_data['perclos']*100:.1f}%",
                    (cur_w - 220, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        # Red Border Alert for Level 3
        if alarm_data["alarm_level"] == 3:
            cv2.rectangle(display_frame, (0, 0), (cur_w, display_frame.shape[0]), (0, 0, 255), 8)

        cv2.imshow("Low-Light Driver Drowsiness + Explainability Monitor", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('x'):
            show_xai_dashboard = not show_xai_dashboard

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Camera index or video filepath")
    parser.add_argument("--checkpoint", default=None, help="Trained model checkpoint path")
    parser.add_argument("--no-xai", action="store_true", help="Disable XAI dashboard view")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    run_realtime_inference(
        video_source=src,
        checkpoint_path=args.checkpoint,
        show_xai_dashboard=not args.no_xai
    )
