import os
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
    show_xai_dashboard=True,
    output_capture_dir="demo_captures"
):
    os.makedirs(output_capture_dir, exist_ok=True)
    print(f"============================================================")
    print(f" SOTA Low-Light Driver Drowsiness Detection Live Demo")
    print(f" Device: {device} | Source: {video_source}")
    print(f" Key Controls:")
    print(f"   [C] Capture & Save Snapshot + Full XAI Diagnosis Receipt")
    print(f"   [X] Toggle Explainability (XAI) Panel")
    print(f"   [E] Toggle LLFormer Enhancement Side-by-Side")
    print(f"   [Q] Quit Demo")
    print(f"============================================================")
    
    model = LowLightDrowsinessPipeline(num_classes=5, sequence_length=seq_len).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[INFO] Loaded model checkpoint from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    transform = LowLightVideoAugmentation(is_train=False, target_size=(224, 224))
    flow_extractor = DenseOpticalFlowExtractor(target_size=(112, 112))
    alarm_system = AdaptiveAlarmSystem()
    xai_engine = MasterXAIExplainer(model)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Could not access camera/video stream at {video_source}.")
        print("[FALLBACK] Launching simulated interactive camera demo feed...")
        # Interactive simulated webcam generator
        simulate_feed = True
    else:
        simulate_feed = False

    frame_buffer = collections.deque(maxlen=seq_len)
    fps_start_time = time.time()
    frame_count = 0
    fps = 0.0
    show_enhanced_preview = False
    xai_composite = None
    last_explanation = None
    notification_msg = "Demo Running | Press 'C' to snapshot image"
    notification_timer = 0.0

    CLASS_NAMES = ["Normal", "Slow Blinking", "Yawning", "Nodding", "Eye Closure"]

    while True:
        frame_count += 1
        if simulate_feed:
            # Simulate live camera feed with periodic blinks/yawns
            frame = np.full((480, 640, 3), 30, dtype=np.uint8)
            cv2.circle(frame, (320, 240), 110, (55, 55, 65), -1)
            is_drowsy_cycle = (frame_count % 90) > 45
            eye_h = 2 if is_drowsy_cycle else 10
            cv2.ellipse(frame, (270, 210), (15, eye_h), 0, 0, 360, (140, 140, 150), -1)
            cv2.ellipse(frame, (370, 210), (15, eye_h), 0, 0, 360, (140, 140, 150), -1)
            mouth_h = 20 if is_drowsy_cycle else 6
            cv2.ellipse(frame, (320, 290), (25, mouth_h), 0, 0, 360, (100, 70, 80), -1)
            time.sleep(0.03)
        else:
            ret, frame = cap.read()
            if not ret:
                break

        if frame_count % 10 == 0:
            fps = 10.0 / max(1e-5, (time.time() - fps_start_time))
            fps_start_time = time.time()

        frame_buffer.append(frame)
        h_orig, w_orig = frame.shape[:2]

        alarm_data = {
            "alarm_level": 0,
            "status_text": "Buffering sequence...",
            "hud_color": (150, 150, 150),
            "smoothed_fatigue_score": 0.0,
            "perclos": 0.0,
            "closure_duration": 0.0,
            "predicted_class": 0
        }

        # Run inference once buffer is populated
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

            # Update XAI dashboard periodically
            if show_xai_dashboard and (frame_count % 5 == 0 or alarm_data["alarm_level"] >= 1):
                try:
                    last_explanation = xai_engine.generate_full_explanation(
                        video_tensor=video_tensor,
                        flow_tensor=flow_tensor,
                        raw_last_frame_bgr=frame,
                        target_class=pred_class,
                        perclos=alarm_data["perclos"],
                        closure_duration=alarm_data["closure_duration"],
                        alarm_level=alarm_data["alarm_level"]
                    )
                    xai_composite = last_explanation["composite_image"]
                except Exception:
                    pass

        # Select display frame
        if show_xai_dashboard and xai_composite is not None:
            display_frame = xai_composite.copy()
            cur_w = display_frame.shape[1]
        else:
            display_frame = frame.copy()
            cur_w = w_orig

        hud_color = alarm_data["hud_color"]

        # Top Banner
        cv2.rectangle(display_frame, (0, 0), (cur_w, 65), (20, 20, 20), -1)
        cv2.putText(display_frame, f"STATUS: {alarm_data['status_text']}", (20, 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, hud_color, 2)
        cv2.putText(display_frame, f"STATE: {CLASS_NAMES[alarm_data['predicted_class']]} | FPS: {fps:.1f}", (20, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

        # Fatigue & PERCLOS HUD Gauge
        score = alarm_data["smoothed_fatigue_score"]
        gauge_w = int(180 * score)
        cv2.rectangle(display_frame, (cur_w - 220, 12), (cur_w - 20, 28), (60, 60, 60), 1)
        cv2.rectangle(display_frame, (cur_w - 220, 12), (cur_w - 220 + gauge_w, 28), hud_color, -1)
        cv2.putText(display_frame, f"Fatigue: {score*100:.1f}% | PERCLOS: {alarm_data['perclos']*100:.1f}%",
                    (cur_w - 220, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

        # Bottom Controls / Notification Ribbon
        cv2.rectangle(display_frame, (0, display_frame.shape[0] - 28), (cur_w, display_frame.shape[0]), (15, 15, 15), -1)
        if time.time() - notification_timer < 3.0:
            cv2.putText(display_frame, notification_msg, (15, display_frame.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        else:
            cv2.putText(display_frame, "[C] Capture Snapshot & XAI Card | [X] Toggle XAI | [Q] Quit",
                        (15, display_frame.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1)

        # Red Border Warning
        if alarm_data["alarm_level"] == 3:
            cv2.rectangle(display_frame, (0, 0), (cur_w, display_frame.shape[0]), (0, 0, 255), 6)

        cv2.imshow("Driver Drowsiness Detection: Live Demonstration & XAI", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('x'):
            show_xai_dashboard = not show_xai_dashboard
            notification_msg = f"XAI Dashboard: {'ENABLED' if show_xai_dashboard else 'DISABLED'}"
            notification_timer = time.time()
        elif key == ord('c'):
            # Save instant demonstration snapshot card
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            snap_path = os.path.join(output_capture_dir, f"demo_capture_{timestamp}.png")
            cv2.imwrite(snap_path, display_frame)
            
            # If we have an explanation card, save receipt text
            if last_explanation and "alarm_card" in last_explanation:
                receipt_path = os.path.join(output_capture_dir, f"demo_diagnosis_{timestamp}.txt")
                with open(receipt_path, "w", encoding="utf-8") as f:
                    f.write(last_explanation["alarm_card"]["formatted_card"])
                print(f"[DEMO SNAPSHOT] Saved XAI Card Image -> {snap_path}")
                print(f"[DEMO SNAPSHOT] Saved Diagnosis Receipt -> {receipt_path}")
            
            notification_msg = f"CAPTURED! Saved to {os.path.basename(snap_path)}"
            notification_timer = time.time()

    if not simulate_feed:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Camera index (0, 1) or video filepath")
    parser.add_argument("--checkpoint", default=None, help="Trained model checkpoint path")
    parser.add_argument("--no-xai", action="store_true", help="Disable XAI dashboard view by default")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    run_realtime_inference(
        video_source=src,
        checkpoint_path=args.checkpoint,
        show_xai_dashboard=not args.no_xai
    )
