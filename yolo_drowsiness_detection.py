import os
import sys
import time
import argparse
import cv2
import torch
import numpy as np

def load_yolo_model(weights_path="yolov5s", conf_thresh=0.4, device="cpu"):
    """
    Load YOLOv5 model via PyTorch Hub. Supports pretrained models ('yolov5s') 
    or custom trained checkpoint files (.pt).
    """
    print(f"[*] Loading YOLOv5 model from: {weights_path}...")
    device_obj = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')

    if os.path.exists(weights_path):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=False)
        print(f"[✓] Custom weight loaded: {weights_path}")
    else:
        print(f"[ℹ] Path '{weights_path}' not found as local file. Loading hub pretrained model '{weights_path}'.")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    model.conf = conf_thresh  # NMS confidence threshold
    model.to(device_obj)
    model.eval()
    return model, device_obj

def run_realtime_detection(weights="yolov5s", source="0", conf=0.4, device="cpu"):
    """
    Real-time drowsiness detection loop using OpenCV webcam/video stream and YOLOv5 model.
    """
    model, device_obj = load_yolo_model(weights_path=weights, conf_thresh=conf, device=device)

    # Determine video source
    video_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"[ERROR] Unable to open video source: {source}")
        return

    print("[*] Starting real-time detection stream. Press 'q' or 'ESC' to exit.")

    prev_time = time.time()
    fps = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[ℹ] End of video stream or feed disconnected.")
            break

        # Calculate FPS
        curr_time = time.time()
        time_diff = curr_time - prev_time
        if time_diff > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / time_diff)
        prev_time = curr_time

        # Perform YOLO inference
        results = model(frame)
        rendered_frame = np.squeeze(results.render())

        # Check detected labels for drowsiness warning
        df = results.pandas().xyxy[0]
        is_drowsy = False
        if not df.empty and 'name' in df.columns:
            drowsy_matches = df[df['name'].str.lower() == 'drowsy']
            if len(drowsy_matches) > 0:
                is_drowsy = True

        # Render status banner
        if is_drowsy:
            status_text = f"ALERT: DROWSY DRIVER DETECTED | FPS: {fps:.1f}"
            color = (0, 0, 255) # Red
        else:
            status_text = f"STATUS: NORMAL / ALERT | FPS: {fps:.1f}"
            color = (0, 255, 0) # Green

        cv2.putText(rendered_frame, status_text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        cv2.imshow('YOLO Real-Time Drowsiness Detection', rendered_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: # 'q' or ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[✓] Video stream closed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Drowsiness Detection with YOLOv5")
    parser.add_argument("--weights", type=str, default="yolov5s", help="Path to weights .pt file or hub model name (e.g., 'yolov5s')")
    parser.add_argument("--source", type=str, default="0", help="Webcam index (0) or video file path")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold for detections")
    parser.add_argument("--device", type=str, default="cpu", help="Execution device: 'cpu' or 'cuda'")
    args = parser.parse_args()

    run_realtime_detection(weights=args.weights, source=args.source, conf=args.conf, device=args.device)
