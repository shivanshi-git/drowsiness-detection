import os
import glob
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm

class MixedDataPreprocessor:
    """
    Scans raw dataset folders containing mixed images (.png, .jpg, .jpeg) and videos (.mp4, .avi, .mov).
    Extracts eye & face ROIs using MediaPipe Face Mesh, normalizes resolution to target_size (default 128x128),
    and organizes cropped images into a clean PyTorch ImageFolder format.
    """
    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
    VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv'}

    def __init__(self, target_size=(128, 128), sample_every_n_frames=10):
        self.target_size = target_size
        self.sample_every_n_frames = sample_every_n_frames
        
        # Initialize MediaPipe Face Mesh if available, or fallback to OpenCV
        self.face_mesh = None
        if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
            try:
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except Exception:
                self.face_mesh = None

        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Facial Landmark Indices for Left and Right Eyes
        self.LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 144, 145, 153]
        self.RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 373, 374, 380]

    def extract_eye_crops(self, frame_bgr):
        """
        Detects facial landmarks and returns left and right eye crop images.
        If no face is detected (or image is already a pre-cropped eye like MRL), returns the original image resized.
        """
        h, w, _ = frame_bgr.shape
        
        if self.face_mesh is not None:
            try:
                rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)

                if results and results.multi_face_landmarks:
                    crops = []
                    landmarks = results.multi_face_landmarks[0].landmark

                    for eye_indices in [self.LEFT_EYE_INDICES, self.RIGHT_EYE_INDICES]:
                        pts = np.array([(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in eye_indices])
                        x, y, eye_w, eye_h = cv2.boundingRect(pts)
                        
                        pad_x = int(eye_w * 0.4)
                        pad_y = int(eye_h * 0.4)
                        x1 = max(0, x - pad_x)
                        y1 = max(0, y - pad_y)
                        x2 = min(w, x + eye_w + pad_x)
                        y2 = min(h, y + eye_h + pad_y)

                        eye_crop = frame_bgr[y1:y2, x1:x2]
                        if eye_crop.size > 0:
                            resized_crop = cv2.resize(eye_crop, self.target_size)
                            crops.append(resized_crop)

                    if crops:
                        return crops
            except Exception:
                pass

        # Fallback: Image is already pre-cropped eye/face or fallback to resize
        resized = cv2.resize(frame_bgr, self.target_size)
        return [resized]

        crops = []
        landmarks = results.multi_face_landmarks[0].landmark

        for eye_indices in [self.LEFT_EYE_INDICES, self.RIGHT_EYE_INDICES]:
            pts = np.array([(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in eye_indices])
            x, y, eye_w, eye_h = cv2.boundingRect(pts)
            
            # Add padding
            pad_x = int(eye_w * 0.4)
            pad_y = int(eye_h * 0.4)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + eye_w + pad_x)
            y2 = min(h, y + eye_h + pad_y)

            eye_crop = frame_bgr[y1:y2, x1:x2]
            if eye_crop.size > 0:
                resized_crop = cv2.resize(eye_crop, self.target_size)
                crops.append(resized_crop)

        return crops if crops else [cv2.resize(frame_bgr, self.target_size)]

    def process_video_file(self, video_path, output_dir, prefix, label_str):
        """
        Reads a video file, samples frames, extracts eye crops, and saves them as images.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Warning] Unable to open video: {video_path}")
            return 0

        frame_count = 0
        saved_count = 0
        video_name = Path(video_path).stem

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % self.sample_every_n_frames == 0:
                crops = self.extract_eye_crops(frame)
                for idx, crop in enumerate(crops):
                    out_filename = f"{prefix}_{video_name}_f{frame_count}_crop{idx}.jpg"
                    out_filepath = os.path.join(output_dir, label_str, out_filename)
                    cv2.imwrite(out_filepath, crop)
                    saved_count += 1

            frame_count += 1

        cap.release()
        return saved_count

    def process_image_file(self, image_path, output_dir, prefix, label_str):
        """
        Reads a single image file, extracts eye crops, and saves them.
        """
        frame = cv2.imread(image_path)
        if frame is None:
            return 0

        img_name = Path(image_path).stem
        crops = self.extract_eye_crops(frame)
        saved_count = 0

        for idx, crop in enumerate(crops):
            out_filename = f"{prefix}_{img_name}_crop{idx}.jpg"
            out_filepath = os.path.join(output_dir, label_str, out_filename)
            cv2.imwrite(out_filepath, crop)
            saved_count += 1

        return saved_count

    def build_unified_dataset(self, raw_data_dirs, output_dir="processed_dataset", val_split=0.2, max_samples_per_class=3000):
        """
        Scans raw_data_dirs (list of folders or single folder) structured with subfolders.
        Extracts crops and organizes them into 'processed_dataset/train' and 'processed_dataset/val'.
        """
        if isinstance(raw_data_dirs, str):
            raw_data_dirs = [raw_data_dirs]

        print(f"[*] Initializing Unified Data Preprocessor...")
        print(f"[*] Source Directories: {raw_data_dirs}")
        print(f"[*] Destination Directory: {output_dir}")

        train_dir = os.path.join(output_dir, "train")
        val_dir = os.path.join(output_dir, "val")

        for split in [train_dir, val_dir]:
            for label in ["0_alert", "1_drowsy"]:
                os.makedirs(os.path.join(split, label), exist_ok=True)

        alert_files = []
        drowsy_files = []

        for raw_dir in raw_data_dirs:
            if not os.path.exists(raw_dir):
                print(f"[!] Warning: Directory '{raw_dir}' does not exist. Skipping.")
                continue

            raw_path = Path(raw_dir)
            files_found = list(raw_path.rglob("*"))
            media_files = [f for f in files_found if f.suffix.lower() in self.IMAGE_EXTS or f.suffix.lower() in self.VIDEO_EXTS]

            for filepath in media_files:
                path_str_lower = str(filepath).lower()

                # Infer label robustly
                if any(k in path_str_lower for k in ['no_yawn', 'open', 'awake', 'notdrowsy', 'active', 'normal', 'alert']):
                    alert_files.append(filepath)
                elif any(k in path_str_lower for k in ['closed', 'drowsy', 'sleepy', 'fatigue', 'yawn', 'sleep', 'micro_sleep', 'heavy']):
                    drowsy_files.append(filepath)

        print(f"[*] Discovered Total Alert Files: {len(alert_files)}")
        print(f"[*] Discovered Total Drowsy Files: {len(drowsy_files)}")

        np.random.seed(42)
        np.random.shuffle(alert_files)
        np.random.shuffle(drowsy_files)

        if max_samples_per_class and max_samples_per_class > 0:
            alert_files = alert_files[:max_samples_per_class]
            drowsy_files = drowsy_files[:max_samples_per_class]
            print(f"[*] Subsampling to balanced {len(alert_files)} Alert and {len(drowsy_files)} Drowsy samples for efficient model training.")

        all_media = [(f, "0_alert") for f in alert_files] + [(f, "1_drowsy") for f in drowsy_files]
        np.random.shuffle(all_media)

        split_idx = int(len(all_media) * (1 - val_split))
        train_items = all_media[:split_idx]
        val_items = all_media[split_idx:]

        for split_name, item_list in [("train", train_items), ("val", val_items)]:
            dest_base = train_dir if split_name == "train" else val_dir
            print(f"[*] Processing {split_name} split ({len(item_list)} files)...")

            for filepath, label_str in tqdm(item_list):
                ext = filepath.suffix.lower()
                prefix = filepath.parent.name

                if ext in self.VIDEO_EXTS:
                    self.process_video_file(str(filepath), dest_base, prefix=prefix, label_str=label_str)
                elif ext in self.IMAGE_EXTS:
                    self.process_image_file(str(filepath), dest_base, prefix=prefix, label_str=label_str)

        print("[✓] Preprocessing Complete! Dataset saved under:", output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process mixed images & videos into unified crop dataset.")
    parser.add_argument("--raw_dirs", nargs="+", default=["archive", "archive(1)", "archive(2)", "archive(3)"], help="Directories containing raw datasets")
    parser.add_argument("--out_dir", type=str, default="processed_dataset", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=3000, help="Max samples per class for training speed")
    args = parser.parse_args()

    preprocessor = MixedDataPreprocessor(target_size=(128, 128), sample_every_n_frames=10)
    preprocessor.build_unified_dataset(args.raw_dirs, args.out_dir, max_samples_per_class=args.max_samples)

