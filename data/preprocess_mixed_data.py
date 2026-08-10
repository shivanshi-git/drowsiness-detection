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
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Facial Landmark Indices for Left and Right Eyes
        self.LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 144, 145, 153]
        self.RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 373, 374, 380]

    def extract_eye_crops(self, frame_bgr):
        """
        Detects facial landmarks and returns left and right eye crop images.
        If no face is detected (or image is already a pre-cropped eye like MRL), returns the original image resized.
        """
        h, w, _ = frame_bgr.shape
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            # Fallback: Assume image is already pre-cropped eye/face
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

    def build_unified_dataset(self, raw_data_dir, output_dir="processed_dataset", val_split=0.2):
        """
        Scans raw_data_dir structured as:
          raw_data_dir/
            ├── alert/ (or open_eyes, normal)
            └── drowsy/ (or closed_eyes, yawn)
        or any nested subfolders containing images/videos.
        """
        print(f"[*] Initializing Unified Data Preprocessor...")
        print(f"[*] Raw Source Directory: {raw_data_dir}")
        print(f"[*] Destination Directory: {output_dir}")

        train_dir = os.path.join(output_dir, "train")
        val_dir = os.path.join(output_dir, "val")

        for split in [train_dir, val_dir]:
            for label in ["0_alert", "1_drowsy"]:
                os.makedirs(os.path.join(split, label), exist_ok=True)

        raw_path = Path(raw_data_dir)
        files_found = list(raw_path.rglob("*"))
        
        media_files = [f for f in files_found if f.suffix.lower() in self.IMAGE_EXTS or f.suffix.lower() in self.VIDEO_EXTS]
        print(f"[*] Discovered total media files (Images & Videos): {len(media_files)}")

        np.random.seed(42)
        np.random.shuffle(media_files)

        split_idx = int(len(media_files) * (1 - val_split))
        train_files = media_files[:split_idx]
        val_files = media_files[split_idx:]

        for split_name, file_list in [("train", train_files), ("val", val_files)]:
            dest_base = train_dir if split_name == "train" else val_dir
            print(f"[*] Processing {split_name} split ({len(file_list)} files)...")

            for filepath in tqdm(file_list):
                ext = filepath.suffix.lower()
                path_str_lower = str(filepath).lower()

                # Infer label from path keywords
                if any(k in path_str_lower for k in ['closed', 'drowsy', 'yawn', 'sleep', 'micro_sleep', 'heavy']):
                    label_str = "1_drowsy"
                else:
                    label_str = "0_alert"

                if ext in self.VIDEO_EXTS:
                    self.process_video_file(str(filepath), dest_base, prefix="vid", label_str=label_str)
                elif ext in self.IMAGE_EXTS:
                    self.process_image_file(str(filepath), dest_base, prefix="img", label_str=label_str)

        print("[✓] Preprocessing Complete! Dataset saved under:", output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process mixed images & videos into unified crop dataset.")
    parser.add_argument("--raw_dir", type=str, default="raw_data", help="Directory containing raw images & videos")
    parser.add_argument("--out_dir", type=str, default="processed_dataset", help="Output directory")
    args = parser.parse_args()

    preprocessor = MixedDataPreprocessor(target_size=(128, 128), sample_every_n_frames=10)
    if os.path.exists(args.raw_dir):
        preprocessor.build_unified_dataset(args.raw_dir, args.out_dir)
    else:
        print(f"[!] Path {args.raw_dir} does not exist. Place your downloaded datasets under '{args.raw_dir}'.")
