import os
import glob
import cv2
import numpy as np
import mediapipe as mp
import re
from pathlib import Path
from tqdm import tqdm

def apply_clahe(img_bgr):
    """Applies CLAHE on L-channel in LAB color space."""
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
    try:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        merged = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    except Exception:
        return img_bgr

class DualROIProcessor:
    """
    Extracts Dual-ROI: Eye Crop (128x128) + Mouth Yawn Crop (128x128)
    and combines them into a side-by-side 256x128 composite ROI image.
    """
    def __init__(self, single_roi_size=(128, 128)):
        self.single_roi_size = single_roi_size
        self.output_size = (single_roi_size[0] * 2, single_roi_size[1]) # 256x128

        solutions = getattr(mp, 'solutions', None)
        self.face_mesh = None
        if solutions is not None and hasattr(solutions, 'face_mesh'):
            try:
                self.face_mesh = solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5
                )
            except Exception:
                self.face_mesh = None

        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                self.face_cascade = None
        except Exception:
            self.face_cascade = None

        # Landmark indices
        self.EYE_INDICES = [33, 133, 160, 159, 158, 144, 145, 153, 362, 263, 387, 386, 385, 373, 374, 380]
        self.MOUTH_INDICES = [61, 81, 13, 311, 291, 402, 14, 178, 0, 17, 37, 267, 84, 314]

    def extract_dual_roi(self, frame_bgr):
        """
        Extracts Eye ROI and Mouth ROI, resizes each to 128x128, and concatenates horizontally (256x128).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        h, w, _ = frame_bgr.shape
        eye_crop = None
        mouth_crop = None

        # 1. MediaPipe Face Mesh Extraction
        if self.face_mesh is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Eye bounding box
                eye_pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in self.EYE_INDICES])
                ex_min, ey_min = np.min(eye_pts, axis=0)
                ex_max, ey_max = np.max(eye_pts, axis=0)
                pad_e = int(max(ex_max - ex_min, ey_max - ey_min) * 0.3)
                ey1, ey2 = max(0, ey_min - pad_e), min(h, ey_max + pad_e)
                ex1, ex2 = max(0, ex_min - pad_e), min(w, ex_max + pad_e)
                if ey2 > ey1 and ex2 > ex1:
                    eye_crop = cv2.resize(frame_bgr[ey1:ey2, ex1:ex2], self.single_roi_size)

                # Mouth bounding box
                mouth_pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in self.MOUTH_INDICES])
                mx_min, my_min = np.min(mouth_pts, axis=0)
                mx_max, my_max = np.max(mouth_pts, axis=0)
                pad_m = int(max(mx_max - mx_min, my_max - my_min) * 0.3)
                my1, my2 = max(0, my_min - pad_m), min(h, my_max + pad_m)
                mx1, mx2 = max(0, mx_min - pad_m), min(w, mx_max + pad_m)
                if my2 > my1 and mx2 > mx1:
                    mouth_crop = cv2.resize(frame_bgr[my1:my2, mx1:mx2], self.single_roi_size)

        # Fallback to OpenCV Cascade if MediaPipe landmarks failed
        if (eye_crop is None or mouth_crop is None) and self.face_cascade is not None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                fx, fy, fw, fh = faces[0]
                if eye_crop is None:
                    eye_roi = frame_bgr[fy:fy + int(fh * 0.5), fx:fx + fw]
                    eye_crop = cv2.resize(eye_roi, self.single_roi_size) if eye_roi.size > 0 else None
                if mouth_crop is None:
                    mouth_roi = frame_bgr[fy + int(fh * 0.5):fy + fh, fx:fx + fw]
                    mouth_crop = cv2.resize(mouth_roi, self.single_roi_size) if mouth_roi.size > 0 else None

        # Ultimate fallback: crop upper half (eye) & lower half (mouth)
        if eye_crop is None:
            eye_crop = cv2.resize(frame_bgr[:h // 2, :], self.single_roi_size)
        if mouth_crop is None:
            mouth_crop = cv2.resize(frame_bgr[h // 2:, :], self.single_roi_size)

        # CLAHE contrast equalization
        eye_crop = apply_clahe(eye_crop)
        mouth_crop = apply_clahe(mouth_crop)

        # Side-by-side concatenation: [Eye Crop | Mouth Crop] -> (128, 256, 3)
        dual_roi_composite = np.hstack([eye_crop, mouth_crop])
        return dual_roi_composite

def process_dataset_folder(src_dir="processed_dataset", out_dir="processed_dual_dataset"):
    """
    Scans src_dir (train/val splits with 0_alert / 1_drowsy),
    applies Dual-ROI extraction (Eye + Mouth Yawn) to generate 256x128 composite crops,
    and saves to out_dir preserving class splits and subject isolation.
    """
    processor = DualROIProcessor()
    if not os.path.exists(src_dir):
        print(f"[!] Source dataset directory '{src_dir}' not found.")
        return

    print(f"[*] Extracting Dual-ROI (Eye + Mouth) from '{src_dir}' to '{out_dir}'...")
    img_exts = {'.png', '.jpg', '.jpeg', '.bmp'}

    total_images_processed = 0
    stats = {}

    for split in ['train', 'val']:
        stats[split] = {}
        for cls_name in ['0_alert', '1_drowsy']:
            in_folder = os.path.join(src_dir, split, cls_name)
            out_folder = os.path.join(out_dir, split, cls_name)
            if not os.path.exists(in_folder):
                stats[split][cls_name] = 0
                continue
            os.makedirs(out_folder, exist_ok=True)

            file_list = [Path(p) for p in glob.glob(os.path.join(in_folder, "*")) if Path(p).suffix.lower() in img_exts]
            print(f"[*] Processing {split}/{cls_name} ({len(file_list)} images)...")

            count = 0
            for img_path in tqdm(file_list):
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    continue
                
                dual_crop = processor.extract_dual_roi(img_bgr)
                if dual_crop is not None:
                    out_filepath = os.path.join(out_folder, img_path.name)
                    cv2.imwrite(out_filepath, dual_crop)
                    count += 1

            stats[split][cls_name] = count
            total_images_processed += count

    # Calculate total size on disk
    total_bytes = 0
    for root, _, files in os.walk(out_dir):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total_bytes += os.path.getsize(fp)

    size_mb = total_bytes / (1024 * 1024)
    size_gb = total_bytes / (1024 * 1024 * 1024)
    size_str = f"{size_gb:.2f} GB" if size_gb >= 1.0 else f"{size_mb:.2f} MB"

    print("\n==================================================")
    print("        DUAL-ROI PREPROCESSED DATASET SUMMARY     ")
    print("==================================================")
    print(f"Output Directory:    {out_dir}")
    print(f"Total Disk Size:     {size_str} ({total_bytes:,} bytes)")
    print(f"Total Images:        {total_images_processed:,} composite 256x128 images")
    print(f"Train Split:         {stats.get('train', {}).get('0_alert', 0):,} Alert | {stats.get('train', {}).get('1_drowsy', 0):,} Drowsy")
    print(f"Val Split:           {stats.get('val', {}).get('0_alert', 0):,} Alert | {stats.get('val', {}).get('1_drowsy', 0):,} Drowsy")
    print("==================================================\n")
    print(f"[✓] Dual-ROI Preprocessing Complete! Ready for model training.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract Dual-ROI (Eye + Mouth) composite images.")
    parser.add_argument("--src_dir", type=str, default="processed_dataset")
    parser.add_argument("--out_dir", type=str, default="processed_dual_dataset")
    args = parser.parse_args()

    process_dataset_folder(src_dir=args.src_dir, out_dir=args.out_dir)
