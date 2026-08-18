import os
import glob
import cv2
import numpy as np
import mediapipe as mp
import re
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

def extract_group_id(filepath):
    """
    Extracts the subject/group identifier from folder structure or filename:
    - NTHU-DDD: Uses subject ID prefix (e.g., '001', '002', '003', etc. from '001_glasses_...')
    - MRL Eye Dataset: Uses subject ID prefix (e.g., 's0001', 's0002', 's0013')
    - UTA-RLDD / Kaggle: Uses subject / participant folder or prefix (e.g., 'subject1', 'sub_05')
    - Fallback: Uses the video file stem or parent directory name.
    """
    path_obj = Path(filepath)
    filename = path_obj.name
    full_path_str = str(path_obj)

    # Pattern 1: NTHU-DDD (e.g., 001_glasses_sleepyCombination_1000_drowsy.jpg)
    nthu_match = re.match(r'^(\d{3})_', filename)
    if nthu_match:
        return f"nthu_{nthu_match.group(1)}"

    # Pattern 2: MRL Eye Dataset (e.g., s0013_01946_0_1_0_0_0_01.png)
    mrl_match = re.search(r'(s\d{3,4})', filename, re.IGNORECASE)
    if mrl_match:
        return f"mrl_{mrl_match.group(1).lower()}"

    # Pattern 3: Subject / Sub / Participant pattern in path or filename
    subj_match = re.search(r'(subject[_\-]?\d+|participant[_\-]?\d+|sub[_\-]?\d+|\d{2}_\d{2})', full_path_str, re.IGNORECASE)
    if subj_match:
        return subj_match.group(0).lower()

    # Pattern 4: Parent directory is a numeric subject ID (e.g., /1/, /01/, /18/)
    for part in reversed(path_obj.parts[:-1]):
        if part.isdigit() and len(part) <= 3:
            return f"subj_dir_{part}"

    # Fallback: Video file stem or parent directory name
    if path_obj.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'}:
        return f"vid_{path_obj.stem}"
    return f"dir_{path_obj.parent.name}"

def apply_clahe(img_bgr):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) on L-channel
    in LAB color space to equalize IR night vision and daylight RGB lighting conditions.
    """
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

class MixedDataPreprocessor:
    """
    Scans raw dataset folders containing mixed images (.png, .jpg, .jpeg) and videos (.mp4, .avi, .mov).
    Extracts eye & face ROIs using a Cascaded 3-Tier Detector (MediaPipe -> OpenCV Cascade -> Anatomical Upper-Third Crop),
    applies CLAHE contrast equalization, normalizes resolution to target_size (default 128x128),
    and organizes cropped images into a clean PyTorch ImageFolder format with zero data leakage.
    """
    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
    VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv'}

    def __init__(self, target_size=(128, 128), sample_every_n_frames=10):
        self.target_size = target_size
        self.sample_every_n_frames = sample_every_n_frames
        
        # Initialize MediaPipe Face Mesh if available, or fallback to OpenCV
        self.face_mesh = None
        solutions = getattr(mp, 'solutions', None)
        if solutions is not None and hasattr(solutions, 'face_mesh'):
            try:
                self.face_mesh = solutions.face_mesh.FaceMesh(
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
        Extracts eye crops using a Cascaded 3-Tier Detector:
        - Tier 1: MediaPipe Face Mesh landmark extraction.
        - Tier 2: OpenCV Cascade Face Detection + Upper-Half Eye Region Crop.
        - Tier 3: Anatomical Upper-Third Crop Fallback (eliminates full-frame scale artifacts).
        Applies CLAHE contrast equalization to all extracted crops.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        h, w, _ = frame_bgr.shape
        crops = []

        # TIER 1: MediaPipe Face Mesh Extraction
        if self.face_mesh is not None:
            try:
                rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)

                if results and results.multi_face_landmarks:
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
                            eq_crop = apply_clahe(resized_crop)
                            crops.append(eq_crop)

                    if crops:
                        return crops
            except Exception:
                pass

        # TIER 2: OpenCV Cascade Face Detection + Upper-Half Eye Region Crop
        if self.cascade is not None and not self.cascade.empty():
            try:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                if len(faces) > 0:
                    fx, fy, fw, fh = faces[0]
                    # Eye region typically spans upper 25% to 60% of face bounding box
                    eye_y1 = max(0, fy + int(fh * 0.22))
                    eye_y2 = min(h, fy + int(fh * 0.58))
                    eye_x1 = max(0, fx + int(fw * 0.10))
                    eye_x2 = min(w, fx + int(fw * 0.90))

                    face_eye_crop = frame_bgr[eye_y1:eye_y2, eye_x1:eye_x2]
                    if face_eye_crop.size > 0:
                        resized_crop = cv2.resize(face_eye_crop, self.target_size)
                        eq_crop = apply_clahe(resized_crop)
                        return [eq_crop]
            except Exception:
                pass

        # TIER 3: Check if image is already a pre-cropped eye (e.g., MRL eye dataset < 150x150)
        if w < 160 and h < 160:
            resized = cv2.resize(frame_bgr, self.target_size)
            eq_crop = apply_clahe(resized)
            return [eq_crop]

        # TIER 3 Fallback: Anatomical Upper-Third Crop for large video frames (prevents full-frame squeeze distortion)
        crop_y1 = max(0, int(h * 0.15))
        crop_y2 = min(h, int(h * 0.55))
        crop_x1 = max(0, int(w * 0.10))
        crop_x2 = min(w, int(w * 0.90))
        upper_third_crop = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

        if upper_third_crop.size > 0:
            resized = cv2.resize(upper_third_crop, self.target_size)
        else:
            resized = cv2.resize(frame_bgr, self.target_size)

        eq_crop = apply_clahe(resized)
        return [eq_crop]

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

    def build_unified_dataset(self, raw_data_dirs, output_dir="processed_dataset", val_split=0.15, test_split=0.15, max_samples_per_class=3000):
        """
        Scans raw_data_dirs (list of folders or single folder).
        Performs group-aware splitting at subject/video level so 100% of any subject's data stays in train OR val.
        Extracts crops and organizes them into disjoint train, val, and test splits.
        """
        if isinstance(raw_data_dirs, str):
            raw_data_dirs = [raw_data_dirs]
        if val_split < 0 or test_split < 0 or val_split + test_split >= 1:
            raise ValueError("val_split and test_split must be non-negative and sum to less than 1")

        print(f"[*] Initializing Unified Data Preprocessor (Group-Aware Split)...")
        print(f"[*] Source Directories: {raw_data_dirs}")
        print(f"[*] Destination Directory: {output_dir}")

        if os.path.exists(output_dir):
            import shutil
            print(f"[*] Cleaning existing output directory: '{output_dir}'...")
            shutil.rmtree(output_dir)

        train_dir = os.path.join(output_dir, "train")
        val_dir = os.path.join(output_dir, "val")
        test_dir = os.path.join(output_dir, "test")

        for split in [train_dir, val_dir, test_dir]:
            for label in ["0_alert", "1_drowsy"]:
                os.makedirs(os.path.join(split, label), exist_ok=True)

        # Collect files by class and group ID
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

        # Group items by Subject/Group ID
        groups = defaultdict(lambda: {'alert': [], 'drowsy': []})
        for filepath in alert_files:
            gid = extract_group_id(filepath)
            groups[gid]['alert'].append(filepath)

        for filepath in drowsy_files:
            gid = extract_group_id(filepath)
            groups[gid]['drowsy'].append(filepath)

        print(f"[*] Total Unique Subject / Group IDs Discovered: {len(groups)}")

        # Partition groups into Train and Val splits to ensure 0 subject leakage
        sorted_gids = sorted(groups.keys())
        np.random.seed(42)
        np.random.shuffle(sorted_gids)

        train_gids = set()
        val_gids = set()
        test_gids = set()

        total_alert_count = len(alert_files)
        total_drowsy_count = len(drowsy_files)

        val_alert_target = int(total_alert_count * val_split)
        val_drowsy_target = int(total_drowsy_count * val_split)
        test_alert_target = int(total_alert_count * test_split)
        test_drowsy_target = int(total_drowsy_count * test_split)

        curr_val_alert = 0
        curr_val_drowsy = 0
        curr_test_alert = 0
        curr_test_drowsy = 0

        # Assign groups to test, then validation; all remaining groups stay in train.
        for gid in sorted_gids:
            num_alert = len(groups[gid]['alert'])
            num_drowsy = len(groups[gid]['drowsy'])

            if (curr_test_alert + num_alert <= test_alert_target and curr_test_alert < test_alert_target) or \
               (curr_test_drowsy + num_drowsy <= test_drowsy_target and curr_test_drowsy < test_drowsy_target):
                test_gids.add(gid)
                curr_test_alert += num_alert
                curr_test_drowsy += num_drowsy
            elif (curr_val_alert + num_alert <= val_alert_target and curr_val_alert < val_alert_target) or \
                 (curr_val_drowsy + num_drowsy <= val_drowsy_target and curr_val_drowsy < val_drowsy_target):
                val_gids.add(gid)
                curr_val_alert += num_alert
                curr_val_drowsy += num_drowsy
            else:
                train_gids.add(gid)

        print(f"[*] Group Split Allocation:")
        print(f"    - Train Groups: {len(train_gids)} subjects/sources")
        print(f"    - Val Groups:   {len(val_gids)} subjects/sources")
        print(f"    - Test Groups:  {len(test_gids)} subjects/sources")

        train_alert_files = [f for gid in train_gids for f in groups[gid]['alert']]
        train_drowsy_files = [f for gid in train_gids for f in groups[gid]['drowsy']]
        val_alert_files = [f for gid in val_gids for f in groups[gid]['alert']]
        val_drowsy_files = [f for gid in val_gids for f in groups[gid]['drowsy']]
        test_alert_files = [f for gid in test_gids for f in groups[gid]['alert']]
        test_drowsy_files = [f for gid in test_gids for f in groups[gid]['drowsy']]

        np.random.seed(42)
        np.random.shuffle(train_alert_files)
        np.random.shuffle(train_drowsy_files)
        np.random.shuffle(val_alert_files)
        np.random.shuffle(val_drowsy_files)

        # Enforce strict 50/50 physical class balancing to eliminate training bias
        if max_samples_per_class and max_samples_per_class > 0:
            train_max = int(max_samples_per_class * (1 - val_split))
            val_max = int(max_samples_per_class * val_split)
            test_max = int(max_samples_per_class * test_split)

            train_alert_files = train_alert_files[:train_max]
            train_drowsy_files = train_drowsy_files[:train_max]
            val_alert_files = val_alert_files[:val_max]
            val_drowsy_files = val_drowsy_files[:val_max]
            test_alert_files = test_alert_files[:test_max]
            test_drowsy_files = test_drowsy_files[:test_max]
        else:
            # Physical 50/50 balancing on full dataset
            train_balanced_count = min(len(train_alert_files), len(train_drowsy_files))
            val_balanced_count = min(len(val_alert_files), len(val_drowsy_files))

            train_alert_files = train_alert_files[:train_balanced_count]
            train_drowsy_files = train_drowsy_files[:train_balanced_count]
            val_alert_files = val_alert_files[:val_balanced_count]
            val_drowsy_files = val_drowsy_files[:val_balanced_count]
            test_balanced_count = min(len(test_alert_files), len(test_drowsy_files))
            test_alert_files = test_alert_files[:test_balanced_count]
            test_drowsy_files = test_drowsy_files[:test_balanced_count]

        print(f"[*] Physical 50/50 Balanced Split Allocation:")
        print(f"    - Train Split (50/50): {len(train_alert_files):,} Alert / {len(train_drowsy_files):,} Drowsy ({len(train_alert_files)*2:,} total)")
        print(f"    - Val Split   (50/50): {len(val_alert_files):,} Alert / {len(val_drowsy_files):,} Drowsy ({len(val_alert_files)*2:,} total)")

        train_items = [(f, "0_alert") for f in train_alert_files] + [(f, "1_drowsy") for f in train_drowsy_files]
        val_items = [(f, "0_alert") for f in val_alert_files] + [(f, "1_drowsy") for f in val_drowsy_files]
        test_items = [(f, "0_alert") for f in test_alert_files] + [(f, "1_drowsy") for f in test_drowsy_files]

        for split_name, item_list in [("train", train_items), ("val", val_items), ("test", test_items)]:
            dest_base = {"train": train_dir, "val": val_dir, "test": test_dir}[split_name]
            print(f"[*] Processing {split_name} split ({len(item_list)} files)...")

            for filepath, label_str in tqdm(item_list):
                ext = filepath.suffix.lower()
                gid = extract_group_id(filepath)
                prefix = f"{gid}_{filepath.parent.name}"

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

