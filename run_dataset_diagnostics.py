import os
import glob
import re
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

def extract_group_and_source(filepath):
    """
    Extracts source dataset name and subject/group identifier from filepath.
    """
    path_str = str(filepath).lower()
    filename = Path(filepath).name.lower()

    # Determine Source Dataset
    if 'mrl' in path_str or filename.startswith('s0'):
        source = 'MRL'
    elif 'nthu' in path_str or re.match(r'^\d{3}_', filename):
        source = 'NTHU'
    elif 'uta' in path_str or 'rldd' in path_str:
        source = 'UTA'
    elif 'kaggle' in path_str or 'yawdd' in path_str:
        source = 'Kaggle'
    else:
        source = 'Other/Unknown'

    # Pattern 1: NTHU-DDD (e.g., 001_glasses_...)
    nthu_match = re.match(r'^(\d{3})_', filename)
    if nthu_match:
        subject_id = f"nthu_{nthu_match.group(1)}"
    # Pattern 2: MRL Eye Dataset (e.g., s0013_...)
    elif re.search(r'(s\d{3,4})', filename):
        subject_id = re.search(r'(s\d{3,4})', filename).group(0)
    # Pattern 3: Generic Subject ID
    elif re.search(r'(subject[_\-]?\d+|participant[_\-]?\d+|sub[_\-]?\d+)', path_str):
        subject_id = re.search(r'(subject[_\-]?\d+|participant[_\-]?\d+|sub[_\-]?\d+)', path_str).group(0)
    else:
        subject_id = f"{source}_default"

    return source, subject_id

def analyze_raw_dataset(raw_dirs):
    """
    Analyzes raw dataset files for source distribution, label mapping, and subject counts.
    """
    if isinstance(raw_dirs, str):
        raw_dirs = [raw_dirs]

    print("=" * 70)
    print(" RAW DATASET DIAGNOSTIC AUDIT")
    print("=" * 70)

    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi'}
    media_files = []

    for raw_dir in raw_dirs:
        if not os.path.exists(raw_dir):
            print(f"[!] Warning: Directory '{raw_dir}' not found.")
            continue
        for p in Path(raw_dir).rglob("*"):
            if p.suffix.lower() in image_exts:
                media_files.append(p)

    print(f"[*] Total Media Files Found: {len(media_files)}\n")

    source_counts = Counter()
    source_label_counts = defaultdict(lambda: {'alert': 0, 'drowsy': 0, 'yawn': 0, 'eye_closed': 0, 'unclassified': 0})
    source_subjects = defaultdict(set)

    for filepath in media_files:
        source, subject_id = extract_group_and_source(filepath)
        source_counts[source] += 1
        source_subjects[source].add(subject_id)

        path_str_lower = str(filepath).lower()

        # Categorize detailed original label signals
        if 'yawn' in path_str_lower:
            source_label_counts[source]['yawn'] += 1
        elif 'closed' in path_str_lower and source == 'MRL':
            source_label_counts[source]['eye_closed'] += 1
        elif any(k in path_str_lower for k in ['no_yawn', 'open', 'awake', 'notdrowsy', 'active', 'normal', 'alert']):
            source_label_counts[source]['alert'] += 1
        elif any(k in path_str_lower for k in ['drowsy', 'sleepy', 'fatigue', 'sleep', 'micro_sleep', 'heavy']):
            source_label_counts[source]['drowsy'] += 1
        else:
            source_label_counts[source]['unclassified'] += 1

    print(f"{'Source':<12} | {'Total Files':<12} | {'Unique Subjects':<16} | {'Alert/Open':<12} | {'Drowsy/Closed':<14} | {'Yawn':<8}")
    print("-" * 80)

    for source in sorted(source_counts.keys()):
        total = source_counts[source]
        num_subjs = len(source_subjects[source])
        alerts = source_label_counts[source]['alert']
        drowsy = source_label_counts[source]['drowsy'] + source_label_counts[source]['eye_closed']
        yawns = source_label_counts[source]['yawn']
        print(f"{source:<12} | {total:<12} | {num_subjs:<16} | {alerts:<12} | {drowsy:<14} | {yawns:<8}")

    print("=" * 80)
    print("\nCRITICAL DIAGNOSTIC INSIGHTS:")
    print(" 1. MRL provides EYE-STATE annotations (Open vs Closed), NOT driver drowsiness.")
    print(" 2. Datasets with YAWNING (Kaggle/NTHU) lose their mouth signal when cropped to 128x128 eyes.")
    print(" 3. Mixing MRL eye-closed images with Yawn-labeled eye crops creates severe label contradiction.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Raw Dataset Diagnostic Audit")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to raw dataset directory")
    args = parser.parse_args()

    analyze_raw_dataset(args.data_dir)
