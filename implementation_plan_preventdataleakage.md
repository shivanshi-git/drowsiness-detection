# Implementation Plan: Group-Aware Dataset Splitting & Leakage Prevention

## Problem Description
In the current `data/preprocess_mixed_data.py`, dataset files are randomly shuffled (`np.random.shuffle(all_media)`) at the individual file/frame level before processing. When source datasets (such as NTHU-DDD, UTA-RLDD, or MRL) contain thousands of frame images or multiple video clips per subject:
- Consecutive frames (e.g. `frame_240.jpg` and `frame_250.jpg`) or clips of the same subject get split across `train` and `val`.
- Crops derived from the same subject land in both splits, causing **subject identity leakage**.
- The validation score becomes artificially inflated because the model memorizes driver-specific features (lighting, eye shape, skin tone) instead of learning general drowsiness cues.

## Proposed Changes

### 1. Refactor `data/preprocess_mixed_data.py`

#### [MODIFY] [preprocess_mixed_data.py](file:///d:/drowsiness%20detection/data/preprocess_mixed_data.py)

- **Subject / Group ID Extraction**: Add a function `extract_group_id(filepath)` that extracts the subject/group identifier from folder structure or filename:
  - NTHU-DDD: Uses subject ID folder (e.g., `001`, `002`, `003`, etc.).
  - UTA-RLDD: Uses participant folder / clip prefix (e.g., `1`, `2`, `3`).
  - MRL Eye Dataset: Uses subject ID prefix (e.g., `s0001`, `s0002`).
  - Fallback: Uses the video file stem or parent directory name.
- **Group-Aware Train/Val Split**:
  - Collect all files into groups `dict[group_id] -> list[filepaths]`.
  - Perform a Group Stratified Split (or GroupKFold/GroupShuffleSplit) at the `group_id` level.
  - Ensures **100% of all frames/videos from any single subject stay strictly within `train` or `val`**, with zero overlap.
- **Balanced Crop Generation**:
  - Sample frames and extract MediaPipe eye crops after the subject-isolated splits are formed.
  - Maintain class balance (`0_alert` vs `1_drowsy`) across the resulting train and val splits.

---

### 2. Validation & Audit

#### [MODIFY] [audit_leakage.py](file:///d:/drowsiness%20detection/audit_leakage.py)
- Ensure subject-parsing regex covers NTHU, UTA, MRL, and Kaggle directory naming conventions.
- Provide detailed reporting on group distribution across splits.

## Verification Plan

### Automated Verification
1. Run `python data/preprocess_mixed_data.py --raw_dirs archive archive(1) archive(2) archive(3) --out_dir processed_dataset`
2. Run `python audit_leakage.py processed_dataset` to verify:
   - `Overlapping Source Files/Videos: 0`
   - `Overlapping Subjects in both Train & Val: 0`
   - `Val Crops from Train Subjects: 0 (0.00%)`

### Manual Verification
- Review the generated group distribution log during preprocessing to confirm train/val subject ratios match the target 80/20 split.
