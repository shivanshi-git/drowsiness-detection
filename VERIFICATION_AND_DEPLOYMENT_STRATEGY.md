# ⏱️ Verification & Deployment Engineering Strategy

> **Project:** Driver Drowsiness Detection System with Explainable AI (XAI)  
> **Document Purpose:** Technical strategy to establish Leave-One-Dataset-Out (LODO) generalization testing, strictly subject-disjoint test evaluation, temporal PERCLOS state machine buffering, and real-time CPU deployment allocation.

---

## 📌 Executive Summary

Building a reliable driver drowsiness detection system requires bridging two major gaps:

1. **The Verification Gap:** Single-epoch sanity checks fail to prove scientific validity. True real-world readiness requires **Subject-Disjoint Held-Out Testing** and **Leave-One-Dataset-Out (LODO) Cross-Domain Evaluation** to prove the system works on unseen drivers and environments.
2. **The Deployment Gap:** Frame-level CNN models output instantaneous predictions, whereas true drowsiness metrics (like PERCLOS) require multi-frame temporal aggregation. Furthermore, heavy benchmark models (ResNet50, VGG16) cannot maintain 30+ FPS on CPU webcams without explicit runtime allocation.

This document establishes a **Rigorous 3-Split Verification Protocol**, defines a **Temporal PERCLOS Buffer State Machine**, and categorizes models into **Production Edge vs. Benchmark Tiers**.

---

## 🚨 Diagnosis of Verification & Deployment Gaps

### 1. Verification Protocol Weakness
* **The Pitfall:** Testing only on an 80/20 train/val split where data comes from the same dataset archives.
* **The Reality:** A model trained and validated on NTHU frames can easily memorize NTHU driving simulator lighting. It will fail when deployed in an actual car or on a different dataset like UTA-RLDD.
* **The Requirement:** We must perform **Leave-One-Dataset-Out (LODO) Cross-Domain Evaluation** (e.g. train on NTHU + MRL + Kaggle, test strictly on UTA-RLDD).

### 2. Temporal Metric Mismatch (PERCLOS vs Single-Frame CNN)
* **The Pitfall:** Expecting a single-frame CNN to measure PERCLOS (Percentage of Eye Closure Over Time).
* **The Reality:** A quick blink lasts ~100–400ms (3–12 frames at 30 FPS) and is completely normal. An instantaneous "drowsy" frame prediction during a normal blink should **not** trigger an emergency alarm.
* **The Requirement:** Single-frame CNN outputs must feed into a **Temporal Sliding Window Buffer** (60 frames / 2 seconds) to compute PERCLOS and apply hysteresis thresholding.

### 3. Real-Time CPU FPS Constraints
* **The Pitfall:** Promising live webcam monitoring using ResNet50 or VGG16 on CPU.
* **The Reality:** ResNet50 achieves ~15–25 FPS on typical CPU hardware, causing frame drops and video lag. 
* **The Requirement:** We must explicitly decouple models into **Production Edge Targets** (Custom CNN / MobileNetV3 at 60+ FPS) and **Offline Research Benchmarks** (ResNet50 / ViT).

---

## 🛠️ The Verification & Deployment Framework

### 1. Subject-Disjoint 3-Split Protocol (Train / Val / Held-Out Test)

The dataset is partitioned into three strictly **Subject-Disjoint** sets:

```text
Full Dataset (100%)
├── Train Set (70%)         --> Model Weight Updates (Gradient Descent)
├── Validation Set (15%)    --> Hyperparameter Tuning & Early Stopping
└── Held-Out Test Set (15%) --> Touched EXACTLY ONCE at final report submission
```

*Rule:* Zero subject overlap across Train, Val, and Held-Out Test sets.

---

### 2. Leave-One-Dataset-Out (LODO) Generalization Protocol

To evaluate true out-of-domain transferability, we execute 3 cross-domain benchmark runs:

| LODO Experiment | Training Datasets | Held-Out Test Dataset | Evaluates |
|---|---|---|---|
| **Run A: UTA Generalization** | NTHU-DDD + MRL + Kaggle | **UTA-RLDD** (100% Unseen) | Real-life multi-stage drowsiness transfer |
| **Run B: NTHU Generalization** | UTA-RLDD + MRL + Kaggle | **NTHU-DDD** (100% Unseen) | Driving simulator camera angle transfer |
| **Run C: MRL Generalization** | NTHU-DDD + UTA-RLDD + Kaggle | **MRL Eye Dataset** (100% Unseen) | IR / tight eye crop transfer |

---

### 3. Temporal PERCLOS Sliding Window State Machine

PERCLOS is defined as the proportion of time the eyes are closed (>= 80% closed) over a specific time window:

PERCLOS = (N_closed / N_total) * 100%

#### Temporal Buffer Architecture
- **Window Size (N_total):** 60 frames (~2.0 seconds at 30 FPS).
- **Update Rule:** FIFO (First-In, First-Out) rolling queue.
- **Alert States:**
  - `ALERT (Normal):` PERCLOS < 40%
  - `WARNING (Fatigue):` 40% <= PERCLOS < 70%
  - `DANGER (Drowsy Alarm):` PERCLOS >= 70% sustained for > 1.5 seconds.

---

### 4. Model Runtime Allocation (Production vs. Benchmark Tiers)

| Model Architecture | Target Device | Expected CPU FPS | Target Role | Deployment Status |
|---|---|---|---|---|
| **Custom CNN** | CPU / Web / Edge | **120+ FPS** | Live Webcam Dashboard Engine | 🚀 **Production Primary** |
| **MobileNetV3-Small** | CPU / Mobile | **85+ FPS** | Production Mobile / Embedded Engine | 🚀 **Production Primary** |
| **EfficientNet-B0** | CPU / GPU | **45+ FPS** | High-Accuracy Edge Option | ⚠️ Secondary Edge |
| **ResNet50** | GPU Only | **20 FPS (CPU)** | Research Baseline & Feature Benchmark | 🔬 **Benchmark Only** |
| **ViT-Tiny** | GPU Only | **15 FPS (CPU)** | Transformer Attention Benchmark | 🔬 **Benchmark Only** |

---

## 💻 Code Implementation

### 1. Temporal PERCLOS Buffer State Machine (`utils/temporal_buffer.py`)

```python
from collections import deque
import time

class TemporalPERCLOSBuffer:
    """
    Rolling 60-frame temporal queue to calculate PERCLOS 
    and maintain hysteresis alert state machine.
    """
    def __init__(self, window_size: int = 60, warning_thresh: float = 0.40, danger_thresh: float = 0.70):
        self.window_size = window_size
        self.warning_thresh = warning_thresh
        self.danger_thresh = danger_thresh
        
        self.buffer = deque(maxlen=window_size)
        self.state = "ALERT"
        self.drowsy_start_time = None

    def update(self, is_drowsy_frame: bool) -> dict:
        """
        Pushes new frame prediction (1 = Drowsy/Closed, 0 = Alert/Open) 
        and updates state machine.
        """
        self.buffer.append(1 if is_drowsy_frame else 0)
        
        # Calculate current PERCLOS ratio
        perclos = sum(self.buffer) / float(len(self.buffer))
        
        # State Machine Transitions
        if perclos >= self.danger_thresh:
            if self.state != "DANGER":
                self.drowsy_start_time = time.time()
            self.state = "DANGER"
        elif perclos >= self.warning_thresh:
            self.state = "WARNING"
            self.drowsy_start_time = None
        else:
            self.state = "ALERT"
            self.drowsy_start_time = None

        return {
            "perclos": round(perclos * 100, 2),
            "state": self.state,
            "buffer_fill": len(self.buffer),
            "drowsy_duration_sec": round(time.time() - self.drowsy_start_time, 2) if self.drowsy_start_time else 0.0
        }
```

---

### 2. Leave-One-Dataset-Out (LODO) Splitter (`data/lodo_splitter.py`)

```python
from pathlib import Path
from typing import List, Tuple

def get_lodo_splits(all_files: List[Path], held_out_dataset_name: str) -> Tuple[List[Path], List[Path]]:
    """
    Splits files such that ALL files from held_out_dataset_name 
    are isolated exclusively into the test set.
    """
    train_files = []
    test_files = []
    
    held_out_lower = held_out_dataset_name.lower()
    
    for f in all_files:
        path_str = str(f).lower()
        if held_out_lower in path_str:
            test_files.append(f)
        else:
            train_files.append(f)
            
    print(f"[*] LODO Split for Held-Out Target '{held_out_dataset_name}':")
    print(f"    - Training Files (Other Datasets): {len(train_files):,}")
    print(f"    - Test Files ({held_out_dataset_name} Only): {len(test_files):,}")
    
    return train_files, test_files
```

---

## 📈 Summary & Action Plan

1. **Enforce 3-Split Isolation:** Train (70%), Val (15%), and Held-Out Test (15%) are strictly subject-disjoint. Held-Out Test is touched once at project completion.
2. **Execute LODO Cross-Domain Benchmarks:** Train on NTHU+MRL+Kaggle, test on UTA-RLDD to prove real-world out-of-domain transferability.
3. **Bridge Single-Frame to Temporal:** Implement `TemporalPERCLOSBuffer` (60-frame queue) in `app.py` to prevent blink-induced false alarms.
4. **Target Production Hardware:** Designate **Custom CNN** and **MobileNetV3-Small** as the 60+ FPS CPU live webcam engines, reserving ResNet50 for offline benchmarks.
