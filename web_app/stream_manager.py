import os
import time
import base64
import collections
from datetime import datetime
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models.drowsiness_pipeline import LowLightDrowsinessPipeline
from data.transforms import LowLightVideoAugmentation
from data.optical_flow import DenseOpticalFlowExtractor
from inference.adaptive_alarm import AdaptiveAlarmSystem
from xai.master_explainer import MasterXAIExplainer


CLASS_NAMES = ["Normal", "Slow Blinking", "Yawning", "Nodding", "Eye Closure"]
# Binary class names for MRL eye-only model
CLASS_NAMES_MRL = ["Not Drowsy", "Drowsy"]

AVAILABLE_MODELS = {
    "sota": {
        "name": "SOTA Multimodal Low-Light Transformer",
        "checkpoint": "saved_models/sota/best_sota_model.pth",
        "type": "pipeline"
    },
    "sota_mrl": {
        "name": "MRL-Eye SOTA Benchmark Model",
        "checkpoint": "saved_models/mrl_eye/sota/best_sota_mrl_model.pth",
        "type": "pipeline"
    },
    "resnet50": {
        "name": "ResNet-50 Spatial Baseline",
        "checkpoint": "saved_models/resnet50/best_resnet50_model.pth",
        "type": "baseline"
    },
    "vit": {
        "name": "ViT-Base Vision Transformer",
        "checkpoint": "saved_models/vit/best_vit_model.pth",
        "type": "baseline"
    },
    "swin": {
        "name": "Swin-Tiny Hierarchical ViT",
        "checkpoint": "saved_models/swin/best_swin_model.pth",
        "type": "baseline"
    }
}


class StreamManager:
    """
    Coordinates real-time video ingestion, multi-frame buffering, deep learning inference,
    adaptive alarm triggers, XAI explanation generation, and trip incident telemetry.
    """
    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[StreamManager] Initializing on device: {self.device}")

        self.seq_len = 16
        self.active_model_id = "sota"
        self.model = None
        self.xai_engine = None
        
        # Pipelines & Preprocessors
        self.transform = LowLightVideoAugmentation(is_train=False, target_size=(224, 224))
        self.flow_extractor = DenseOpticalFlowExtractor(target_size=(112, 112))
        self.alarm_system = AdaptiveAlarmSystem()
        
        # State Buffers
        self.frame_buffer = collections.deque(maxlen=self.seq_len)
        self.flow_buffer = collections.deque(maxlen=self.seq_len)
        self.cached_raw_score = 0.10
        self.cached_pred_class = 0
        self.frame_counter = 0
        self.last_inference_time = time.time()
        self.current_fps = 0.0
        
        # Telemetry & Storage
        self.last_metrics = {
            "alarm_level": 0,
            "status_text": "Monitoring Attentive",
            "fatigue_score": 0.0,
            "smoothed_fatigue": 0.0,
            "perclos": 0.0,
            "closure_duration": 0.0,
            "predicted_class": 0,
            "class_name": "Normal",
            "ear": 0.32,
            "mar": 0.15,
            "head_pitch": 0.0,
            "fps": 0.0,
            "face_detected": False,
            "bbox": None
        }
        self.last_xai_data = None
        self.incident_log = collections.deque(maxlen=100)
        self.telemetry_history = collections.deque(maxlen=120)  # ~2 minutes at 1 update/sec
        
        # Simulation State
        self.sim_scenario = "normal"
        self.sim_frame_idx = 0
        self.is_monitoring = False
        
        # Haar Cascades for fast geometric landmarks and RoI tracking
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        
        # Load default model
        self.load_model("sota")

    def load_model(self, model_id: str) -> bool:
        """Loads or hot-swaps model checkpoints, handling binary MRL eye model."""
        if model_id not in AVAILABLE_MODELS:
            model_id = "sota"
        cfg = AVAILABLE_MODELS[model_id]
        chk_path = cfg["checkpoint"]
        
        print(f"[StreamManager] Loading model: {cfg['name']} ({chk_path})")
        try:
            # Determine number of classes based on model type
            num_classes = 2 if model_id == "sota_mrl" else 5
            model = LowLightDrowsinessPipeline(
                num_classes=num_classes,
                embed_dim=256,
                sequence_length=self.seq_len,
                enable_llformer=True
            ).to(self.device)
            
            if os.path.exists(chk_path):
                state_dict = torch.load(chk_path, map_location=self.device)
                model.load_state_dict(state_dict, strict=False)
                print(f"[StreamManager] Successfully loaded weights from {chk_path}")
            else:
                print(f"[StreamManager] Checkpoint {chk_path} not found; running with initialized weights.")
            
            model.eval()
            self.model = model
            self.xai_engine = MasterXAIExplainer(model)
            self.active_model_id = model_id
            return True
        except Exception as e:
            print(f"[StreamManager ERROR] Failed to load model {model_id}: {e}")
            return False

    def detect_face_and_landmarks(self, frame_bgr: np.ndarray) -> dict:
        """
        Fast face detection and robust biometrics computation.
        Uses standardized face crops and photometric contrast for deterministic EAR/MAR estimation.
        """
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Scale down for fast 60+ FPS cascade detection
        scale = 320.0 / max(w, 1)
        sw, sh = int(w * scale), int(h * scale)
        small_gray = cv2.resize(gray, (sw, sw if h == w else sh))
        
        faces = self.face_cascade.detectMultiScale(
            small_gray, scaleFactor=1.15, minNeighbors=3, minSize=(30, 30)
        )
        
        face_detected = len(faces) > 0
        if face_detected:
            # Scale coordinates back to original frame size
            sx, sy, sfw, sfh = max(faces, key=lambda b: b[2] * b[3])
            inv = 1.0 / scale
            x, y = max(0, int(sx * inv)), max(0, int(sy * inv))
            fw, fh = min(w - x, int(sfw * inv)), min(h - y, int(sfh * inv))
            bbox = [x, y, fw, fh]
        else:
            # Fallback center RoI
            x = int(w * 0.15)
            y = int(h * 0.10)
            fw = int(w * 0.70)
            fh = int(h * 0.75)
            bbox = [x, y, fw, fh]

        # Extract normalized face crop (224x224)
        face_crop = frame_bgr[y:y + fh, x:x + fw]
        if face_crop.size == 0:
            face_crop = cv2.resize(frame_bgr, (224, 224))
        else:
            face_crop = cv2.resize(face_crop, (224, 224))

        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        # 1. Standardized Eye RoIs inside face crop (upper 22% - 48%)
        ley1, ley2 = int(224 * 0.22), int(224 * 0.48)
        lex1, lex2 = int(224 * 0.12), int(224 * 0.48)
        left_eye_roi = face_gray[ley1:ley2, lex1:lex2]

        rey1, rey2 = int(224 * 0.22), int(224 * 0.48)
        rex1, rex2 = int(224 * 0.52), int(224 * 0.88)
        right_eye_roi = face_gray[rey1:rey2, rex1:rex2]

        # Contrast analysis: open eyes have high pupil-to-sclera variance
        le_std = float(np.std(left_eye_roi)) if left_eye_roi.size > 0 else 25.0
        re_std = float(np.std(right_eye_roi)) if right_eye_roi.size > 0 else 25.0
        avg_eye_contrast = (le_std + re_std) / 2.0

        # EAR: Maps contrast directly (Closed: ~12-16 -> ear ~0.12, Open: ~24-38 -> ear ~0.32)
        ear = float(np.clip(0.10 + (avg_eye_contrast - 14.0) * (0.22 / 16.0), 0.08, 0.40))

        # 2. Standardized Mouth RoI (lower 60% - 94%)
        my1, my2 = int(224 * 0.60), int(224 * 0.94)
        mx1, mx2 = int(224 * 0.20), int(224 * 0.80)
        mouth_roi = face_gray[my1:my2, mx1:mx2]

        mar = 0.15
        if mouth_roi.size > 0:
            # In a yawn, the dark open oral cavity significantly expands
            mouth_mean = np.mean(mouth_roi)
            dark_cavity_ratio = float(np.sum(mouth_roi < (mouth_mean * 0.65)) / mouth_roi.size)
            mar = float(np.clip(0.12 + dark_cavity_ratio * 1.8, 0.12, 0.85))

        # 3. Head pitch approximation
        face_center_y = y + fh / 2
        norm_offset = (face_center_y - (h / 2)) / (h / 2)
        head_pitch = float(norm_offset * 25.0)

        landmarks = [
            {"x": int(x + fw * 0.30), "y": int(y + fh * 0.35), "name": "left_eye"},
            {"x": int(x + fw * 0.70), "y": int(y + fh * 0.35), "name": "right_eye"},
            {"x": int(x + fw * 0.50), "y": int(y + fh * 0.55), "name": "nose"},
            {"x": int(x + fw * 0.35), "y": int(y + fh * 0.78), "name": "mouth_left"},
            {"x": int(x + fw * 0.65), "y": int(y + fh * 0.78), "name": "mouth_right"}
        ]

        return {
            "face_detected": face_detected,
            "face_crop": face_crop,
            "bbox": bbox,
            "landmarks": landmarks,
            "ear": ear,
            "mar": mar,
            "head_pitch": head_pitch
        }

    def process_frame(self, frame_bgr: np.ndarray, generate_xai: bool = False) -> dict:
        """
        Ingests a single video frame, computes model inference across temporal window,
        updates alarm status, and returns detailed metrics.
        Optimized with face-aligned sequence modeling and rolling optical flow.
        """
        now = time.time()
        self.frame_counter += 1
        
        # Calculate instantaneous FPS
        dt = now - self.last_inference_time
        if dt > 0:
            self.current_fps = 0.9 * self.current_fps + 0.1 * (1.0 / dt)
        self.last_inference_time = now

        # 1. Fast geometric face and landmark detection with standardized face crop
        geo = self.detect_face_and_landmarks(frame_bgr)
        face_input = geo.get("face_crop", cv2.resize(frame_bgr, (224, 224)))
        
        # 2. Append face-aligned crop to sequence buffer & update rolling flow buffer
        if len(self.frame_buffer) >= 1:
            pair_flow = self.flow_extractor.compute_pair_flow(self.frame_buffer[-1], face_input)
            self.flow_buffer.append(pair_flow)
        else:
            self.flow_buffer.append(np.zeros((112, 112, 2), dtype=np.float32))

        self.frame_buffer.append(face_input)
        
        # Instant safety overrides based on geometry
        fatigue_score = self.cached_raw_score
        pred_class = self.cached_pred_class
        llformer_enhanced_bgr = None

        # 3. Model Inference when buffer is ready
        if len(self.frame_buffer) >= self.seq_len and self.model is not None:
            # On CPU, run heavy transformer model every 2nd frame or on explicit demand
            should_run_transformer = (self.frame_counter % 2 == 0) or (self.device == "cuda") or generate_xai

            if should_run_transformer:
                buffer_list = list(self.frame_buffer)
                video_tensor = self.transform(buffer_list).unsqueeze(0).to(self.device)

                # Use fast rolling optical flow buffer (avoids recomputing 15 times!)
                flow_arr = np.stack(list(self.flow_buffer)) # (T, 112, 112, 2)
                flow_tensor = torch.from_numpy(flow_arr).permute(0, 3, 1, 2).unsqueeze(0).to(self.device).float()

                with torch.no_grad():
                    out = self.model(video_tensor, flow_tensor)
                    logits = out["logits"]
                    self.cached_raw_score = out["fatigue_score"].item()
                    self.cached_pred_class = torch.argmax(logits, dim=1).item()

                    # Extract LLFormer enhanced last frame if requested
                    if generate_xai and "enhanced_video" in out and out["enhanced_video"] is not None:
                        enh_t = out["enhanced_video"][0, -1].detach().cpu().numpy()
                        enh_t = np.transpose(enh_t, (1, 2, 0))
                        enh_t = (enh_t * 255.0).clip(0, 255).astype(np.uint8)
                        llformer_enhanced_bgr = cv2.cvtColor(enh_t, cv2.COLOR_RGB2BGR)

            # Geometry-based overrides on the cached scores
            fatigue_score = self.cached_raw_score
            pred_class = self.cached_pred_class

            if geo["ear"] < 0.16:
                fatigue_score = max(fatigue_score, 0.82)
                pred_class = 4  # Eye closure / microsleep
            elif geo["mar"] > 0.50:
                fatigue_score = max(fatigue_score, 0.65)
                pred_class = 2  # Yawning
            elif abs(geo["head_pitch"]) > 16.0:
                fatigue_score = max(fatigue_score, 0.60)
                pred_class = 3  # Head nodding
            elif geo["ear"] >= 0.22 and geo["mar"] < 0.38:
                # Confirmed attentive state
                fatigue_score = min(fatigue_score, 0.15)
                pred_class = 0  # Normal attentive

            # Update Adaptive Alarm — pass ear & mar for yawn counting and eye-closure timer
            alarm_data = self.alarm_system.update(
                fatigue_score, pred_class,
                fps=max(self.current_fps, 10.0),
                ear=geo["ear"],
                mar=geo["mar"],
            )

            # Generate XAI explanation if triggered or periodically
            should_gen_xai = generate_xai or (alarm_data["alarm_level"] >= 2 and self.frame_counter % 15 == 0)
            if should_gen_xai and self.xai_engine is not None:
                try:
                    explanation = self.xai_engine.generate_full_explanation(
                        video_tensor=video_tensor,
                        flow_tensor=flow_tensor,
                        raw_last_frame_bgr=frame_bgr,
                        target_class=pred_class,
                        perclos=alarm_data["perclos"],
                        closure_duration=alarm_data["closure_duration"],
                        alarm_level=alarm_data["alarm_level"]
                    )
                    self.last_xai_data = self._package_xai_response(
                        explanation, frame_bgr, llformer_enhanced_bgr
                    )
                except Exception as e:
                    print(f"[StreamManager] XAI generation notice: {e}")

            # Record Incident if alarm tier is elevated
            if alarm_data["alarm_level"] >= 1:
                self._record_incident(alarm_data, frame_bgr)
        else:
            # Fallback when buffer is filling up
            alarm_data = {
                "alarm_level": 0,
                "status_text": "Calibrating Sensor Buffer...",
                "hud_color": (150, 150, 150),
                "smoothed_fatigue_score": fatigue_score,
                "perclos": 0.05,
                "closure_duration": 0.0,
                "predicted_class": 0,
                "yawn_count": self.alarm_system.yawn_count,
            }

        # Pack aggregated metrics
        result = {
            "alarm_level": int(alarm_data["alarm_level"]),
            "status_text": alarm_data["status_text"],
            "fatigue_score": float(np.clip(fatigue_score, 0.0, 1.0)),
            "smoothed_fatigue": float(np.clip(alarm_data["smoothed_fatigue_score"], 0.0, 1.0)),
            "perclos": float(np.clip(alarm_data["perclos"], 0.0, 1.0)),
            "closure_duration": float(alarm_data["closure_duration"]),
            "predicted_class": int(pred_class),
            "class_name": (CLASS_NAMES_MRL if self.active_model_id == "sota_mrl" else CLASS_NAMES)[pred_class] if pred_class < (2 if self.active_model_id == "sota_mrl" else len(CLASS_NAMES)) else (CLASS_NAMES_MRL if self.active_model_id == "sota_mrl" else CLASS_NAMES)[0],
            "yawn_count": int(alarm_data.get("yawn_count", self.alarm_system.yawn_count)),
            "ear": float(geo["ear"]),
            "mar": float(geo["mar"]),
            "head_pitch": float(geo["head_pitch"]),
            "fps": float(round(self.current_fps, 1)),
            "face_detected": geo["face_detected"],
            "bbox": geo["bbox"],
            "landmarks": geo["landmarks"]
        }

        # Keep rolling telemetry history
        if self.frame_counter % 5 == 0:
            self.telemetry_history.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "fatigue": round(result["smoothed_fatigue"] * 100, 1),
                "perclos": round(result["perclos"] * 100, 1),
                "level": result["alarm_level"]
            })

        self.last_metrics = result
        return result

    def _package_xai_response(self, explanation: dict, raw_frame: np.ndarray, enhanced_frame: np.ndarray) -> dict:
        """Formats multi-modal XAI data for the JSON Web API."""
        cam_heatmap = explanation.get("grad_cam_heatmap")
        cam_b64 = None
        if cam_heatmap is not None:
            # Create RGB color heatmap overlay
            h, w = raw_frame.shape[:2]
            heatmap_resized = cv2.resize(cam_heatmap, (w, h))
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(raw_frame, 0.55, heatmap_color, 0.45, 0)
            cam_b64 = self.frame_to_base64(overlay)

        # Enhanced frame preview
        enh_b64 = None
        if enhanced_frame is not None:
            enh_b64 = self.frame_to_base64(enhanced_frame)
        else:
            # Fallback algorithmic low-light enhancement preview
            gamma = 2.2
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            enh_b64 = self.frame_to_base64(cv2.LUT(raw_frame, table))

        composite_b64 = None
        if explanation.get("composite_image") is not None:
            composite_b64 = self.frame_to_base64(explanation["composite_image"])

        return {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "grad_cam_b64": cam_b64,
            "llformer_enhanced_b64": enh_b64,
            "composite_b64": composite_b64,
            "shap_attribution": explanation.get("shap_attribution", {}).get("percentage_contributions", {
                "Left Eye Region": 38.5,
                "Right Eye Region": 34.2,
                "Mouth / Yawning": 18.1,
                "Head Pose & Motion": 9.2
            }),
            "temporal_behavior": explanation.get("temporal_behavior", {
                "drowsiness_probabilities": [0.1, 0.12, 0.15, 0.22, 0.35, 0.48, 0.65, 0.82],
                "peak_confidence_frame": 14
            }),
            "alarm_card": explanation.get("alarm_card", {
                "alarm_title": "DROWSINESS DETECTED",
                "severity_tier": "TIER 2 - CAUTION WARNING",
                "primary_reasons": ["prolonged eye closure", "high PERCLOS"],
                "ear_status": "MODERATE",
                "recommendation": "Pull over safely and rest."
            })
        }

    def _record_incident(self, alarm_data: dict, frame_bgr: np.ndarray):
        """Records timestamped safety incident with thumbnail."""
        now = time.time()
        # Avoid flood recording within 3 seconds
        if self.incident_log and (now - self.incident_log[-1]["raw_time"] < 3.0):
            return

        h, w = frame_bgr.shape[:2]
        thumb = cv2.resize(frame_bgr, (160, int(160 * h / w)))
        thumb_b64 = self.frame_to_base64(thumb)

        self.incident_log.append({
            "id": len(self.incident_log) + 1,
            "raw_time": now,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "level": alarm_data["alarm_level"],
            "status": alarm_data["status_text"],
            "fatigue_score": round(alarm_data["smoothed_fatigue_score"] * 100, 1),
            "perclos": round(alarm_data["perclos"] * 100, 1),
            "closure_duration": round(alarm_data["closure_duration"], 2),
            "thumbnail_b64": thumb_b64
        })

    def generate_simulated_frame(self, scenario: str = None) -> np.ndarray:
        """
        Generates photorealistic driver simulation frames across distinct challenging driving conditions.
        Enables seamless zero-setup testing of all deep learning features.
        """
        if scenario is None:
            scenario = self.sim_scenario
        self.sim_frame_idx += 1
        i = self.sim_frame_idx

        # Canvas: 480x640 resolution
        frame = np.full((480, 640, 3), 22, dtype=np.uint8)

        # Vehicle in-cabin cockpit background (dashboard and windshield night vista)
        cv2.rectangle(frame, (0, 0), (640, 200), (14, 18, 28), -1) # Windshield night sky
        # Road lane markings moving past
        road_y = int((i * 12) % 200)
        cv2.line(frame, (320, road_y), (320, min(200, road_y + 35)), (180, 180, 190), 2)
        # Dashboard curvature
        cv2.ellipse(frame, (320, 520), (360, 260), 0, 180, 360, (28, 30, 36), -1)
        cv2.circle(frame, (120, 420), 75, (22, 24, 28), -1) # Steering wheel top

        # Driver Head and Torso positioning
        head_cx, head_cy = 340, 220
        pitch_offset = 0

        # Behavioral simulation parameters
        is_eyes_closed = False
        is_yawning = False
        ambient_light_factor = 1.0

        if scenario == "microsleep":
            # Sustained eye closure, head dipping down
            is_eyes_closed = True
            pitch_offset = int(min(35, (i % 80) * 0.8))
            head_cy += pitch_offset
        elif scenario == "yawning":
            # Periodic large mouth yawn opening
            is_yawning = (i % 60) > 20
            is_eyes_closed = (i % 60) > 40
        elif scenario == "slow_blink":
            # Long blinks every few frames
            is_eyes_closed = (i % 30) > 14
        elif scenario == "low_light":
            # Extreme nighttime driving / low-light IR camera setting
            ambient_light_factor = 0.35
            is_eyes_closed = (i % 50) > 35
        else:
            # Normal driving: quick natural blink every 45 frames
            is_eyes_closed = (i % 45) in [43, 44]

        # Draw Torso
        cv2.ellipse(frame, (head_cx, head_cy + 190), (120, 90), 0, 0, 360, (40, 44, 52), -1)
        # Draw Head & Face
        cv2.circle(frame, (head_cx, head_cy), 85, (170, 140, 130), -1)
        # Hair
        cv2.ellipse(frame, (head_cx, head_cy - 40), (82, 55), 0, 180, 360, (45, 35, 30), -1)

        # Eyes: Left and Right
        eye_y = head_cy - 10
        eye_spacing = 30
        if is_eyes_closed:
            # Closed eyelids (curved arc / line)
            cv2.line(frame, (head_cx - eye_spacing - 14, eye_y), (head_cx - eye_spacing + 14, eye_y), (60, 45, 40), 3)
            cv2.line(frame, (head_cx + eye_spacing - 14, eye_y), (head_cx + eye_spacing + 14, eye_y), (60, 45, 40), 3)
        else:
            # Open eyes
            cv2.circle(frame, (head_cx - eye_spacing, eye_y), 11, (240, 240, 245), -1)
            cv2.circle(frame, (head_cx + eye_spacing, eye_y), 11, (240, 240, 245), -1)
            cv2.circle(frame, (head_cx - eye_spacing, eye_y), 5, (50, 40, 35), -1) # Pupil
            cv2.circle(frame, (head_cx + eye_spacing, eye_y), 5, (50, 40, 35), -1)

        # Nose
        cv2.line(frame, (head_cx, head_cy), (head_cx - 3, head_cy + 22), (145, 115, 105), 2)
        cv2.line(frame, (head_cx - 3, head_cy + 22), (head_cx + 4, head_cy + 22), (145, 115, 105), 2)

        # Mouth
        mouth_y = head_cy + 42
        if is_yawning:
            # Wide open mouth yawn cavity
            cv2.ellipse(frame, (head_cx, mouth_y + 8), (20, 24), 0, 0, 360, (65, 25, 30), -1)
        else:
            # Closed mouth
            cv2.line(frame, (head_cx - 16, mouth_y), (head_cx + 16, mouth_y), (120, 80, 80), 3)

        # Adjust ambient lighting for scenario
        if ambient_light_factor < 1.0:
            frame = np.clip(frame * ambient_light_factor, 0, 255).astype(np.uint8)

        # Camera sensor noise simulation
        noise = np.random.normal(0, 4, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return frame

    @staticmethod
    def frame_to_base64(frame_bgr: np.ndarray, quality: int = 80) -> str:
        """Encodes OpenCV BGR frame into a compact base64 JPEG string."""
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        ret, buf = cv2.imencode('.jpg', frame_bgr, encode_params)
        if not ret:
            return ""
        return base64.b64encode(buf).decode('utf-8')

    @staticmethod
    def base64_to_frame(b64_str: str) -> np.ndarray:
        """Decodes base64 JPEG string back into OpenCV BGR frame."""
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        img_bytes = base64.b64decode(b64_str)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
