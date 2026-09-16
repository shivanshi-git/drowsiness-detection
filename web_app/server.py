import os
import io
import time
import json
import asyncio
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException   
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import torch

from web_app.stream_manager import StreamManager, AVAILABLE_MODELS


# Initialize FastAPI app
app = FastAPI(
    title="Driver Guardian API",
    description="Real-Time SOTA Low-Light Driver Drowsiness Detection & Explainable AI (XAI) Suite",
    version="1.0.0"
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Instantiate Global Stream Manager
manager = StreamManager()


# Request Schemas
class FramePayload(BaseModel):
    image: str
    request_xai: Optional[bool] = False

class ScenarioPayload(BaseModel):
    scenario: str

class SwitchModelPayload(BaseModel):
    model_id: str


@app.get("/", response_class=FileResponse)
async def serve_index():
    """Serves the main Driver Guardian Cockpit Dashboard."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_path)


@app.get("/favicon.ico", response_class=FileResponse)
async def serve_favicon():
    """Serves the cockpit favicon."""
    fav_path = os.path.join(STATIC_DIR, "favicon.svg")
    return FileResponse(fav_path, media_type="image/svg+xml")


@app.get("/api/status")
async def get_system_status():
    """Returns real-time GPU/CPU status, active model, and engine telemetry."""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "CPU (Standard)"
    
    return {
        "status": "online",
        "device": manager.device,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "active_model_id": manager.active_model_id,
        "active_model_name": AVAILABLE_MODELS.get(manager.active_model_id, {}).get("name", "Unknown"),
        "current_fps": round(manager.current_fps, 1),
        "total_incidents": len(manager.incident_log),
        "buffer_size": len(manager.frame_buffer),
        "buffer_target": manager.seq_len
    }


@app.get("/api/models")
async def list_models():
    """Lists all available deep learning backbones and their checkpoint status."""
    models_list = []
    for m_id, info in AVAILABLE_MODELS.items():
        models_list.append({
            "id": m_id,
            "name": info["name"],
            "type": info["type"],
            "has_checkpoint": os.path.exists(info["checkpoint"]),
            "is_active": (m_id == manager.active_model_id)
        })
    return {"models": models_list}


@app.post("/api/models/switch")
async def switch_model(payload: SwitchModelPayload):
    """Hot-swaps model architecture and weights."""
    success = manager.load_model(payload.model_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to load requested model.")
    return {
        "status": "success",
        "active_model_id": manager.active_model_id,
        "active_model_name": AVAILABLE_MODELS[manager.active_model_id]["name"]
    }


@app.post("/api/process_frame")
async def process_frame(payload: FramePayload):
    """
    Ingests live base64 frame from browser webcam, evaluates sequence model,
    and returns real-time fatigue score, alarm tier, PERCLOS, and landmarks.
    """
    try:
        frame_bgr = manager.base64_to_frame(payload.image)
        if frame_bgr is None:
            raise HTTPException(status_code=400, detail="Corrupt frame format.")
            
        metrics = manager.process_frame(frame_bgr, generate_xai=payload.request_xai)
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/simulation/set_scenario")
async def set_simulation_scenario(payload: ScenarioPayload):
    """Sets active preset demonstration scenario."""
    valid = ["normal", "microsleep", "yawning", "slow_blink", "low_light"]
    if payload.scenario not in valid:
        raise HTTPException(status_code=400, detail=f"Invalid scenario. Valid: {valid}")
    manager.sim_scenario = payload.scenario
    return {"status": "success", "scenario": manager.sim_scenario}


@app.get("/api/simulation/frame")
async def get_simulation_frame(request_xai: bool = False):
    """
    Generates a single frame from the current simulated scenario,
    runs inference, and returns both image and telemetry.
    """
    frame_bgr = manager.generate_simulated_frame()
    metrics = manager.process_frame(frame_bgr, generate_xai=request_xai)
    frame_b64 = manager.frame_to_base64(frame_bgr)
    return {
        "image": f"data:image/jpeg;base64,{frame_b64}",
        "metrics": metrics
    }


def mjpeg_frame_generator():
    """Streams MJPEG frames for video feed."""
    while True:
        frame = manager.generate_simulated_frame()
        # Overlay HUD on streaming feed
        metrics = manager.last_metrics
        hud_color = (0, 255, 0)
        if metrics["alarm_level"] == 3:
            hud_color = (0, 0, 255)
        elif metrics["alarm_level"] == 2:
            hud_color = (0, 140, 255)
        elif metrics["alarm_level"] == 1:
            hud_color = (0, 255, 255)

        # Draw status overlay
        cv2.rectangle(frame, (0, 0), (640, 45), (15, 15, 20), -1)
        cv2.putText(frame, f"STATUS: {metrics['status_text']}", (15, 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, hud_color, 2)
        cv2.putText(frame, f"FATIGUE: {metrics['smoothed_fatigue']*100:.1f}%", (460, 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.60, hud_color, 2)

        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.04)  # ~25 fps


@app.get("/api/stream_mjpeg")
async def stream_mjpeg():
    """Streams live MJPEG video feed for direct browser consumption."""
    return StreamingResponse(
        mjpeg_frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/xai/latest")
async def get_latest_xai():
    """Returns latest Grad-CAM heatmaps, LLFormer comparison, and diagnostic receipt."""
    if manager.last_xai_data is None:
        # Generate on current state if none exists yet
        dummy_frame = manager.generate_simulated_frame()
        manager.process_frame(dummy_frame, generate_xai=True)
        
    return {"xai": manager.last_xai_data}


@app.post("/api/diagnose_snapshot")
async def diagnose_snapshot(payload: Optional[FramePayload] = None):
    """
    Triggers an on-demand complete XAI diagnosis on the current or provided frame.
    Generates Grad-CAM, SHAP, temporal confidence curve, and Safety Receipt Card.
    """
    if payload and payload.image:
        frame_bgr = manager.base64_to_frame(payload.image)
    else:
        frame_bgr = manager.frame_buffer[-1] if manager.frame_buffer else manager.generate_simulated_frame()

    # Process and force XAI generation
    metrics = manager.process_frame(frame_bgr, generate_xai=True)
    return {
        "status": "success",
        "metrics": metrics,
        "xai": manager.last_xai_data
    }


@app.get("/api/incidents")
async def get_incidents():
    """Returns trip safety alerts and incident log."""
    return {"incidents": list(manager.incident_log)}


@app.post("/api/reset_yawns")
async def reset_yawn_counter():
    """Resets the session yawn counter (e.g. at start of a new trip)."""
    manager.alarm_system.reset_yawn_count()
    return {"status": "success", "yawn_count": 0}


@app.get("/api/telemetry")
async def get_telemetry():
    """Returns rolling fatigue and PERCLOS telemetry points for charts."""
    return {"telemetry": list(manager.telemetry_history)}


@app.post("/api/upload_media")
async def upload_media(file: UploadFile = File(...)):
    """
    Uploads a driving video or photo for sequence analysis and XAI diagnosis.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image or unsupported video container.")

    # Populate sequence buffer with the uploaded image and evaluate
    for _ in range(manager.seq_len):
        manager.process_frame(frame, generate_xai=False)
    metrics = manager.process_frame(frame, generate_xai=True)

    return {
        "status": "success",
        "filename": file.filename,
        "metrics": metrics,
        "xai": manager.last_xai_data
    }


@app.get("/api/export_report")
async def export_session_report():
    """Exports session telemetry summary and safety audit report."""
    hist = list(manager.telemetry_history)
    avg_fatigue = np.mean([h["fatigue"] for h in hist]) if hist else 0.0
    max_fatigue = max([h["fatigue"] for h in hist]) if hist else 0.0
    total_events = len(manager.incident_log)
    
    report = {
        "report_id": f"DG-{int(time.time())}",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_used": AVAILABLE_MODELS.get(manager.active_model_id, {}).get("name"),
        "summary": {
            "average_fatigue_pct": round(float(avg_fatigue), 2),
            "peak_fatigue_pct": round(float(max_fatigue), 2),
            "safety_events_count": total_events,
            "overall_status": "HIGH RISK" if max_fatigue > 80 else ("MODERATE" if max_fatigue > 50 else "SAFE")
        },
        "incidents": list(manager.incident_log)
    }
    return JSONResponse(content=report)
