"""
WeldGuard-AI: Factory Surveillance & Telemetry Backend API
Version 3.0: Comprehensive In-Situ AI Defect Prevention, MJPEG Streaming & Web SCADA Console
Serves:
  1. Operator Web SCADA Console UI (GET /, GET /dashboard)
  2. Live In-Situ Surveillance Video Feed (GET /api/v2/surveillance/stream)
  3. Live Telemetry & Weld Pool Geometry (GET /api/v2/surveillance/telemetry)
  4. Manual E-STOP Trip Reset (POST /api/v2/surveillance/reset_estop)
  5. Multi-Dataset Industrial Benchmark Suite (GET /api/v2/benchmark/report)
  6. Factory Health & GPU Acceleration Diagnostics (GET /api/health)
  7. Crane Gallery 28-stage fabrication database (GET /api/v2/stages)
  8. Robotic & Manual benchmark telemetry (GET /api/v2/telemetry/robotic, manual)
  9. Real-Time Factory Incidents Ledger (GET /api/v2/surveillance/incidents)
 10. SCADA / MES Webhook Ingress (POST /api/v2/surveillance/webhook)
"""

import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
MODELS_DIR = ROOT_DIR / "models"
ALERTS_DIR = DATA_DIR / "alerts"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

from src.realtime_surveillance import RealTimeSurveillanceEngine, AlertSeverity

app = FastAPI(
    title="WeldGuard-AI Surveillance Platform",
    description="In-Situ AI Welding Surveillance, Telemetry & Real-Time Alert Engine",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file mounts
if DATA_DIR.exists():
    app.mount("/static/data", StaticFiles(directory=str(DATA_DIR)), name="data")

if OUTPUT_DIR.exists():
    app.mount("/static/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

VIDEO_SOURCES = {
    "panasonic_nominal": DATA_DIR / "external_videos" / "panasonic_burn_through_after.mp4",
    "panasonic_defect": DATA_DIR / "external_videos" / "panasonic_burn_through_before.mp4",
    "xiris_hdr": DATA_DIR / "external_videos" / "xiris_burn_through.mp4",
    "kml_robotic": DATA_DIR / "external_videos" / "kml_robotic_inspection.mp4",
}

# Global shared telemetry cache & active streaming engines
ACTIVE_ENGINES: Dict[str, RealTimeSurveillanceEngine] = {}
LATEST_TELEMETRY: Dict[str, Any] = {
    "status": "ONLINE",
    "timestamp_sec": 0.0,
    "is_arc_on": False,
    "stressor_level": 0.0,
    "is_halted": False,
    "halt_reason": "",
    "status_tier": "NOMINAL",
    "differential_status": "STANDBY_NOMINAL",
    "differential_energy": 0.0,
    "pool_width": 0.0,
    "mean_width": 0.0,
    "pool_length": 0.0,
    "aspect_ratio": 0.0,
    "spatter_count": 0,
    "reflections_suppressed": 0,
    "alerts_dispatched": 0,
    "recent_alert_message": "PROCESS NOMINAL - WELDING STABLE"
}


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    """Serves the Operator Web SCADA Console UI."""
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Web console template not found")
    with open(index_file, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.get("/api/v2/surveillance/stream")
def stream_surveillance_feed(video: str = "panasonic_nominal"):
    """
    Streams live annotated factory surveillance feed with T0 Anchor HUD via MJPEG.
    Updates real-time telemetry dictionary for dashboard consumption.
    """
    video_path = VIDEO_SOURCES.get(video, VIDEO_SOURCES.get("panasonic_nominal"))
    if not video_path or not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video stream '{video}' not found at {video_path}")

    model_path = MODELS_DIR / "convlstm_best.pth"
    engine = RealTimeSurveillanceEngine(
        video_source=str(video_path),
        model_path=str(model_path) if model_path.exists() else None,
        use_torch_roi=True,
        roi_size=400,
        stressor_bound=80.0,
        auto_halt=True,
        output_resolution=(1280, 720)
    )
    ACTIVE_ENGINES[video] = engine

    def frame_generator():
        for chunk in engine.stream_mjpeg(loop=True):
            if engine.latest_telemetry:
                LATEST_TELEMETRY.update(engine.latest_telemetry)
                LATEST_TELEMETRY["active_video"] = video
            yield chunk

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/v2/surveillance/telemetry")
def get_live_telemetry():
    """Returns current real-time telemetry from active surveillance engine."""
    return LATEST_TELEMETRY


@app.post("/api/v2/surveillance/reset_estop")
def reset_safety_estop():
    """Operator manual reset for tripped E-STOP safety halt."""
    for vid, engine in ACTIVE_ENGINES.items():
        if hasattr(engine, "safety_engine"):
            engine.safety_engine.is_halted = False
            engine.safety_engine.current_stressor = 0.0
            engine.safety_engine.halt_reason = ""
    LATEST_TELEMETRY["is_halted"] = False
    LATEST_TELEMETRY["stressor_level"] = 0.0
    LATEST_TELEMETRY["halt_reason"] = ""
    LATEST_TELEMETRY["status_tier"] = "NOMINAL"
    LATEST_TELEMETRY["recent_alert_message"] = "E-STOP RESET BY OPERATOR - SURVEILLANCE NOMINAL"
    return {
        "status": "RESET_SUCCESSFUL",
        "message": "E-STOP state cleared. Surveillance nominal.",
        "timestamp": time.time()
    }


@app.get("/api/v2/benchmark/report")
def get_benchmark_report():
    """Returns multi-dataset benchmark comparison results (Panasonic Nominal vs Defect vs Xiris HDR)."""
    report_file = DATA_DIR / "benchmark_comparison_report.json"
    if not report_file.exists():
        raise HTTPException(status_code=404, detail="Benchmark comparison report not found.")
    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/api/health")
def get_health():
    """System health check and GPU acceleration status."""
    gpu_ok = torch.cuda.is_available()
    return {
        "status": "ONLINE",
        "system": "WeldGuard-AI Industrial Surveillance Engine",
        "version": "3.0.0",
        "gpu_accelerated": gpu_ok,
        "device_name": torch.cuda.get_device_name(0) if gpu_ok else "CPU",
        "models_available": {
            "convlstm_best": (MODELS_DIR / "convlstm_best.pth").exists(),
            "kaggle_t4_trained": (MODELS_DIR / "convlstm_weldguard_kaggle_t4.pth").exists()
        },
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    }


@app.get("/api/v2/stages")
def get_fabrication_stages():
    """Returns the 28 fabrication stages parsed from crane_gallery_stages.xlsx."""
    excel_path = DATA_DIR / "crane_gallery_stages.xlsx"
    if not excel_path.exists():
        raise HTTPException(status_code=404, detail="crane_gallery_stages.xlsx not found")

    df = pd.read_excel(excel_path)
    stages = []
    for _, row in df.iterrows():
        stages.append({
            "seq": int(row["Seq"]),
            "section": str(row["Section"]),
            "stage_activity": str(row["Stage / Activity"]),
            "cycle_time_min": float(row["Total Time (Cycle Time, min)"]),
            "takt_time_min": float(row["Takt Time (min)"]),
            "variance_vs_takt": float(row["Variance vs Takt"]),
            "tracking_status": str(row["Tracking Status"])
        })

    return {
        "total_stages": len(stages),
        "standard_takt_min": 227.0,
        "stages": stages
    }


@app.get("/api/v2/telemetry/robotic")
def get_robotic_telemetry():
    """Returns telemetry benchmark reports for robotic welding seams."""
    seams = ["video_20260908_170143", "video_20260908_171604"]
    results = {}

    for key in seams:
        json_file = DATA_DIR / f"telemetry_report_{key}.json"
        if json_file.exists():
            with open(json_file, "r", encoding="utf-8") as f:
                results[key] = json.load(f)
        else:
            results[key] = {"error": f"Report {json_file.name} not found"}

    return {
        "benchmark_seams_count": len(seams),
        "telemetry": results
    }


@app.get("/api/v2/telemetry/manual")
def get_manual_telemetry():
    """Returns manual fabrication telemetry (IMG_3601.MOV)."""
    json_file = DATA_DIR / "manual_telemetry_report_IMG_3601.json"
    if not json_file.exists():
        raise HTTPException(status_code=404, detail="Manual telemetry report not found")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


@app.get("/api/v2/surveillance/incidents")
def get_surveillance_incidents(limit: int = 50):
    """Returns recent real-time factory alerts and incidents from surveillance."""
    log_file = ALERTS_DIR / "factory_incidents_log.jsonl"
    if not log_file.exists():
        return {"total_incidents": 0, "incidents": []}

    incidents = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    incidents.append(json.loads(line))
                except Exception:
                    pass

    return {
        "total_incidents": len(incidents),
        "recent_incidents": incidents[-limit:]
    }


@app.post("/api/v2/surveillance/webhook")
async def receive_surveillance_webhook(request: Request):
    """Ingests factory SCADA / MES alert events."""
    payload = await request.json()
    ALERTS_DIR.mkdir(parents=True, exist_ok=True)
    webhook_log = ALERTS_DIR / "webhook_inbox.jsonl"
    with open(webhook_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    return {"status": "RECEIVED", "timestamp": time.time()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
