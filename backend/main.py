"""
WeldGuard-AI: Factory Surveillance & Telemetry Backend API
Serves:
  1. Health & GPU acceleration diagnostics (/api/health)
  2. Crane Gallery 28-stage fabrication database (/api/v2/stages)
  3. Robotic benchmark seam telemetry (/api/v2/telemetry/robotic)
  4. Manual workstation telemetry (/api/v2/telemetry/manual)
  5. Live factory surveillance incidents & alerts (/api/v2/surveillance/incidents)
  6. Factory MES / SCADA webhook ingress (/api/v2/surveillance/webhook)
"""

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
ALERTS_DIR = DATA_DIR / "alerts"

app = FastAPI(
    title="WeldGuard-AI Surveillance Platform",
    description="In-Situ AI Welding Surveillance, Telemetry & Real-Time Alert Engine",
    version="2.0.0"
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


@app.get("/api/health")
def get_health():
    """System health check and GPU acceleration status."""
    gpu_ok = torch.cuda.is_available()
    return {
        "status": "ONLINE",
        "system": "WeldGuard-AI Industrial Surveillance Engine",
        "version": "2.0.0",
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
