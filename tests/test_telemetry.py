"""
AutoTwin-AI v2.0: Telemetry & Backend Unit Tests
Verifies:
  1. Factory Stages Database parsing & takt time compliance
  2. Telemetry JSON schemas (Robotic & Manual)
  3. FastAPI backend endpoints (/api/health, /api/v2/stages, /api/v2/telemetry/robotic)
"""

import json
from pathlib import Path
import pytest
import pandas as pd
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT_DIR))

from backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_backend_health(client):
    """Ensure FastAPI health endpoint reports ONLINE."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ONLINE"
    assert "server_time" in data


def test_crane_stages_database():
    """Verify crane_gallery_stages.xlsx exists and contains 28 stages with valid takt times."""
    excel_path = ROOT_DIR / "data" / "crane_gallery_stages.xlsx"
    assert excel_path.exists(), "crane_gallery_stages.xlsx must exist in data/"

    df = pd.read_excel(excel_path)
    assert len(df) == 28, f"Expected 28 stages, found {len(df)}"

    required_cols = [
        "Seq", "Section", "Stage / Activity", 
        "Total Time (Cycle Time, min)", "Takt Time (min)", 
        "Variance vs Takt", "Tracking Status"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # Verify standard takt time is 227 min
    assert (df["Takt Time (min)"] == 227).all(), "All stages must have standard takt time of 227 min"


def test_api_v2_stages_endpoint(client):
    """Verify /api/v2/stages returns all 28 parsed stages."""
    res = client.get("/api/v2/stages")
    assert res.status_code == 200
    data = res.json()
    assert data["total_stages"] == 28
    assert len(data["stages"]) == 28
    assert data["stages"][0]["seq"] == 1
    assert data["stages"][0]["takt_time_min"] == 227


def test_robotic_telemetry_reports():
    """Verify robotic telemetry reports exist and contain required telemetry fields."""
    video_keys = ["video_20260908_170143", "video_20260908_171604"]
    for key in video_keys:
        json_path = ROOT_DIR / "data" / f"telemetry_report_{key}.json"
        assert json_path.exists(), f"Missing telemetry report: {json_path.name}"

        with open(json_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        assert "video_name" in data
        assert "arc_telemetry" in data
        assert "spatter_telemetry" in data
        assert "stability_telemetry" in data

        # Check arc telemetry values
        arc = data["arc_telemetry"]
        assert arc["total_arc_on_sec"] > 0
        assert arc["duty_cycle_pct"] >= 99.0  # Controlled robot weld has near 100% duty cycle

        # Check spatter telemetry
        spatter = data["spatter_telemetry"]
        assert spatter["average_sparks_per_frame"] > 0
        assert spatter["peak_sparks_count"] > 0

        # Check stability telemetry
        stability = data["stability_telemetry"]
        assert 0.0 <= stability["process_stability_index_pct"] <= 100.0


def test_manual_telemetry_report():
    """Verify manual fabrication telemetry report for IMG_3601.MOV."""
    json_path = ROOT_DIR / "data" / "manual_telemetry_report_IMG_3601.json"
    assert json_path.exists(), "manual_telemetry_report_IMG_3601.json must exist in data/"

    with open(json_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    assert "operator_telemetry" in data
    assert "crane_gallery_benchmark_correlation" in data
    op = data["operator_telemetry"]
    assert op["total_active_grinding_sec"] > 0
    assert op["grinding_duty_cycle_pct"] > 0
    assert len(data["crane_gallery_benchmark_correlation"]) == 3
