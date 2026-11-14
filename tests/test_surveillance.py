"""
Unit & Integration Tests for WeldGuard-AI Real-Time Factory Surveillance Engine
Verifies:
  1. Alert Severity & Event Schema
  2. Alert Dispatcher debouncing, incident logging, and snapshot creation
  3. Optical Telemetry (Arc-On & Spatter Particle isolation)
  4. Spatiotemporal ConvLSTM Anomaly Engine
  5. End-to-End Surveillance Execution on factory video
"""

import json
from pathlib import Path
import shutil
import tempfile
import cv2
import numpy as np
import pytest
import torch

from src.realtime_surveillance import (
    AlertSeverity,
    AlertEvent,
    AlertDispatcher,
    RealTimeSurveillanceEngine,
    IndustrialSafetyTripEngine
)
from src.models.conv_lstm import SpatiotemporalConvLSTMAutoencoder


@pytest.fixture
def temp_alerts_dir():
    d = Path(tempfile.mkdtemp(prefix="weldguard_test_alerts_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_alert_event_schema():
    alert = AlertEvent(
        alert_id="ALT-TEST-001",
        timestamp_sec=12.5,
        frame_idx=375,
        severity=AlertSeverity.CRITICAL,
        alert_type="spatter_burst",
        message="Test alert message",
        metrics={"spatter_count": 65, "burst_threshold": 25},
        suggested_action="Adjust voltage trim"
    )
    d = alert.to_dict()
    assert d["alert_id"] == "ALT-TEST-001"
    assert d["severity"] == "CRITICAL"
    assert d["metrics"]["spatter_count"] == 65


def test_alert_dispatcher_logging_and_debouncing(temp_alerts_dir):
    dispatcher = AlertDispatcher(temp_alerts_dir, cooldown_sec=1.0)

    dummy_frame = np.zeros((128, 128, 3), dtype=np.uint8)
    alert1 = AlertEvent(
        alert_id="ALT-1",
        timestamp_sec=1.0,
        frame_idx=30,
        severity=AlertSeverity.WARNING,
        alert_type="spatter_burst",
        message="Burst 1",
        metrics={"spatter_count": 30},
        suggested_action="Check gas"
    )

    # First dispatch should succeed
    res1 = dispatcher.dispatch(alert1, annotated_frame=dummy_frame)
    assert res1 is True
    assert alert1.snapshot_path is not None
    assert Path(alert1.snapshot_path).exists()

    # Second immediate dispatch of same type should be debounced by cooldown
    alert2 = AlertEvent(
        alert_id="ALT-2",
        timestamp_sec=1.1,
        frame_idx=33,
        severity=AlertSeverity.WARNING,
        alert_type="spatter_burst",
        message="Burst 2",
        metrics={"spatter_count": 32},
        suggested_action="Check gas"
    )
    res2 = dispatcher.dispatch(alert2)
    assert res2 is False

    # Verify log file exists and contains the first event
    assert dispatcher.log_file.exists()
    with open(dispatcher.log_file, "r") as f:
        lines = f.readlines()
    assert len(lines) == 1
    logged_event = json.loads(lines[0])
    assert logged_event["alert_id"] == "ALT-1"


def test_convlstm_model_forward():
    model = SpatiotemporalConvLSTMAutoencoder(in_channels=3, latent_dim=64)
    model.eval()

    # Batch=1, T=8 frames, 3 channels, 128x128
    dummy_seq = torch.randn(1, 8, 3, 128, 128)
    with torch.no_grad():
        pred_next = model(dummy_seq)

    assert pred_next.shape == (1, 3, 128, 128)
    # Sigmoid output in [0, 1]
    assert pred_next.min() >= 0.0
    assert pred_next.max() <= 1.0


def test_end_to_end_surveillance_short_run(temp_alerts_dir):
    project_root = Path(__file__).resolve().parent.parent
    video_path = project_root / "data" / "video_20260908_170143.mp4"

    if not video_path.exists():
        pytest.skip("Test video not found in data/")

    engine = RealTimeSurveillanceEngine(
        video_source=str(video_path),
        model_path=str(project_root / "models" / "convlstm_best.pth"),
        alerts_dir=str(temp_alerts_dir),
        spatter_burst_threshold=50
    )

    # Run for 100 frames: verifies both the initial standby traverse (frames 0-68)
    # and the subsequent arc ignition strike (frame 70+)
    metrics = engine.run(max_frames=100, show_display=False)

    assert metrics["total_frames"] == 100
    assert metrics["duration_sec"] > 0
    assert metrics["arc_on_sec"] > 0
    assert 20.0 <= metrics["duty_cycle_pct"] <= 50.0

    # Verify session report JSON was written
    report_file = temp_alerts_dir / "surveillance_session_report.json"
    assert report_file.exists()
    with open(report_file, "r") as f:
        data = json.load(f)
    assert "session_summary" in data
    assert "dispatched_alerts" in data


def test_safety_trip_engine_debounce_and_no_flapping():
    """Verify single-frame 50ms transient anomalies do not flap status from NOMINAL."""
    engine = IndustrialSafetyTripEngine(stressor_bound=100.0, decay_rate=2.0)

    # 1. Transient single frame defect spike
    is_halted, status, stress = engine.update(
        instantaneous_defect_score=20.0,
        defect_name="SPATTER_BURST",
        curr_timestamp=1.0,
        frame_idx=30
    )
    # 20.0 * 0.28 = 5.6 stress. 5.6 < 35.0, so status stays NOMINAL
    assert not is_halted
    assert status == "NOMINAL"
    assert round(stress, 1) == 5.6

    # 2. Next consecutive frame is nominal (no defect): stress cools down
    is_halted2, status2, stress2 = engine.update(
        instantaneous_defect_score=0.0,
        defect_name="",
        curr_timestamp=1.033,
        frame_idx=31
    )
    assert not is_halted2
    assert status2 == "NOMINAL"
    assert round(stress2, 1) == 3.6  # 5.6 - 2.0 = 3.6


def test_safety_trip_engine_sustained_defect_halt():
    """Verify sustained severe defects breach the 100.0 bound and trigger EMERGENCY STOP (E-STOP)."""
    engine = IndustrialSafetyTripEngine(stressor_bound=100.0, decay_rate=1.5)

    # Sustained burn-through defect over 12 frames
    for f in range(12):
        is_halted, status, stress = engine.update(
            instantaneous_defect_score=40.0,
            defect_name="BURN_THROUGH_RISK",
            curr_timestamp=2.0 + (f * 0.033),
            frame_idx=60 + f
        )

    # 40 * 0.28 = 11.2 per frame. After 10-12 frames, stress reaches 100.0
    assert is_halted is True
    assert status == "E_STOP_HALTED"
    assert stress >= 100.0
    assert "BURN_THROUGH_RISK" in engine.halt_reason

    # Verify manual operator reset clears the halt
    engine.reset()
    assert engine.is_halted is False
    assert engine.status_tier == "NOMINAL"
    assert engine.stressor_level == 0.0


def test_t0_ground_truth_anchor_and_photometric_normalization():
    """Verify that initial 5-10 frames establish the immutable T0 ground truth anchor and normalize color."""
    from src.realtime_surveillance import T0GroundTruthAnchorEngine

    anchor_engine = T0GroundTruthAnchorEngine(num_anchor_frames=8)
    assert not anchor_engine.is_calibrated

    # Generate 8 synthetic calibration frames with slight noise around mean 60
    np.random.seed(42)
    for i in range(8):
        frame = np.full((100, 100, 3), 60 + i, dtype=np.uint8)
        ready = anchor_engine.update_calibration(frame)
        if i < 7:
            assert ready is False
        else:
            assert ready is True

    assert anchor_engine.is_calibrated is True
    assert anchor_engine.anchor_frame is not None
    assert anchor_engine.anchor_frame.shape == (100, 100, 3)

    # Test photometric color normalization:
    # A drifting frame with higher exposure should be mapped back into anchor domain
    bright_frame = np.full((100, 100, 3), 120, dtype=np.uint8)
    norm_frame = anchor_engine.normalize_color(bright_frame)
    assert norm_frame.shape == bright_frame.shape
    # Normalized frame mean should shift towards anchor mean
    assert norm_frame.mean() < bright_frame.mean()

    # Test differential status when welding starts (localized arc strike):
    welding_frame = bright_frame.copy()
    cv2.circle(welding_frame, (50, 50), 15, (255, 255, 255), -1)
    norm_weld = anchor_engine.normalize_color(welding_frame)
    diff_status = anchor_engine.compute_differential_status(norm_weld)
    assert "status" in diff_status
    assert "mean_delta" in diff_status
    assert diff_status["mean_delta"] > 0.0

    # Test neural differential input tensor shape:
    crop = norm_frame[:32, :32]
    anchor_crop = anchor_engine.anchor_frame[:32, :32]
    tensor_input, diff_resized = anchor_engine.extract_differential_neural_input(crop, anchor_crop, target_size=(128, 128))
    assert tensor_input.shape == (3, 128, 128)
    assert diff_resized.shape == (128, 128, 3)


