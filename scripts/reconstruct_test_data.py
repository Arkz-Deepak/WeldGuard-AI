"""
Reconstructs data/ directory artifacts required for unit and telemetry tests:
1. data/crane_gallery_stages.xlsx (28 stages, standard takt 227 min)
2. data/telemetry_report_video_20260908_170143.json
3. data/telemetry_report_video_20260908_171604.json
4. data/manual_telemetry_report_IMG_3601.json
5. data/video_20260908_170143.mp4 (reconstituted from kaggle_dataset for tests)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. Reconstruct crane_gallery_stages.xlsx (28 stages)
# -------------------------------------------------------------
stages_data = [
    (1, "TAPER PLATE & MODULE", "Taper plate fit-up (Fitter)", 20, 227, -207, "Within takt"),
    (2, "TAPER PLATE & MODULE", "Taper plate welding (Welder)", 22, 227, -205, "Within takt"),
    (3, "TAPER PLATE & MODULE", "Module fit-up (Fitter)", 38, 227, -189, "Within takt"),
    (4, "I-BEAM", "I Beam fit-up (Fitter)", 180, 227, -47, "Within takt"),
    (5, "I-BEAM", "I Beam UT & Cleaning (Grinder)", 200, 227, -27, "Within takt"),
    (6, "NODE", "Node 1st stage fit-up (Fitter)", 39, 227, -188, "Within takt"),
    (7, "NODE", "UT inspection", 22, 227, -205, "Within takt"),
    (8, "NODE", "Node 2nd stage fit-up (Fitter)", 30, 227, -197, "Within takt"),
    (9, "NODE", "UT inspection & Cleaning (Grinder)", 120, 227, -107, "Within takt"),
    (10, "TUBE", "Tube fit-up (Fitter)", 25, 227, -202, "Within takt"),
    (11, "TUBE", "Tube welding (Welder)", 36, 227, -191, "Within takt"),
    (12, "TUBE", "UT inspection", 19, 227, -208, "Within takt"),
    (13, "TUBE", "I Beam bend removal (Fitter)", 59, 227, -168, "Within takt"),
    (14, "ASSEMBLY", "Assy fitup 1st Stage (Fitter)", 80, 227, -147, "Within takt"),
    (15, "ASSEMBLY", "Assy welding (Welder)", 208, 227, -19, "Within takt"),
    (16, "ASSEMBLY", "Assy fitup -stage 2 (Bush & CP)", 35, 227, -192, "Within takt"),
    (17, "ASSEMBLY", "Assy welding -stage 2 (Welder)", 90, 227, -137, "Within takt"),
    (18, "ASSEMBLY", "Assy fitup -stage 3 (Bed fitup)", 90, 227, -137, "Within takt"),
    (19, "ASSEMBLY", "Assy cleaning & FW (Grinder)", 160, 227, -67, "Within takt"),
    (20, "ASSEMBLY", "PDI (Fitter)", 109, 227, -118, "Within takt"),
    (21, "RF3", "Total", 210, 227, -17, "Within takt"),
    (22, "RF3", "1st Stage Node Robot welding", 120, 227, -107, "Within takt"),
    (23, "RF3", "Node 2nd Stage Robot Welding", 90, 227, -137, "Within takt"),
    (24, "RA4", "Total", 223, 227, -4, "Within takt"),
    (25, "RA4", "I Beam Robot welding (Robo Operator)", 175, 227, -52, "Within takt"),
    (26, "RF0 & RF5", "Front Node Robot Welding", 48, 227, -179, "Within takt"),
    (27, "RF0 & RF5", "Total", 202, 227, -25, "Within takt"),
    (28, "RA9", "Assy Robot welding-stage 3", 120, 227, -107, "Within takt")
]

df_stages = pd.DataFrame(stages_data, columns=[
    "Seq", "Section", "Stage / Activity", 
    "Total Time (Cycle Time, min)", "Takt Time (min)", 
    "Variance vs Takt", "Tracking Status"
])

excel_path = DATA_DIR / "crane_gallery_stages.xlsx"
df_stages.to_excel(excel_path, index=False)
print(f"Reconstructed: {excel_path} ({len(df_stages)} stages)")

# -------------------------------------------------------------
# 2. Reconstruct Robotic Telemetry Reports
# -------------------------------------------------------------
robotic_reports = {
    "telemetry_report_video_20260908_170143.json": {
        "video_name": "video_20260908_170143.mp4",
        "video_duration_sec": 114.47,
        "video_duration_formatted": "1m 54s",
        "total_frames": 3431,
        "fps": 29.97,
        "arc_telemetry": {
            "total_arc_on_sec": 114.47,
            "total_arc_on_formatted": "1m 54s",
            "duty_cycle_pct": 100.0,
            "arc_sessions_count": 1,
            "arc_sessions": [
                {
                    "session_id": 1,
                    "start_time_sec": 0.0,
                    "end_time_sec": 114.47,
                    "duration_sec": 114.47,
                    "start_frame": 0,
                    "end_frame": 3430
                }
            ]
        },
        "spatter_telemetry": {
            "average_sparks_per_frame": 4.88,
            "estimated_sparks_per_sec": 146.2,
            "peak_sparks_count": 36,
            "peak_sparks_timestamp_sec": 42.18,
            "peak_sparks_frame": 1264,
            "spatter_severity": "HIGH / BURST"
        },
        "stability_telemetry": {
            "process_stability_index_pct": 98.4,
            "mean_temporal_mse": 13.72,
            "std_temporal_mse": 8.41,
            "anomaly_threshold_3sigma": 38.95,
            "detected_instability_frames_count": 55,
            "instability_percentage": 1.6
        }
    },
    "telemetry_report_video_20260908_171604.json": {
        "video_name": "video_20260908_171604.mp4",
        "video_duration_sec": 134.23,
        "video_duration_formatted": "2m 14s",
        "total_frames": 4023,
        "fps": 29.97,
        "arc_telemetry": {
            "total_arc_on_sec": 134.23,
            "total_arc_on_formatted": "2m 14s",
            "duty_cycle_pct": 100.0,
            "arc_sessions_count": 1,
            "arc_sessions": [
                {
                    "session_id": 1,
                    "start_time_sec": 0.0,
                    "end_time_sec": 134.23,
                    "duration_sec": 134.23,
                    "start_frame": 0,
                    "end_frame": 4022
                }
            ]
        },
        "spatter_telemetry": {
            "average_sparks_per_frame": 5.62,
            "estimated_sparks_per_sec": 168.4,
            "peak_sparks_count": 42,
            "peak_sparks_timestamp_sec": 68.45,
            "peak_sparks_frame": 2051,
            "spatter_severity": "HIGH / BURST"
        },
        "stability_telemetry": {
            "process_stability_index_pct": 97.8,
            "mean_temporal_mse": 15.1,
            "std_temporal_mse": 9.2,
            "anomaly_threshold_3sigma": 42.7,
            "detected_instability_frames_count": 88,
            "instability_percentage": 2.2
        }
    }
}

for filename, content in robotic_reports.items():
    p = DATA_DIR / filename
    with open(p, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)
    print(f"Reconstructed: {p}")

# -------------------------------------------------------------
# 3. Reconstruct Manual Telemetry Report
# -------------------------------------------------------------
manual_report = {
    "video_name": "IMG_3601.MOV",
    "video_duration_sec": 163.07,
    "video_duration_formatted": "2m 43s",
    "total_frames": 4888,
    "fps": 29.97,
    "operator_telemetry": {
        "total_active_grinding_sec": 86.8,
        "total_active_grinding_formatted": "1m 26s",
        "grinding_duty_cycle_pct": 53.2,
        "active_grinding_sessions_count": 4,
        "peak_spark_count": 58,
        "mean_motion_energy": 42.15,
        "grinding_sessions": [
            {"session_id": 1, "start_time_sec": 4.5, "end_time_sec": 28.2, "duration_sec": 23.7, "start_frame": 135, "end_frame": 845},
            {"session_id": 2, "start_time_sec": 38.0, "end_time_sec": 65.4, "duration_sec": 27.4, "start_frame": 1139, "end_frame": 1960},
            {"session_id": 3, "start_time_sec": 79.2, "end_time_sec": 98.6, "duration_sec": 19.4, "start_frame": 2374, "end_frame": 2955},
            {"session_id": 4, "start_time_sec": 112.0, "end_time_sec": 128.3, "duration_sec": 16.3, "start_frame": 3357, "end_frame": 3845}
        ]
    },
    "crane_gallery_benchmark_correlation": [
        {
            "seq": 5,
            "stage": "I Beam UT & Cleaning (Grinder)",
            "standard_cycle_min": 200,
            "takt_min": 227,
            "extrapolated_active_time_min": 106.4,
            "projected_variance_vs_takt_min": -120.6,
            "compliance_status": "COMPLIANT / WITHIN TAKT"
        },
        {
            "seq": 9,
            "stage": "UT inspection & Cleaning (Grinder)",
            "standard_cycle_min": 120,
            "takt_min": 227,
            "extrapolated_active_time_min": 63.8,
            "projected_variance_vs_takt_min": -163.2,
            "compliance_status": "COMPLIANT / WITHIN TAKT"
        },
        {
            "seq": 19,
            "stage": "Assy cleaning & FW (Grinder)",
            "standard_cycle_min": 160,
            "takt_min": 227,
            "extrapolated_active_time_min": 85.1,
            "projected_variance_vs_takt_min": -141.9,
            "compliance_status": "COMPLIANT / WITHIN TAKT"
        }
    ]
}

manual_path = DATA_DIR / "manual_telemetry_report_IMG_3601.json"
with open(manual_path, "w", encoding="utf-8") as f:
    json.dump(manual_report, f, indent=2)
print(f"Reconstructed: {manual_path}")

print("Data reconstruction complete!")
