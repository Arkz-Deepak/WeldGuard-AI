"""
AutoTwin-AI v2.0: Manual Fabrication & Grinding Telemetry Analyzer
Analyzes manual fabrication videos (e.g. IMG_3601.MOV):
  1. Grinding Tool Engagement & Spark Stream Detection (Active Grinding vs Idle/Setup)
  2. Workstation Motion Dynamics & Operator Activity Index
  3. Work Session Clustering (Grinding Bursts vs Inspection/Repositioning Pauses)
  4. Benchmark Correlation against Crane Gallery Fabrication Stages (crane_gallery_stages.xlsx)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def analyze_manual_fabrication(video_path, stages_excel_path=None, output_dir=None, frame_stride=2, max_frames=None):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_dir is None:
        output_dir = video_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_name = video_path.stem
    print("=" * 75)
    print(f"[AutoTwin Manual Telemetry] Analyzing: {video_path.name}")
    print("=" * 75)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_duration_sec = total_frames / fps

    print(f"Resolution: {width}x{height} | FPS: {fps:.2f} | Frames: {total_frames} | Duration: {total_duration_sec:.2f}s ({total_duration_sec/60:.2f} min)")
    print(f"Processing stride: every {frame_stride} frame(s)")

    # Data collection
    timestamps = []
    frame_indices = []
    grinding_active_flags = []
    spark_counts = []
    motion_energies = []

    prev_gray = None
    processed_count = 0
    t0 = time.time()

    # Visual samples
    keyframe_grinding = None
    keyframe_idle = None
    max_sparks = -1
    keyframe_peak_spark = None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames and frame_idx >= max_frames:
            break

        if frame_idx % frame_stride == 0:
            current_time = frame_idx / fps
            timestamps.append(current_time)
            frame_indices.append(frame_idx)

            # ---------------------------------------------------------
            # 1. GRINDING SPARK DETECTION
            # ---------------------------------------------------------
            # Grinding sparks are high-intensity incandescent trails
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Threshold for bright incandescent sparks (pixel > 215)
            _, spark_thresh = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
            
            contours, _ = cv2.findContours(spark_candidates := spark_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            spark_count = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 2 <= area <= 400:
                    spark_count += 1

            # Grinding wheel contact creates stream of > 8 sparks
            is_grinding = spark_count >= 8
            grinding_active_flags.append(1 if is_grinding else 0)
            spark_counts.append(spark_count)

            if spark_count > max_sparks:
                max_sparks = spark_count
                keyframe_peak_spark = (frame.copy(), frame_idx, current_time, spark_count)

            if is_grinding and keyframe_grinding is None and current_time > 5.0:
                keyframe_grinding = (frame.copy(), frame_idx, current_time, spark_count)

            if not is_grinding and keyframe_idle is None and current_time > 10.0:
                keyframe_idle = (frame.copy(), frame_idx, current_time)

            # ---------------------------------------------------------
            # 2. WORKSTATION MOTION DYNAMICS
            # ---------------------------------------------------------
            small_gray = cv2.resize(gray, (320, 180))
            if prev_gray is not None:
                diff = cv2.absdiff(small_gray, prev_gray)
                motion_energy = float(np.mean(diff))
                motion_energies.append(motion_energy)
            else:
                motion_energies.append(0.0)
            prev_gray = small_gray

            processed_count += 1
            if processed_count % 300 == 0:
                elapsed = time.time() - t0
                print(f"  Frame {frame_idx:05d}/{total_frames:05d} ({current_time:6.1f}s) | Grinding: {'ACTIVE' if is_grinding else 'IDLE  '} | Sparks: {spark_count:03d} | Motion: {motion_energies[-1]:.2f} | Speed: {processed_count/elapsed:.1f} fps")

        frame_idx += 1

    cap.release()
    total_elapsed = time.time() - t0
    print(f"[AutoTwin Manual Telemetry] Processed {processed_count} frames in {total_elapsed:.1f}s ({processed_count/total_elapsed:.1f} fps)")

    # ---------------------------------------------------------
    # 3. METRICS COMPUTATION & CLUSTERING
    # ---------------------------------------------------------
    timestamps = np.array(timestamps)
    grinding_flags = np.array(grinding_active_flags)
    spark_counts = np.array(spark_counts)
    motion_energies = np.array(motion_energies)

    dt = frame_stride / fps
    total_grinding_sec = float(np.sum(grinding_flags) * dt)
    total_idle_sec = max(0.0, total_duration_sec - total_grinding_sec)
    grinding_duty_cycle_pct = (total_grinding_sec / total_duration_sec) * 100.0 if total_duration_sec > 0 else 0.0

    # Cluster grinding sessions
    grinding_sessions = []
    in_session = False
    session_start = 0

    for i, flag in enumerate(grinding_flags):
        if flag == 1 and not in_session:
            in_session = True
            session_start = timestamps[i]
            session_start_idx = frame_indices[i]
        elif flag == 0 and in_session:
            in_session = False
            session_end = timestamps[i-1]
            session_end_idx = frame_indices[i-1]
            dur = session_end - session_start
            if dur >= 0.3:
                grinding_sessions.append({
                    "session_id": len(grinding_sessions) + 1,
                    "start_time_sec": round(float(session_start), 2),
                    "end_time_sec": round(float(session_end), 2),
                    "duration_sec": round(float(dur), 2),
                    "start_frame": int(session_start_idx),
                    "end_frame": int(session_end_idx)
                })

    if in_session:
        session_end = timestamps[-1]
        session_end_idx = frame_indices[-1]
        dur = session_end - session_start
        grinding_sessions.append({
            "session_id": len(grinding_sessions) + 1,
            "start_time_sec": round(float(session_start), 2),
            "end_time_sec": round(float(session_end), 2),
            "duration_sec": round(float(dur), 2),
            "start_frame": int(session_start_idx),
            "end_frame": int(session_end_idx)
        })

    # Benchmark comparison from crane_gallery_stages.xlsx
    benchmark_stages = [
        {"seq": 5, "stage": "I Beam UT & Cleaning (Grinder)", "standard_cycle_min": 200, "takt_min": 227},
        {"seq": 9, "stage": "UT inspection & Cleaning (Grinder)", "standard_cycle_min": 120, "takt_min": 227},
        {"seq": 19, "stage": "Assy cleaning & FW (Grinder)", "standard_cycle_min": 160, "takt_min": 227}
    ]

    # Extrapolate observed duty cycle to a 120-200 min cleaning stage
    # E.g. at this duty cycle, how many net tool-contact minutes vs positioning/inspection minutes?
    extrapolated_benchmarks = []
    for b in benchmark_stages:
        net_grinding_min = b['standard_cycle_min'] * (grinding_duty_cycle_pct / 100.0)
        setup_inspection_min = b['standard_cycle_min'] * (1.0 - grinding_duty_cycle_pct / 100.0)
        extrapolated_benchmarks.append({
            "seq": b['seq'],
            "stage_name": b['stage'],
            "total_cycle_min": b['standard_cycle_min'],
            "takt_min": b['takt_min'],
            "estimated_net_grinding_min": round(net_grinding_min, 1),
            "estimated_setup_inspection_min": round(setup_inspection_min, 1),
            "takt_compliance": "PASS (WITHIN TAKT)" if b['standard_cycle_min'] <= b['takt_min'] else "FAIL (EXCEEDS TAKT)"
        })

    summary_json = {
        "video_name": video_path.name,
        "video_duration_sec": round(total_duration_sec, 2),
        "video_duration_formatted": f"{int(total_duration_sec // 60)}m {int(total_duration_sec % 60)}s",
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "operator_telemetry": {
            "total_active_grinding_sec": round(total_grinding_sec, 2),
            "total_active_grinding_formatted": f"{int(total_grinding_sec // 60)}m {int(total_grinding_sec % 60)}s",
            "total_idle_setup_sec": round(total_idle_sec, 2),
            "total_idle_setup_formatted": f"{int(total_idle_sec // 60)}m {int(total_idle_sec % 60)}s",
            "grinding_duty_cycle_pct": round(grinding_duty_cycle_pct, 2),
            "active_grinding_sessions_count": len(grinding_sessions),
            "peak_spark_count": max_sparks,
            "mean_motion_energy": round(float(np.mean(motion_energies)), 2),
            "grinding_sessions": grinding_sessions[:10]  # First 10 sessions
        },
        "crane_gallery_benchmark_correlation": extrapolated_benchmarks
    }

    # Save JSON report
    json_path = output_dir / f"manual_telemetry_report_{video_name}.json"
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(summary_json, fp, indent=2)
    print(f"[AutoTwin Manual Telemetry] Saved JSON report to: {json_path}")

    # ---------------------------------------------------------
    # 4. VISUALIZATION PLOT
    # ---------------------------------------------------------
    plt.style.use('dark_background')
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=False)
    fig.suptitle(f"AutoTwin-AI v2.0: Manual Fabrication & Grinding Telemetry\nVideo: {video_path.name} (Duration: {int(total_duration_sec//60)}m {int(total_duration_sec%60)}s | Active Grinding: {int(total_grinding_sec//60)}m {int(total_grinding_sec%60)}s | Duty: {grinding_duty_cycle_pct:.1f}%)", fontsize=13, fontweight='bold', color='#00f0ff', y=0.98)

    # Panel 1: Grinding Active Flag Timeline
    ax1 = axes[0]
    ax1.plot(timestamps, grinding_flags, color='#00f0ff', linewidth=1.5, label='Tool Engagement (1=Grinding, 0=Idle/Setup)')
    ax1.fill_between(timestamps, 0, grinding_flags, color='#00f0ff', alpha=0.25)
    ax1.set_ylabel("Tool Status", fontsize=10, color='#00f0ff', fontweight='bold')
    ax1.set_ylim(-0.1, 1.2)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['IDLE/SETUP', 'GRINDING'])
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='upper right', framealpha=0.7)

    for s in grinding_sessions[:5]:
        ax1.axvspan(s['start_time_sec'], s['end_time_sec'], color='#10b981', alpha=0.15)

    # Panel 2: Spark Intensity Timeline
    ax2 = axes[1]
    ax2.plot(timestamps, spark_counts, color='#f59e0b', linewidth=1.2, label='Grinding Spark Stream Count')
    ax2.axhline(8, color='#ef4444', linestyle='--', linewidth=1.0, label='Contact Threshold (>=8 sparks)')
    ax2.set_ylabel("Spark Count", fontsize=10, color='#f59e0b', fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc='upper right', framealpha=0.7)

    if max_sparks > 0 and len(spark_counts) > 0:
        peak_t = timestamps[np.argmax(spark_counts)]
        ax2.scatter([peak_t], [max_sparks], color='#ef4444', s=70, zorder=5)
        ax2.annotate(f"Peak Stream: {max_sparks} sparks\n@ {peak_t:.1f}s", xy=(peak_t, max_sparks), xytext=(peak_t + 5, max_sparks * 0.85),
                     arrowprops=dict(facecolor='#ef4444', shrink=0.05, width=1, headwidth=6), fontsize=8, color='#ef4444', fontweight='bold')

    # Panel 3: Workstation Motion Energy
    ax3 = axes[2]
    ax3.plot(timestamps, motion_energies, color='#a855f7', linewidth=1.2, label='Operator & Tool Motion Energy')
    ax3.set_ylabel("Motion Energy", fontsize=10, color='#a855f7', fontweight='bold')
    ax3.set_xlabel("Video Timestamp (seconds)", fontsize=10, color='#94a3b8')
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.legend(loc='upper right', framealpha=0.7)

    # Panel 4: Representative Keyframes
    ax4 = axes[3]
    ax4.axis('off')

    keyframes = []
    titles = []
    if keyframe_grinding:
        keyframes.append(cv2.cvtColor(keyframe_grinding[0], cv2.COLOR_BGR2RGB))
        titles.append(f"Active Grinding Contact\nt={keyframe_grinding[2]:.1f}s | Sparks: {keyframe_grinding[3]}")
    if keyframe_idle:
        keyframes.append(cv2.cvtColor(keyframe_idle[0], cv2.COLOR_BGR2RGB))
        titles.append(f"Idle / Inspection / Setup\nt={keyframe_idle[2]:.1f}s (Frame #{keyframe_idle[1]})")
    if keyframe_peak_spark:
        keyframes.append(cv2.cvtColor(keyframe_peak_spark[0], cv2.COLOR_BGR2RGB))
        titles.append(f"Peak Spark Stream\nt={keyframe_peak_spark[2]:.1f}s | Sparks: {keyframe_peak_spark[3]}")

    if keyframes:
        for idx, (img, ttl) in enumerate(zip(keyframes, titles)):
            sub_ax = fig.add_axes([0.13 + idx * 0.28, 0.03, 0.24, 0.18])
            sub_ax.imshow(img)
            sub_ax.set_title(ttl, fontsize=8, color='#00f0ff', fontweight='bold')
            sub_ax.axis('off')

    plt.tight_layout(rect=[0, 0.22, 1, 0.96])
    plot_path = output_dir / f"manual_telemetry_analysis_{video_name}.png"
    plt.savefig(plot_path, dpi=180, facecolor='#0a0d14')
    plt.close()
    print(f"[AutoTwin Manual Telemetry] Saved plot to: {plot_path}")

    return summary_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoTwin-AI Manual Fabrication Telemetry Analyzer")
    parser.add_argument("--video", type=str, default=r"C:\Projects\DigitalTwin\data\IMG_3601.MOV", help="Path to manual video")
    parser.add_argument("--stride", type=int, default=3, help="Frame stride for processing")
    args = parser.parse_args()

    analyze_manual_fabrication(args.video, frame_stride=args.stride)
