"""
AutoTwin-AI v2.0: Real-World Welding Video Telemetry & Anomaly Analyzer
Extracts:
  1. Arc-On Time (Exact duration & duty cycle from frame luminosity)
  2. Spatter Particle Tracking (OpenCV contour detection of flying sparks)
  3. Process Stability & Anomaly Detection (Temporal MSE residuals & 3-sigma thresholds)
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


def analyze_welding_video(video_path, output_dir=None, frame_stride=1, max_frames=None):
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
    print(f"[AutoTwin Telemetry] Analyzing: {video_path.name}")
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

    print(f"Resolution: {width}x{height} | FPS: {fps:.2f} | Total Frames: {total_frames} | Duration: {total_duration_sec:.2f}s ({total_duration_sec/60:.2f} min)")
    print(f"Processing stride: every {frame_stride} frame(s)")

    # Data collection arrays
    timestamps = []
    frame_indices = []
    arc_on_flags = []
    spatter_counts = []
    spatter_areas = []
    temporal_mse = []

    prev_gray = None
    processed_count = 0
    t0 = time.time()

    # Keyframes to capture for visual report
    keyframe_arc_on = None
    keyframe_spatter_burst = None
    keyframe_stable = None
    max_spatter_count = -1
    max_spatter_frame = None

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
            # 1. ARC-ON DETECTION
            # ---------------------------------------------------------
            # High intensity threshold in RGB: Arc is extremely bright white-blue
            # Detect pixels where V (brightness) is saturated > 240 and B+G > 460
            b_channel = frame[:, :, 0]
            g_channel = frame[:, :, 1]
            r_channel = frame[:, :, 2]

            arc_mask = (b_channel > 240) & (g_channel > 230) & (r_channel > 200)
            arc_pixel_count = int(np.count_nonzero(arc_mask))

            # Arc is active if there is a sustained core cluster of saturated pixels
            is_arc_on = arc_pixel_count > 150
            arc_on_flags.append(1 if is_arc_on else 0)

            if is_arc_on and keyframe_arc_on is None and current_time > 1.0:
                keyframe_arc_on = (frame.copy(), frame_idx, current_time)

            # ---------------------------------------------------------
            # 2. SPATTER TRACKING (OpenCV Contours)
            # ---------------------------------------------------------
            spark_count = 0
            spark_area_sum = 0
            annotated_frame = None

            if is_arc_on:
                # Dilate the arc mask to create an exclusion zone for the main plasma flame
                kernel_arc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
                dilated_arc = cv2.dilate(arc_mask.astype(np.uint8), kernel_arc)

                # Sparks are bright (high value in HSV / grayscale) outside the main arc flame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Threshold for incandescent sparks
                _, spark_thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
                # Exclude the central arc flame
                spark_candidates = cv2.bitwise_and(spark_thresh, spark_thresh, mask=(1 - dilated_arc))

                # Find individual spark contours
                contours, _ = cv2.findContours(spark_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    # Filter: individual sparks are between 2 and 250 pixels
                    if 2 <= area <= 300:
                        spark_count += 1
                        spark_area_sum += area

                if spark_count > max_spatter_count:
                    max_spatter_count = spark_count
                    max_spatter_frame = frame_idx
                    keyframe_spatter_burst = (frame.copy(), frame_idx, current_time, spark_count)

                if is_arc_on and 3 <= spark_count <= 8 and keyframe_stable is None and current_time > 5.0:
                    keyframe_stable = (frame.copy(), frame_idx, current_time, spark_count)

            spatter_counts.append(spark_count)
            spatter_areas.append(spark_area_sum)

            # ---------------------------------------------------------
            # 3. PROCESS STABILITY & TEMPORAL RESIDUAL (MSE)
            # ---------------------------------------------------------
            # Downsample frame for fast, robust temporal differencing
            small_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 180))

            if prev_gray is not None:
                diff = cv2.absdiff(small_gray, prev_gray)
                mse = float(np.mean(diff ** 2))
                temporal_mse.append(mse)
            else:
                temporal_mse.append(0.0)

            prev_gray = small_gray
            processed_count += 1

            if processed_count % 150 == 0:
                elapsed = time.time() - t0
                fps_proc = processed_count / elapsed
                print(f"  Frame {frame_idx:05d}/{total_frames:05d} ({current_time:6.1f}s) | Arc: {'ON ' if is_arc_on else 'OFF'} | Sparks: {spark_count:03d} | MSE: {temporal_mse[-1]:6.1f} | Speed: {fps_proc:.1f} fps")

        frame_idx += 1

    cap.release()
    total_elapsed = time.time() - t0
    print(f"[AutoTwin Telemetry] Finished analyzing {processed_count} frames in {total_elapsed:.1f}s ({processed_count/total_elapsed:.1f} fps)")

    # ---------------------------------------------------------
    # 4. TELEMETRY COMPUTATION & SUMMARY
    # ---------------------------------------------------------
    timestamps = np.array(timestamps)
    arc_on_flags = np.array(arc_on_flags)
    spatter_counts = np.array(spatter_counts)
    temporal_mse = np.array(temporal_mse)

    # Arc duration calculation
    dt = frame_stride / fps
    total_arc_on_sec = float(np.sum(arc_on_flags) * dt)
    duty_cycle_pct = (total_arc_on_sec / total_duration_sec) * 100.0 if total_duration_sec > 0 else 0.0

    # Group contiguous Arc-On sessions
    arc_sessions = []
    in_session = False
    session_start = 0

    for i, flag in enumerate(arc_on_flags):
        if flag == 1 and not in_session:
            in_session = True
            session_start = timestamps[i]
            session_start_idx = frame_indices[i]
        elif flag == 0 and in_session:
            in_session = False
            session_end = timestamps[i-1]
            session_end_idx = frame_indices[i-1]
            dur = session_end - session_start
            if dur >= 0.2:  # Filter momentary glitches
                arc_sessions.append({
                    "session_id": len(arc_sessions) + 1,
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
        arc_sessions.append({
            "session_id": len(arc_sessions) + 1,
            "start_time_sec": round(float(session_start), 2),
            "end_time_sec": round(float(session_end), 2),
            "duration_sec": round(float(dur), 2),
            "start_frame": int(session_start_idx),
            "end_frame": int(session_end_idx)
        })

    # Spatter statistics
    arc_spatter = spatter_counts[arc_on_flags == 1] if np.any(arc_on_flags == 1) else np.array([0])
    avg_spatter_per_frame = float(np.mean(arc_spatter))
    spatter_rate_per_sec = float(avg_spatter_per_frame * fps)
    peak_spatter = int(np.max(spatter_counts)) if len(spatter_counts) > 0 else 0

    # Process stability index (0 to 100%)
    # Baseline: Low MSE variance during arc-on = high stability
    arc_mse = temporal_mse[arc_on_flags == 1] if np.any(arc_on_flags == 1) else np.array([1.0])
    mse_mean = float(np.mean(arc_mse))
    mse_std = float(np.std(arc_mse))
    mse_threshold_3sigma = mse_mean + 3 * mse_std

    # Anomalies: Frames where MSE exceeds 3-sigma threshold during active arc
    anomaly_mask = (arc_on_flags == 1) & (temporal_mse > mse_threshold_3sigma)
    anomaly_count = int(np.count_nonzero(anomaly_mask))
    stability_index = max(0.0, min(100.0, 100.0 - (anomaly_count / max(1, len(arc_mse)) * 100.0 * 5.0)))

    # Summary report
    telemetry_summary = {
        "video_name": video_path.name,
        "video_duration_sec": round(total_duration_sec, 2),
        "video_duration_formatted": f"{int(total_duration_sec // 60)}m {int(total_duration_sec % 60)}s",
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "arc_telemetry": {
            "total_arc_on_sec": round(total_arc_on_sec, 2),
            "total_arc_on_formatted": f"{int(total_arc_on_sec // 60)}m {int(total_arc_on_sec % 60)}s",
            "duty_cycle_pct": round(duty_cycle_pct, 2),
            "arc_sessions_count": len(arc_sessions),
            "arc_sessions": arc_sessions
        },
        "spatter_telemetry": {
            "average_sparks_per_frame": round(avg_spatter_per_frame, 2),
            "estimated_sparks_per_sec": round(spatter_rate_per_sec, 1),
            "peak_sparks_count": peak_spatter,
            "peak_sparks_timestamp_sec": round(float(timestamps[np.argmax(spatter_counts)]), 2) if len(spatter_counts) > 0 else 0,
            "peak_sparks_frame": int(frame_indices[np.argmax(spatter_counts)]) if len(spatter_counts) > 0 else 0,
            "spatter_severity": "HIGH / BURST" if peak_spatter > 30 else ("MODERATE" if peak_spatter > 12 else "LOW")
        },
        "stability_telemetry": {
            "process_stability_index_pct": round(stability_index, 2),
            "mean_temporal_mse": round(mse_mean, 2),
            "std_temporal_mse": round(mse_std, 2),
            "anomaly_threshold_3sigma": round(mse_threshold_3sigma, 2),
            "detected_instability_frames_count": anomaly_count,
            "instability_percentage": round((anomaly_count / max(1, len(arc_mse))) * 100.0, 2)
        }
    }

    # Save JSON report
    json_path = output_dir / f"telemetry_report_{video_name}.json"
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(telemetry_summary, fp, indent=2)
    print(f"\n[AutoTwin Telemetry] Saved JSON summary to: {json_path}")

    # ---------------------------------------------------------
    # 5. MULTI-PANEL ANALYTICS PLOT
    # ---------------------------------------------------------
    plt.style.use('dark_background')
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=False)
    fig.suptitle(f"AutoTwin-AI v2.0: Welding Telemetry & Spatiotemporal Analysis\nVideo: {video_path.name} (Duration: {int(total_duration_sec//60)}m {int(total_duration_sec%60)}s | Arc-On: {int(total_arc_on_sec//60)}m {int(total_arc_on_sec%60)}s | Duty: {duty_cycle_pct:.1f}%)", fontsize=13, fontweight='bold', color='#00f0ff', y=0.98)

    # Panel 1: Arc-On Binary Pulse & Cumulative Time
    ax1 = axes[0]
    ax1.plot(timestamps, arc_on_flags, color='#00f0ff', linewidth=1.5, label='Arc Status (1=ON, 0=OFF)')
    ax1.fill_between(timestamps, 0, arc_on_flags, color='#00f0ff', alpha=0.25)
    ax1.set_ylabel("Arc Status", fontsize=10, color='#00f0ff', fontweight='bold')
    ax1.set_ylim(-0.1, 1.2)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['OFF', 'ON'])
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='upper right', framealpha=0.7)

    # Annotate arc sessions
    for s in arc_sessions[:4]:
        ax1.axvspan(s['start_time_sec'], s['end_time_sec'], color='#10b981', alpha=0.15)
        ax1.text((s['start_time_sec'] + s['end_time_sec']) / 2, 0.5, f"Weld #{s['session_id']}\n{s['duration_sec']}s", color='#10b981', fontsize=8, ha='center', fontweight='bold')

    # Panel 2: Spatter Count Timeline
    ax2 = axes[1]
    ax2.plot(timestamps, spatter_counts, color='#f59e0b', linewidth=1.2, label='Flying Spark Count')
    ax2.axhline(15, color='#ef4444', linestyle='--', linewidth=1.0, label='Burst Threshold (>15 sparks)')
    ax2.set_ylabel("Spatter Count", fontsize=10, color='#f59e0b', fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc='upper right', framealpha=0.7)

    # Highlight peak burst
    if len(spatter_counts) > 0 and peak_spatter > 0:
        peak_t = timestamps[np.argmax(spatter_counts)]
        ax2.scatter([peak_t], [peak_spatter], color='#ef4444', s=70, zorder=5)
        ax2.annotate(f"Peak Spatter: {peak_spatter} sparks\n@ {peak_t:.1f}s", xy=(peak_t, peak_spatter), xytext=(peak_t + 5, peak_spatter * 0.85),
                     arrowprops=dict(facecolor='#ef4444', shrink=0.05, width=1, headwidth=6), fontsize=8, color='#ef4444', fontweight='bold')

    # Panel 3: Process Stability & Temporal MSE
    ax3 = axes[2]
    ax3.plot(timestamps, temporal_mse, color='#a855f7', linewidth=1.2, label='Temporal Frame Difference MSE')
    ax3.axhline(mse_threshold_3sigma, color='#ef4444', linestyle='--', linewidth=1.2, label=f'Anomaly Threshold (3σ: {mse_threshold_3sigma:.1f})')
    ax3.fill_between(timestamps, mse_threshold_3sigma, temporal_mse, where=(temporal_mse > mse_threshold_3sigma), color='#ef4444', alpha=0.4, label='Process Instability Events')
    ax3.set_ylabel("Temporal MSE", fontsize=10, color='#a855f7', fontweight='bold')
    ax3.set_xlabel("Video Timestamp (seconds)", fontsize=10, color='#94a3b8')
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.legend(loc='upper right', framealpha=0.7)

    # Panel 4: Representative Keyframes
    ax4 = axes[3]
    ax4.axis('off')

    keyframes = []
    titles = []
    if keyframe_arc_on:
        keyframes.append(cv2.cvtColor(keyframe_arc_on[0], cv2.COLOR_BGR2RGB))
        titles.append(f"Arc Active\nt={keyframe_arc_on[2]:.1f}s (Frame #{keyframe_arc_on[1]})")
    if keyframe_stable:
        keyframes.append(cv2.cvtColor(keyframe_stable[0], cv2.COLOR_BGR2RGB))
        titles.append(f"Stable Pool Flow\nt={keyframe_stable[2]:.1f}s | Sparks: {keyframe_stable[3]}")
    if keyframe_spatter_burst:
        keyframes.append(cv2.cvtColor(keyframe_spatter_burst[0], cv2.COLOR_BGR2RGB))
        titles.append(f"Peak Spatter Burst\nt={keyframe_spatter_burst[2]:.1f}s | Sparks: {keyframe_spatter_burst[3]}")

    if keyframes:
        for idx, (img, ttl) in enumerate(zip(keyframes, titles)):
            sub_ax = fig.add_axes([0.13 + idx * 0.28, 0.03, 0.24, 0.18])
            sub_ax.imshow(img)
            sub_ax.set_title(ttl, fontsize=8, color='#00f0ff', fontweight='bold')
            sub_ax.axis('off')

    plt.tight_layout(rect=[0, 0.22, 1, 0.96])
    plot_path = output_dir / f"telemetry_analysis_{video_name}.png"
    plt.savefig(plot_path, dpi=180, facecolor='#0a0d14')
    plt.close()
    print(f"[AutoTwin Telemetry] Saved multi-panel analytics plot to: {plot_path}")

    return telemetry_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoTwin-AI Real-World Welding Video Telemetry Analyzer")
    parser.add_argument("--video", type=str, default=r"C:\Projects\DigitalTwin\data\video_20260908_170143.mp4", help="Path to MP4 video")
    parser.add_argument("--stride", type=int, default=2, help="Frame stride (e.g. 1 for full 30fps, 2 for 15fps sampling)")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional max frames to process for quick testing")
    args = parser.parse_args()

    analyze_welding_video(args.video, frame_stride=args.stride, max_frames=args.max_frames)
