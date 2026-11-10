"""
AutoTwin-AI / WeldGuard-AI: Dataset Packager for Kaggle GPU Training
Extracts synchronized 128x128 frames from high-resolution industrial welding videos
into compact, memory-mapped compressed NPZ archives ready for high-speed GPU training.
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def extract_and_pack_video(video_path, output_npz_path, target_size=(128, 128), stride=2, max_frames=None, roi_size=400):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print(f"[Dataset Packer] Processing {video_path.name} (Stride={stride}, Target={target_size}, ROI={roi_size})...")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Dynamic torch tracker with spark-guided triangulation
    smooth_cx, smooth_cy = None, None
    last_known_pos = None
    smooth_alpha = 0.35

    frames = []
    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames and frame_idx >= max_frames:
            break

        if frame_idx % stride == 0:
            # 1. Detect incandescent sparks
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, th_sparks = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY)
            cnts_sparks, _ = cv2.findContours(th_sparks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            spark_pts = []
            for c in cnts_sparks:
                a = cv2.contourArea(c)
                if 2 <= a <= 150:
                    M = cv2.moments(c)
                    if M['m00'] > 0:
                        spark_pts.append((M['m10'] / M['m00'], M['m01'] / M['m00']))

            spark_center = None
            if len(spark_pts) >= 4:
                spark_center = (float(np.median([p[0] for p in spark_pts])), float(np.median([p[1] for p in spark_pts])))

            # 2. Compact arc core candidates (reject massive sunlight sheets > 4000px)
            b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
            saturated = ((b > 240) & (g > 230) & (r > 200)).astype(np.uint8) * 255
            s_cnts, _ = cv2.findContours(saturated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            candidates = []
            for c in s_cnts:
                a = cv2.contourArea(c)
                x, y, w, h = cv2.boundingRect(c)
                if 30 <= a <= 4000 and w < 220 and h < 220:
                    M = cv2.moments(c)
                    if M['m00'] > 0:
                        candidates.append((int(M['m10'] / M['m00']), int(M['m01'] / M['m00']), a))

            pos = None
            if candidates and spark_center is not None:
                chosen = min(candidates, key=lambda item: np.hypot(item[0] - spark_center[0], item[1] - spark_center[1]))
                if np.hypot(chosen[0] - spark_center[0], chosen[1] - spark_center[1]) < 250:
                    pos = (chosen[0], chosen[1])
                else:
                    pos = (int(spark_center[0]), int(spark_center[1]))
            elif spark_center is not None:
                pos = (int(spark_center[0]), int(spark_center[1]))
            elif candidates and last_known_pos is not None:
                chosen = min(candidates, key=lambda item: np.hypot(item[0] - last_known_pos[0], item[1] - last_known_pos[1]))
                if np.hypot(chosen[0] - last_known_pos[0], chosen[1] - last_known_pos[1]) < 200:
                    pos = (chosen[0], chosen[1])
            elif candidates:
                chosen = max(candidates, key=lambda item: item[2])
                pos = (chosen[0], chosen[1])

            if pos is not None:
                last_known_pos = pos
                raw_cx, raw_cy = pos
                if smooth_cx is None:
                    smooth_cx, smooth_cy = float(raw_cx), float(raw_cy)
                else:
                    smooth_cx = smooth_alpha * raw_cx + (1 - smooth_alpha) * smooth_cx
                    smooth_cy = smooth_alpha * raw_cy + (1 - smooth_alpha) * smooth_cy

            if smooth_cx is not None:
                cx, cy = int(round(smooth_cx)), int(round(smooth_cy))
            else:
                cx, cy = orig_w // 2, orig_h // 2

            # Extract 400x400 torch crop
            half = roi_size // 2
            x1 = max(0, min(orig_w - roi_size, cx - half))
            y1 = max(0, min(orig_h - roi_size, cy - half))
            x2 = min(orig_w, x1 + roi_size)
            y2 = min(orig_h, y1 + roi_size)
            torch_crop = frame[y1:y2, x1:x2]

            # Resize the true moving torch crop to target resolution (128x128)
            small = cv2.resize(torch_crop, target_size, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            frames.append(rgb)

        frame_idx += 1
        if frame_idx % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  --> Decoded {frame_idx}/{total_frames} frames ({frame_idx/elapsed:.1f} fps)")

    cap.release()

    arr = np.array(frames, dtype=np.uint8)
    elapsed = time.time() - t0
    print(f"  Extracted {len(arr)} frames in {elapsed:.1f}s. Array shape: {arr.shape}, dtype: {arr.dtype}")

    # Save to compressed NPZ
    output_npz_path = Path(output_npz_path)
    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz_path, frames=arr, fps=fps, stride=stride, original_total_frames=total_frames)

    file_size_mb = output_npz_path.stat().st_size / (1024 * 1024)
    print(f"  Saved compressed archive: {output_npz_path.name} ({file_size_mb:.2f} MB)")
    return {
        'source_video': video_path.name,
        'archive_file': output_npz_path.name,
        'num_frames': len(arr),
        'resolution': target_size,
        'fps': round(float(fps), 2),
        'stride': stride,
        'file_size_mb': round(file_size_mb, 2)
    }


def prepare_kaggle_dataset_dir(project_root, output_dir=None):
    project_root = Path(project_root)
    if output_dir is None:
        output_dir = project_root / "kaggle_dataset"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = [
        project_root / "data" / "video_20260908_170143.mp4",
        project_root / "data" / "video_20260908_171604.mp4",
        project_root / "data" / "IMG_3601.MOV"
    ]

    manifest = {'dataset_name': 'weldguard-welding-dataset', 'archives': []}

    for vid in videos:
        if vid.exists():
            out_npz = output_dir / f"{vid.stem}_128x128.npz"
            # Using stride 2 provides ~3400 frames per video at 15 FPS temporal resolution
            # which is ideal for sliding window ConvLSTM next-frame prediction
            info = extract_and_pack_video(vid, out_npz, target_size=(128, 128), stride=2)
            manifest['archives'].append(info)

    # Save metadata JSON
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Create dataset-metadata.json for Kaggle CLI
    kaggle_metadata = {
        "title": "WeldGuard Real-World Welding Videos Dataset",
        "id": "deepark121/weldguard-welding-dataset",
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(output_dir / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(kaggle_metadata, f, indent=2)

    print("\n" + "=" * 70)
    print(f"[Dataset Packer Complete] Staged Kaggle dataset in: {output_dir}")
    print(f"Total Archives: {len(manifest['archives'])}")
    print("=" * 70)
    return output_dir


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    prepare_kaggle_dataset_dir(root)
