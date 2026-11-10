#!/usr/bin/env python3
"""
WeldGuard-AI: Multi-Process Welding Video Ingestion & Surveillance Runner
Supports:
  1. YouTube video download via yt-dlp
  2. Local MP4/AVI/MOV video ingestion
  3. Automatic T0 Environment Baseline Lighting Calibration
  4. Multi-Process Welding Defect Surveillance (GMAW/MIG, GTAW/TIG, SMAW/Stick, FCAW, Laser)
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.realtime_surveillance import RealTimeSurveillanceEngine


def download_youtube_video(url: str, output_dir: Path) -> Path:
    """Downloads a YouTube welding video using yt-dlp at optimal 720p/1080p resolution."""
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp is required for YouTube video ingestion. Install with: pip install yt-dlp")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(output_dir / "%(id)s_%(title).50s.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": out_template,
        "merge_output_format": "mp4",
        "quiet": False,
        "no_warnings": True,
    }

    print(f"\n[WeldGuard Ingestion] Downloading YouTube welding source: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        # Handle merged mp4 extension
        downloaded_path = Path(filename).with_suffix(".mp4")
        if not downloaded_path.exists():
            downloaded_path = Path(filename)

    print(f"[WeldGuard Ingestion] Successfully saved to: {downloaded_path}\n")
    return downloaded_path


def main():
    parser = argparse.ArgumentParser(description="WeldGuard-AI Video Ingestion & Surveillance Suite")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", type=str, help="YouTube video URL to download and analyze")
    group.add_argument("--video", type=str, help="Local video file path or camera index (e.g. 0)")

    parser.add_argument(
        "--process",
        type=str,
        default="gmaw",
        choices=["gmaw", "gtaw", "smaw", "fcaw", "laser"],
        help="Welding process physics (gmaw=MIG, gtaw=TIG, smaw=Stick, fcaw=Flux-Cored, laser=Laser)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="robotic",
        choices=["robotic", "manual"],
        help="Operation target mode"
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Path for saving annotated surveillance video"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process (None for full video)"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Display live surveillance HUD window"
    )

    args = parser.parse_args()

    # Determine input video path
    if args.url:
        yt_cache_dir = Path("data/youtube_samples")
        video_path = download_youtube_video(args.url, yt_cache_dir)
        source = str(video_path)
    else:
        source = args.video

    # Determine output path if not specified
    if args.output_video:
        output_p = args.output_video
    else:
        stem = Path(source).stem if not source.isdigit() else f"cam_{source}"
        output_p = f"data/annotated_{args.process}_{stem}.mp4"

    print("=" * 80)
    print(f"[WeldGuard-AI Ingestion Suite]")
    print(f"Source: {source}")
    print(f"Process: {args.process.upper()} | Mode: {args.mode.upper()}")
    print(f"Annotated Output: {output_p}")
    print("=" * 80)

    engine = RealTimeSurveillanceEngine(
        video_source=source,
        mode=args.mode,
        process_type=args.process,
        output_video_path=output_p,
        roi_size=400,
        use_torch_roi=True
    )

    metrics = engine.run(max_frames=args.max_frames, show_display=args.gui)
    print("\n[WeldGuard-AI Ingestion Suite] Complete.")


if __name__ == "__main__":
    main()
