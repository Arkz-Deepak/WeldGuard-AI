"""
AutoTwin-AI v2.0: Spatiotemporal ConvLSTM Video Training & Anomaly Evaluation
Trains the ConvLSTM model on real-world welding video for next-frame prediction,
evaluates temporal MSE reconstruction errors, detects spatter/arc anomalies,
and generates multi-panel visual anomaly reports.
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
import torch
import torch.nn as nn
import torch.optim as optim

from models.conv_lstm import SpatiotemporalConvLSTMAutoencoder
from dataset_video import get_video_dataloader, WeldingVideoDataset


def train_conv_lstm(video_path, epochs=5, batch_size=4, seq_len=8, stride=6, lr=1e-3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 75)
    print(f"[AutoTwin ConvLSTM] Training Spatiotemporal Model on: {Path(video_path).name}")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | Seq Len (T): {seq_len}")
    print("=" * 75)

    # Prepare DataLoader
    dataloader = get_video_dataloader(
        video_path=video_path,
        seq_len=seq_len,
        stride=stride,
        img_size=(128, 128),
        batch_size=batch_size,
        max_sequences=120,  # Curated training set for fast convergence
        shuffle=True
    )

    model = SpatiotemporalConvLSTMAutoencoder(in_channels=3, latent_dim=64).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    model_save_dir = Path(__file__).resolve().parent.parent / "models"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_save_dir / "convlstm_best.pth"

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for past_seq, target_next in dataloader:
            past_seq = past_seq.to(device)
            target_next = target_next.to(device)

            optimizer.zero_grad()
            pred_next = model(past_seq)
            loss = criterion(pred_next, target_next)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        scheduler.step()
        avg_loss = epoch_loss / max(1, batch_count)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Next-Frame MSE Loss: {avg_loss:.6f} | Elapsed: {time.time() - t_start:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': best_loss,
                'seq_len': seq_len,
                'latent_dim': 64
            }, best_model_path)
            print(f"  --> Saved new best ConvLSTM checkpoint to: {best_model_path.name}")

    print("=" * 75)
    print(f"[SUCCESS] Training complete! Best Loss: {best_loss:.6f}")
    return model, best_model_path


def evaluate_video_anomalies(video_path, model_path=None, seq_len=8, stride=4, device=None, output_dir=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_path = Path(video_path)
    if output_dir is None:
        output_dir = video_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print(f"[AutoTwin ConvLSTM] Evaluating Spatiotemporal Video Anomalies: {video_path.name}")
    print("=" * 75)

    model = SpatiotemporalConvLSTMAutoencoder(in_channels=3, latent_dim=64).to(device)
    if model_path and Path(model_path).exists():
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
        print(f"Loaded trained ConvLSTM weights from: {model_path}")
    model.eval()

    # Load video frames sequentially
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    buffer_frames = []
    timestamps = []
    anomaly_scores = []
    frame_indices = []

    # Store sample frames for visual reporting
    sample_nominal = None
    sample_anomaly = None
    max_anomaly_score = -1

    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (128, 128))
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        tensor_frame = torch.from_numpy(rgb_frame).permute(2, 0, 1).float() / 255.0

        buffer_frames.append(tensor_frame)

        if len(buffer_frames) == seq_len + 1:
            if frame_idx % stride == 0:
                past_seq = torch.stack(buffer_frames[:seq_len], dim=0).unsqueeze(0).to(device)  # [1, T, 3, H, W]
                target_next = buffer_frames[seq_len].unsqueeze(0).to(device)                      # [1, 3, H, W]

                with torch.no_grad():
                    pred_next = model(past_seq)
                    diff = torch.abs(target_next - pred_next)
                    mse = torch.mean((target_next - pred_next) ** 2).item()

                cur_time = frame_idx / fps
                timestamps.append(cur_time)
                anomaly_scores.append(mse)
                frame_indices.append(frame_idx)

                # Track nominal vs anomaly visual samples
                if mse > max_anomaly_score and cur_time > 10.0:
                    max_anomaly_score = mse
                    sample_anomaly = {
                        'actual': rgb_frame,
                        'pred': pred_next.squeeze(0).permute(1, 2, 0).cpu().numpy(),
                        'diff': diff.squeeze(0).mean(dim=0).cpu().numpy(),
                        'score': mse,
                        'time': cur_time,
                        'frame': frame_idx
                    }

                if sample_nominal is None and cur_time > 15.0 and mse < 0.015:
                    sample_nominal = {
                        'actual': rgb_frame,
                        'pred': pred_next.squeeze(0).permute(1, 2, 0).cpu().numpy(),
                        'diff': diff.squeeze(0).mean(dim=0).cpu().numpy(),
                        'score': mse,
                        'time': cur_time,
                        'frame': frame_idx
                    }

            buffer_frames.pop(0)

        frame_idx += 1
        if frame_idx % 600 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames ({frame_idx/fps:.1f}s) | Speed: {frame_idx/(time.time() - t0):.1f} fps")

    cap.release()

    timestamps = np.array(timestamps)
    anomaly_scores = np.array(anomaly_scores)

    # Statistical Thresholds
    mean_score = float(np.mean(anomaly_scores))
    std_score = float(np.std(anomaly_scores))
    threshold_3sigma = mean_score + 2.5 * std_score

    anomalous_events = timestamps[anomaly_scores > threshold_3sigma]
    print(f"\n[Results] Evaluated {len(anomaly_scores)} chunks. Mean MSE: {mean_score:.5f} | Std: {std_score:.5f} | Threshold (2.5σ): {threshold_3sigma:.5f}")
    print(f"Detected {len(anomalous_events)} anomaly events exceeding threshold!")

    # ---------------------------------------------------------
    # Visual Multi-Panel Anomaly Report Plot
    # ---------------------------------------------------------
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 11))
    fig.suptitle(f"AutoTwin-AI v2.0: Spatiotemporal ConvLSTM Video Anomaly Detection\nVideo: {video_path.name} | Model: ConvLSTM2d Next-Frame Prediction (T={seq_len})", fontsize=13, fontweight='bold', color='#00f0ff', y=0.98)

    # Timeline Plot (Top Half)
    ax_timeline = plt.subplot2grid((3, 3), (0, 0), colspan=3)
    ax_timeline.plot(timestamps, anomaly_scores, color='#00f0ff', linewidth=1.2, label='ConvLSTM Next-Frame MSE Score')
    ax_timeline.axhline(threshold_3sigma, color='#ef4444', linestyle='--', linewidth=1.2, label=f'Anomaly Threshold (2.5σ: {threshold_3sigma:.4f})')
    ax_timeline.fill_between(timestamps, threshold_3sigma, anomaly_scores, where=(anomaly_scores > threshold_3sigma), color='#ef4444', alpha=0.4, label='Spatiotemporal Anomalies (Spatter / Arc Bursts)')

    if max_anomaly_score > 0:
        peak_t = timestamps[np.argmax(anomaly_scores)]
        ax_timeline.scatter([peak_t], [max_anomaly_score], color='#ef4444', s=80, zorder=5)
        ax_timeline.annotate(f"Peak Anomaly: {max_anomaly_score:.4f}\n@ {peak_t:.1f}s", xy=(peak_t, max_anomaly_score),
                             xytext=(peak_t + 5, max_anomaly_score * 0.85),
                             arrowprops=dict(facecolor='#ef4444', shrink=0.05, width=1.2, headwidth=6),
                             fontsize=9, color='#ef4444', fontweight='bold')

    ax_timeline.set_ylabel("Spatiotemporal MSE", fontsize=10, color='#00f0ff', fontweight='bold')
    ax_timeline.set_xlabel("Video Timestamp (seconds)", fontsize=10, color='#94a3b8')
    ax_timeline.grid(True, linestyle='--', alpha=0.3)
    ax_timeline.legend(loc='upper right', framealpha=0.7)

    # Frame Visualizations: Nominal (Row 1) vs Anomaly (Row 2)
    def plot_frame_triplet(sample, row_idx, label_prefix):
        if sample is None:
            return
        # Actual
        ax1 = plt.subplot2grid((3, 3), (row_idx, 0))
        ax1.imshow(sample['actual'])
        ax1.set_title(f"{label_prefix}: Actual Frame\n(t={sample['time']:.1f}s, #{sample['frame']})", fontsize=8, color='#38bdf8', fontweight='bold')
        ax1.axis('off')

        # Predicted
        ax2 = plt.subplot2grid((3, 3), (row_idx, 1))
        ax2.imshow(np.clip(sample['pred'], 0, 1))
        ax2.set_title(f"ConvLSTM Predicted Next Frame\n(MSE: {sample['score']:.5f})", fontsize=8, color='#10b981', fontweight='bold')
        ax2.axis('off')

        # Error Heatmap
        ax3 = plt.subplot2grid((3, 3), (row_idx, 2))
        im = ax3.imshow(sample['diff'], cmap='inferno')
        ax3.set_title("Prediction Error Heatmap |Actual - Pred|\n(Spatter & Velocity Residuals)", fontsize=8, color='#f59e0b', fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    plot_frame_triplet(sample_nominal, 1, "Nominal Welding")
    plot_frame_triplet(sample_anomaly, 2, "Anomalous Event (Spatter Burst)")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plot_path = output_dir / f"convlstm_anomaly_report_{video_path.stem}.png"
    plt.savefig(plot_path, dpi=180, facecolor='#0a0d14')
    plt.close()
    print(f"[AutoTwin ConvLSTM] Saved anomaly report plot to: {plot_path}")

    # Save JSON summary
    report_json = {
        'video_name': video_path.name,
        'model_architecture': 'SpatiotemporalConvLSTMAutoencoder',
        'sequence_length_T': seq_len,
        'total_evaluated_chunks': len(anomaly_scores),
        'mean_spatiotemporal_mse': round(mean_score, 6),
        'std_spatiotemporal_mse': round(std_score, 6),
        'anomaly_threshold_2_5sigma': round(threshold_3sigma, 6),
        'anomalous_chunks_count': int(np.count_nonzero(anomaly_scores > threshold_3sigma)),
        'peak_anomaly': {
            'mse_score': round(max_anomaly_score, 6),
            'timestamp_sec': round(float(timestamps[np.argmax(anomaly_scores)]), 2) if len(anomaly_scores) > 0 else 0,
            'frame_index': int(frame_indices[np.argmax(anomaly_scores)]) if len(anomaly_scores) > 0 else 0
        }
    }
    json_path = output_dir / f"convlstm_anomaly_report_{video_path.stem}.json"
    with open(json_path, 'w', encoding='utf-8') as fp:
        json.dump(report_json, fp, indent=2)
    print(f"[AutoTwin ConvLSTM] Saved anomaly JSON report to: {json_path}")

    return report_json


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    default_video = str(project_root / "data" / "video_20260908_170143.mp4")
    default_ckpt = str(project_root / "models" / "convlstm_best.pth")

    parser = argparse.ArgumentParser(description="AutoTwin-AI ConvLSTM Video Training & Anomaly Detection")
    parser.add_argument("--video", type=str, default=default_video, help="Path to video file")
    parser.add_argument("--train", action="store_true", help="Train ConvLSTM model")
    parser.add_argument("--eval", action="store_true", help="Evaluate video with trained ConvLSTM model")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--seq_len", type=int, default=8, help="Sequence length T")
    parser.add_argument("--model-path", type=str, default=default_ckpt, help="Path to model checkpoint")
    args = parser.parse_args()

    if args.train:
        train_conv_lstm(args.video, epochs=args.epochs, seq_len=args.seq_len)

    if args.eval:
        evaluate_video_anomalies(args.video, model_path=args.model_path, seq_len=args.seq_len)
