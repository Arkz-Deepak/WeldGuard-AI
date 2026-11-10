"""
AutoTwin-AI v2.0: 5D Spatiotemporal Video Dataset Pipeline
Loads sliding-window chunks from real factory welding videos into 5D PyTorch tensors:
  Input:  [Batch, Sequence_T, Channels, Height, Width]
  Target: [Batch, Channels, Height, Width] (Next-Frame Target)
"""

import os
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class WeldingVideoDataset(Dataset):
    """
    Sliding window spatiotemporal dataset from video files.
    Extracts sequences of length (seq_len + 1) where:
      - past_frames = frames[0:seq_len]  --> Model Input: [T, 3, H, W]
      - target_frame = frames[seq_len]    --> Target Next Frame: [3, H, W]
    """
    def __init__(self, video_path, seq_len=8, stride=4, img_size=(128, 128), max_sequences=None, transform=None):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.seq_len = seq_len
        self.stride = stride
        self.img_size = img_size
        self.transform = transform

        # Open video and index total frames
        cap = cv2.VideoCapture(str(self.video_path))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Build sequence start indices
        self.indices = []
        window_size = seq_len + 1  # T past frames + 1 next frame
        for start_idx in range(0, self.total_frames - window_size, stride):
            self.indices.append(start_idx)
            if max_sequences and len(self.indices) >= max_sequences:
                break

        print(f"[WeldingVideoDataset] Loaded {len(self.indices)} sliding window chunks from {self.video_path.name} (T={seq_len}, Stride={stride}, Target Res={img_size[0]}x{img_size[1]})")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        window_size = self.seq_len + 1

        cap = cv2.VideoCapture(str(self.video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        frames = []
        for _ in range(window_size):
            ret, frame = cap.read()
            if not ret:
                # Handle edge cases by duplicating last frame
                if len(frames) > 0:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8))
                continue

            # Resize to target dimension
            frame = cv2.resize(frame, self.img_size)
            # Convert BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        # Convert to float tensors normalized to [0, 1]
        # Shape per frame: [C, H, W]
        tensor_frames = [torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in frames]

        # Stack past sequence: [T, 3, H, W]
        past_seq = torch.stack(tensor_frames[:self.seq_len], dim=0)

        # Target next frame: [3, H, W]
        target_next = tensor_frames[self.seq_len]

        return past_seq, target_next


def get_video_dataloader(video_path, seq_len=8, stride=4, img_size=(128, 128), batch_size=4, max_sequences=None, shuffle=True):
    dataset = WeldingVideoDataset(
        video_path=video_path,
        seq_len=seq_len,
        stride=stride,
        img_size=img_size,
        max_sequences=max_sequences
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return dataloader


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    test_video = str(project_root / "data" / "video_20260908_170143.mp4")
    if os.path.exists(test_video):
        print("Testing WeldingVideoDataset...")
        loader = get_video_dataloader(test_video, seq_len=8, stride=30, batch_size=2, max_sequences=6)
        for past_seq, next_frame in loader:
            print(f"Batch past_seq shape: {past_seq.shape}")  # [2, 8, 3, 128, 128]
            print(f"Batch next_frame shape: {next_frame.shape}")  # [2, 3, 128, 128]
            break
        print("WeldingVideoDataset test passed successfully!")
