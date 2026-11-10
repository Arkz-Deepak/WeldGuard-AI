"""
WeldGuard-AI: Spatiotemporal ConvLSTM Training & Benchmark on Kaggle GPU
Trains ConvLSTM next-frame prediction autoencoder on real factory welding telemetry
for real-time in-situ defect and anomaly surveillance.
"""

import os
import sys
import time
import json
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------
# 1. Spatiotemporal ConvLSTM Architecture Definition
# ---------------------------------------------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super(ConvLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        return (
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        )


class ConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super(ConvLSTM, self).__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size)

    def forward(self, x_seq, hidden_state=None):
        batch_size, seq_len, _, height, width = x_seq.size()
        device = x_seq.device

        if hidden_state is None:
            h, c = self.cell.init_hidden(batch_size, height, width, device)
        else:
            h, c = hidden_state

        outputs = []
        for t in range(seq_len):
            x_t = x_seq[:, t, :, :, :]
            h, c = self.cell(x_t, h, c)
            outputs.append(h.unsqueeze(1))

        output_seq = torch.cat(outputs, dim=1)
        return output_seq, (h, c)


class SpatiotemporalConvLSTMAutoencoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=64):
        super(SpatiotemporalConvLSTMAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        # 1. 2D CNN Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 2. ConvLSTM Core
        self.conv_lstm = ConvLSTM(
            in_channels=latent_dim,
            hidden_channels=latent_dim,
            kernel_size=3
        )

        # 3. 2D Transposed CNN Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x_seq):
        batch_size, seq_len, c, h, w = x_seq.size()
        x_reshaped = x_seq.view(batch_size * seq_len, c, h, w)
        encoded_feats = self.encoder(x_reshaped)

        _, feat_c, feat_h, feat_w = encoded_feats.size()
        feat_seq = encoded_feats.view(batch_size, seq_len, feat_c, feat_h, feat_w)
        _, (h_last, _) = self.conv_lstm(feat_seq)

        next_frame_pred = self.decoder(h_last)
        return next_frame_pred


# ---------------------------------------------------------------------
# 2. Memory-Efficient Fast NPZ Dataset
# ---------------------------------------------------------------------
class PackedWeldingDataset(Dataset):
    def __init__(self, npz_paths, seq_len=8, stride=2, max_samples=None):
        self.seq_len = seq_len
        self.sequences = []

        for p in npz_paths:
            path = Path(p)
            if not path.exists():
                print(f"Warning: {path} not found, skipping...")
                continue
            data = np.load(path)
            frames = data['frames']  # [N, H, W, 3], uint8
            num_frames = len(frames)
            window = seq_len + 1

            for s in range(0, num_frames - window, stride):
                # Store (frame_array_reference, start_idx)
                self.sequences.append((frames, s))
                if max_samples and len(self.sequences) >= max_samples:
                    break

        print(f"[Dataset] Indexed {len(self.sequences)} sliding window samples across {len(npz_paths)} archive(s)")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        frames, start_idx = self.sequences[idx]
        window = frames[start_idx : start_idx + self.seq_len + 1]  # [T+1, H, W, 3]

        # Convert to float tensor [T+1, 3, H, W] in [0, 1]
        t = torch.from_numpy(window).permute(0, 3, 1, 2).float() / 255.0
        past_seq = t[:self.seq_len]      # [T, 3, H, W]
        target_next = t[self.seq_len]    # [3, H, W]
        return past_seq, target_next


# ---------------------------------------------------------------------
# 3. Main Training & Evaluation Loop
# ---------------------------------------------------------------------
def find_data_paths():
    """Locate datasets either on Kaggle or local environment."""
    paths = []
    if os.path.exists("/kaggle/input"):
        print(f"[Debug] /kaggle/input top-level dirs: {os.listdir('/kaggle/input')}")
        for root, dirs, files in os.walk("/kaggle/input"):
            for f in files:
                if f.endswith(".npz"):
                    full_p = os.path.join(root, f)
                    paths.append(full_p)
                    print(f"  --> Found data archive: {full_p}")
    if paths:
        return paths

    # Prioritize dedicated compact moving torch archives
    compact_paths = glob.glob("kaggle_dataset_compact/*.npz")
    if compact_paths:
        return sorted(compact_paths)
    for root, dirs, files in os.walk("."):
        for f in files:
            if f.endswith(".npz") and "compact" in f:
                paths.append(os.path.join(root, f))
    return sorted(paths)


def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("WeldGuard-AI: Spatiotemporal ConvLSTM Training Engine")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Allocated VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("=" * 70)

    npz_paths = find_data_paths()
    print(f"Discovered data archives: {npz_paths}")
    if not npz_paths:
        raise FileNotFoundError("No .npz archives found in /kaggle/input or local directory!")

    output_dir = Path("/kaggle/working") if os.path.exists("/kaggle") else Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparameters
    seq_len = 8
    stride = 2
    batch_size = 16 if torch.cuda.is_available() else 4
    num_epochs = 12
    learning_rate = 1e-3

    # Split into train and val
    full_dataset = PackedWeldingDataset(npz_paths, seq_len=seq_len, stride=stride)
    total_len = len(full_dataset)
    train_len = int(0.85 * total_len)
    val_len = total_len - train_len
    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2 if torch.cuda.is_available() else 0, pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2 if torch.cuda.is_available() else 0)

    print(f"Dataset split: {train_len} train samples, {val_len} validation samples.")

    model = SpatiotemporalConvLSTMAutoencoder(in_channels=3, latent_dim=64).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_model_path = output_dir / "convlstm_best.pth"

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        n_train = 0

        for past_seq, target_next in train_loader:
            past_seq = past_seq.to(device, non_blocking=True)
            target_next = target_next.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(past_seq)
            loss = criterion(pred, target_next)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(past_seq)
            n_train += len(past_seq)

        scheduler.step()
        avg_train_loss = train_loss / max(1, n_train)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for past_seq, target_next in val_loader:
                past_seq = past_seq.to(device, non_blocking=True)
                target_next = target_next.to(device, non_blocking=True)
                pred = model(past_seq)
                loss = criterion(pred, target_next)
                val_loss += loss.item() * len(past_seq)
                n_val += len(past_seq)

        avg_val_loss = val_loss / max(1, n_val)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        elapsed = time.time() - start_time
        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] | Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f} | Time: {elapsed:.1f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'seq_len': seq_len,
                'latent_dim': 64
            }, best_model_path)
            print(f"  --> Saved new best checkpoint to {best_model_path.name} (Val Loss: {best_val_loss:.6f})")

    # Save final model
    final_path = output_dir / "convlstm_final.pth"
    torch.save(model.state_dict(), final_path)

    # Plot loss curves
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, num_epochs + 1), history['train_loss'], label='Train MSE Loss', color='#00f0ff', linewidth=2)
    plt.plot(range(1, num_epochs + 1), history['val_loss'], label='Val MSE Loss', color='#ef4444', linewidth=2)
    plt.title("WeldGuard-AI: ConvLSTM Next-Frame Prediction Training Curves", fontsize=12, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Next-Frame MSE Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / "training_loss_curve.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    # Save training report JSON
    report = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'best_val_loss': round(float(best_val_loss), 6),
        'final_train_loss': round(float(history['train_loss'][-1]), 6),
        'final_val_loss': round(float(history['val_loss'][-1]), 6),
        'total_training_time_sec': round(time.time() - start_time, 2)
    }
    with open(output_dir / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Training Complete! Best Val MSE: {best_val_loss:.6f}")
    print(f"Saved artifacts in: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    run_training()
