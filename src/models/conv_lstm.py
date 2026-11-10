"""
AutoTwin-AI v2.0: Spatiotemporal ConvLSTM Architecture
Implements:
  1. ConvLSTMCell: 2D Convolutional LSTM cell preserving spatial topology
  2. ConvLSTM: Multi-step spatiotemporal recurrent layer for 5D tensors [B, T, C, H, W]
  3. SpatiotemporalConvLSTMAutoencoder: CNN Encoder + ConvLSTM Core + Transposed CNN Decoder
     for Next-Frame Temporal Prediction & Anomaly Detection.
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    2D Convolutional LSTM Cell.
    Replaces matrix multiplications with 2D convolutions to model
    both temporal dynamics and 2D spatial layout.
    """
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super(ConvLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # Combined convolution for all 4 gates (input, forget, cell candidate, output)
        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(self, x, h_prev, c_prev):
        # x: [B, C_in, H, W]
        # h_prev: [B, C_hidden, H, W]
        # c_prev: [B, C_hidden, H, W]
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
    """
    Recurrent sequence layer unrolling ConvLSTMCell over T timesteps.
    Input:  [B, T, C_in, H, W]
    Output: [B, T, C_hidden, H, W] and (h_last, c_last)
    """
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super(ConvLSTM, self).__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size)

    def forward(self, x_seq, hidden_state=None):
        # x_seq: [B, T, C_in, H, W]
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
    """
    Complete Spatiotemporal Model for Welding Video Anomaly Detection:
      1. Spatial CNN Encoder: compresses each frame from [B, 3, H, W] -> [B, 64, H/4, W/4]
      2. ConvLSTM Core: models temporal evolution across T frames -> [B, 64, H/4, W/4]
      3. Spatial CNN Decoder: predicts next frame [B, 3, H, W]
    """
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

        # 2. ConvLSTM Recurrent Core
        self.conv_lstm = ConvLSTM(
            in_channels=latent_dim,
            hidden_channels=latent_dim,
            kernel_size=3
        )

        # 3. 2D Transposed CNN Decoder (Next-Frame Predictor)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x_seq):
        # x_seq: [B, T, 3, H, W]
        batch_size, seq_len, c, h, w = x_seq.size()

        # Reshape to process all frames through spatial CNN encoder
        x_reshaped = x_seq.view(batch_size * seq_len, c, h, w)
        encoded_feats = self.encoder(x_reshaped)  # [B*T, latent_dim, H/4, W/4]

        # Reshape back to sequence for ConvLSTM
        _, feat_c, feat_h, feat_w = encoded_feats.size()
        feat_seq = encoded_feats.view(batch_size, seq_len, feat_c, feat_h, feat_w)

        # Propagate through ConvLSTM
        _, (h_last, _) = self.conv_lstm(feat_seq)  # h_last: [B, latent_dim, H/4, W/4]

        # Decode the final spatiotemporal state to predict the NEXT frame
        next_frame_pred = self.decoder(h_last)  # [B, 3, H, W]

        return next_frame_pred

    def compute_anomaly_score(self, x_seq, x_next_actual):
        """
        Inference utility:
        Given past sequence x_seq and ground truth x_next_actual,
        returns:
          - mse_anomaly_score: scalar float
          - residual_heatmap: numpy 2D array [H, W] of absolute error
          - predicted_frame: numpy RGB image [H, W, 3]
        """
        self.eval()
        with torch.no_grad():
            pred = self.forward(x_seq)
            diff = torch.abs(x_next_actual - pred)
            mse = torch.mean((x_next_actual - pred) ** 2).item()

            heatmap = torch.mean(diff, dim=1).squeeze(0).cpu().numpy()
            pred_img = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()

        return mse, heatmap, pred_img


if __name__ == "__main__":
    # Smoke test architecture
    print("Testing SpatiotemporalConvLSTMAutoencoder...")
    model = SpatiotemporalConvLSTMAutoencoder(in_channels=3, latent_dim=64)
    dummy_seq = torch.randn(2, 8, 3, 128, 128)  # Batch=2, T=8 frames, 128x128
    pred_next = model(dummy_seq)
    print(f"Input sequence shape: {dummy_seq.shape}")
    print(f"Predicted next frame shape: {pred_next.shape}")
    assert pred_next.shape == (2, 3, 128, 128), "Shape mismatch in next frame prediction!"
    print("ConvLSTM architecture test passed successfully!")
