"""
src/models/autoencoder.py

Convolutional autoencoder for multivariate sensor time-series.
The bottleneck (latent) vector serves as the anomaly embedding
that feeds into the contrastive projection head.

Architecture decision (Research Log, Day 1):
    - 1D CNN encoder: captures local temporal patterns better than
      flat MLP for time-series data
    - Bottleneck dim 256: empirically chosen to balance expressiveness
      and alignment ease with SBERT's 384-d output
    - Reconstruction loss: MSE over all sensor channels
    - Anomaly score: mean reconstruction error per window
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class AutoencoderConfig:
    input_dim: int   = 420      # 30 steps × 14 sensors (flattened)
    seq_len: int     = 30
    n_sensors: int   = 14
    hidden_dims: tuple = (128, 64)
    latent_dim: int  = 256
    dropout: float   = 0.1


class SensorEncoder(nn.Module):
    """
    1D CNN encoder: (batch, n_sensors, seq_len) → (batch, latent_dim)

    Uses depthwise-separable convolutions to capture per-sensor
    temporal patterns before mixing across sensors.
    """

    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        self.cfg = cfg

        self.conv_block = nn.Sequential(
            # Layer 1: local patterns (kernel 3)
            nn.Conv1d(cfg.n_sensors, cfg.hidden_dims[0],
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(cfg.hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(cfg.dropout),

            # Layer 2: medium patterns (kernel 5)
            nn.Conv1d(cfg.hidden_dims[0], cfg.hidden_dims[1],
                      kernel_size=5, padding=2),
            nn.BatchNorm1d(cfg.hidden_dims[1]),
            nn.GELU(),

            # Downsample
            nn.AvgPool1d(kernel_size=2),  # seq_len → seq_len/2
        )

        # Compute flattened size after convolutions
        conv_out_len  = cfg.seq_len // 2
        conv_out_dim  = cfg.hidden_dims[1] * conv_out_len

        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, cfg.latent_dim * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.latent_dim * 2, cfg.latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_sensors)  — time-first convention
        Returns:
            z: (batch, latent_dim)
        """
        x = x.permute(0, 2, 1)          # → (batch, n_sensors, seq_len)
        x = self.conv_block(x)           # → (batch, hidden, seq_len/2)
        x = x.flatten(start_dim=1)       # → (batch, hidden * seq_len/2)
        z = self.fc(x)                   # → (batch, latent_dim)
        return z


class SensorDecoder(nn.Module):
    """
    Transpose CNN decoder: (batch, latent_dim) → (batch, seq_len, n_sensors)
    Mirrors the encoder for symmetric reconstruction.
    """

    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        self.cfg = cfg

        conv_out_len = cfg.seq_len // 2
        conv_out_dim = cfg.hidden_dims[1] * conv_out_len

        self.fc = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim * 2),
            nn.GELU(),
            nn.Linear(cfg.latent_dim * 2, conv_out_dim),
            nn.GELU(),
        )

        self.conv_out_len = conv_out_len

        self.deconv_block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(cfg.hidden_dims[1], cfg.hidden_dims[0],
                      kernel_size=5, padding=2),
            nn.BatchNorm1d(cfg.hidden_dims[0]),
            nn.GELU(),
            nn.Conv1d(cfg.hidden_dims[0], cfg.n_sensors,
                      kernel_size=3, padding=1),
            nn.Sigmoid(),              # sensors normalized to [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(-1, self.cfg.hidden_dims[1], self.conv_out_len)
        x = self.deconv_block(x)       # (batch, n_sensors, seq_len)
        x = x.permute(0, 2, 1)        # (batch, seq_len, n_sensors)
        return x


class SensorAutoencoder(nn.Module):
    """
    Full autoencoder. Training uses MSE reconstruction loss.
    At inference time, only the encoder is used to produce anomaly embeddings.

    Anomaly score = mean squared reconstruction error over all sensors and
    time steps. Higher score → more anomalous window.
    """

    def __init__(self, cfg: AutoencoderConfig = None):
        super().__init__()
        cfg = cfg or AutoencoderConfig()
        self.encoder = SensorEncoder(cfg)
        self.decoder = SensorDecoder(cfg)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            z:    latent embedding (batch, latent_dim)
            x_hat: reconstruction (batch, seq_len, n_sensors)
        """
        z     = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-sample reconstruction error. Used at inference to flag anomalies.

        Args:
            x: (batch, seq_len, n_sensors)
        Returns:
            scores: (batch,) — higher = more anomalous
        """
        with torch.no_grad():
            _, x_hat = self(x)
            scores = ((x - x_hat) ** 2).mean(dim=(1, 2))
        return scores


def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(x_hat, x)


if __name__ == "__main__":
    # Smoke test
    cfg   = AutoencoderConfig()
    model = SensorAutoencoder(cfg)

    dummy = torch.randn(8, 30, 14)   # batch=8, seq_len=30, n_sensors=14
    z, x_hat = model(dummy)

    print(f"Input shape:         {dummy.shape}")
    print(f"Latent shape:        {z.shape}")
    print(f"Reconstruction shape:{x_hat.shape}")
    print(f"Anomaly scores:      {model.anomaly_score(dummy)}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
