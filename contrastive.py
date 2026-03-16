"""
src/models/contrastive.py

Cross-modal contrastive alignment — the core research contribution of FailSense.

Two projection heads map sensor embeddings and text embeddings into a shared
256-d latent space. InfoNCE loss (same objective as CLIP) pulls matching
(sensor_window, maintenance_log) pairs together and pushes non-matches apart.

After training:
    sensor_proj(autoencoder.encoder(window)) ≈ text_proj(sbert(log))
    for semantically matching (window, log) pairs.

This enables cross-modal retrieval: query FAISS with a sensor embedding,
retrieve relevant maintenance text — without any text query.

Reference: Oord et al. "Representation Learning with Contrastive
Predictive Coding" (2018). He et al. "Momentum Contrast for Unsupervised
Visual Representation Learning" (2020).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ContrastiveConfig:
    sensor_dim: int  = 256      # autoencoder latent dim
    text_dim: int    = 384      # SBERT all-MiniLM-L6-v2 output dim
    proj_dim: int    = 256      # shared projection space dim
    temperature: float = 0.07  # InfoNCE temperature τ (ablate: 0.07, 0.1, 0.2)
    hidden_dim: int  = 512      # projection head hidden layer


class ProjectionHead(nn.Module):
    """
    2-layer MLP that projects modality-specific embeddings into shared space.

    Architecture follows SimCLR (Chen et al. 2020):
    input → Linear → BN → ReLU → Linear → L2-normalize

    L2 normalization at output makes cosine similarity = dot product,
    which is computationally convenient and empirically beneficial.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)
        Returns:
            z: (batch, output_dim)  L2-normalized
        """
        return F.normalize(self.net(x), dim=-1)


class CrossModalAlignment(nn.Module):
    """
    Holds both projection heads and computes InfoNCE loss.

    sensor_proj: sensor latent (256-d) → shared space (256-d)
    text_proj:   SBERT embedding (384-d) → shared space (256-d)
    """

    def __init__(self, cfg: ContrastiveConfig = None):
        super().__init__()
        cfg = cfg or ContrastiveConfig()
        self.cfg = cfg

        self.sensor_proj = ProjectionHead(
            cfg.sensor_dim, cfg.hidden_dim, cfg.proj_dim
        )
        self.text_proj = ProjectionHead(
            cfg.text_dim, cfg.hidden_dim, cfg.proj_dim
        )
        self.temperature = nn.Parameter(
            torch.tensor(cfg.temperature), requires_grad=False
        )

    def forward(
        self,
        sensor_emb: torch.Tensor,   # (N, sensor_dim)
        text_emb:   torch.Tensor,   # (N, text_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project both modalities and compute InfoNCE loss.

        Assumes sensor_emb[i] and text_emb[i] are a positive pair
        (matching sensor window ↔ maintenance log). All other
        combinations within the batch are treated as negatives.

        Returns:
            loss:       scalar InfoNCE loss
            s_proj:     (N, proj_dim) sensor projections
            t_proj:     (N, proj_dim) text projections
        """
        s_proj = self.sensor_proj(sensor_emb)   # (N, proj_dim)
        t_proj = self.text_proj(text_emb)       # (N, proj_dim)

        loss = info_nce_loss(s_proj, t_proj, self.temperature)
        return loss, s_proj, t_proj

    def project_sensor(self, sensor_emb: torch.Tensor) -> torch.Tensor:
        """Project sensor embedding into shared space (inference)."""
        return F.normalize(self.sensor_proj(sensor_emb), dim=-1)

    def project_text(self, text_emb: torch.Tensor) -> torch.Tensor:
        """Project text embedding into shared space (inference)."""
        return F.normalize(self.text_proj(text_emb), dim=-1)


def info_nce_loss(
    anchors: torch.Tensor,      # (N, D)  sensor projections
    positives: torch.Tensor,    # (N, D)  text projections
    temperature: torch.Tensor | float = 0.07,
) -> torch.Tensor:
    """
    Symmetric InfoNCE (NT-Xent) loss.

    For a batch of N pairs, the loss is the average of:
      - sensor→text direction: for each sensor, classify its matching text
        among all N texts
      - text→sensor direction: for each text, classify its matching sensor
        among all N sensors

    This symmetric formulation (used in CLIP) is more stable than
    one-directional InfoNCE.

    Mathematical form:
        L = -1/N * Σ log(exp(sim(s_i, t_i)/τ) / Σ_j exp(sim(s_i, t_j)/τ))

    Args:
        anchors:     L2-normalized sensor projections
        positives:   L2-normalized text projections
        temperature: τ controls sharpness of distribution (lower = sharper)

    Returns:
        scalar loss
    """
    N = anchors.shape[0]

    # Similarity matrix: (N, N) — all cross-modal pairs
    # anchors @ positives.T: sim[i,j] = cosine(sensor_i, text_j)
    sim_matrix = (anchors @ positives.T) / temperature

    # Ground truth: diagonal elements are positive pairs
    labels = torch.arange(N, device=anchors.device)

    # Cross-entropy in both directions
    loss_s2t = F.cross_entropy(sim_matrix,   labels)   # sensor → text
    loss_t2s = F.cross_entropy(sim_matrix.T, labels)   # text → sensor

    return (loss_s2t + loss_t2s) / 2.0


def build_training_pairs(
    near_failure_windows: list,     # CMAPSSWindow with is_near_failure=True
    maintenance_logs: list,          # dicts with 'failure_mode', 'text', 'embedding'
    n_pairs: int = 2000,
) -> tuple[list, list]:
    """
    Construct (sensor_window, maintenance_log) positive pairs for training.

    Pairing strategy:
        - A near-failure sensor window is matched to a maintenance log
          with the same failure_mode label
        - This requires failure_mode labels on both sides:
            · Sensor windows: CMAPSS provides RUL; we use heuristic
              failure mode assignment based on which sensors degraded most
            · Maintenance logs: generated with explicit failure_mode field

    Research note: In production, these pairs would come from actual
    maintenance events timestamped alongside sensor readings. For this
    benchmark, we use the CMAPSS degradation trajectory + generated logs
    as a controlled proxy.

    Args:
        near_failure_windows: windows close to engine failure
        maintenance_logs: corpus entries with failure_mode and text fields
        n_pairs: number of positive pairs to construct

    Returns:
        (sensor_pairs, log_pairs) — aligned lists of matching items
    """
    import random
    from collections import defaultdict

    # Group logs by failure mode for efficient lookup
    logs_by_mode = defaultdict(list)
    for log in maintenance_logs:
        logs_by_mode[log["failure_mode"]].append(log)

    sensor_pairs = []
    log_pairs    = []

    # Simplified: pair each near-failure window with a random
    # critical maintenance log. Replace with failure-mode-matched
    # pairing once sensor failure mode classification is added.
    critical_logs = [l for l in maintenance_logs if l["is_critical"]]

    for _ in range(n_pairs):
        window = random.choice(near_failure_windows)
        log    = random.choice(critical_logs)
        sensor_pairs.append(window)
        log_pairs.append(log)

    return sensor_pairs, log_pairs


if __name__ == "__main__":
    # Smoke test
    cfg   = ContrastiveConfig()
    model = CrossModalAlignment(cfg)

    sensor_emb = torch.randn(16, 256)   # batch of 16 sensor embeddings
    text_emb   = torch.randn(16, 384)   # batch of 16 SBERT embeddings

    loss, s_proj, t_proj = model(sensor_emb, text_emb)

    print(f"InfoNCE loss:         {loss.item():.4f}")
    print(f"Sensor proj shape:    {s_proj.shape}")
    print(f"Text proj shape:      {t_proj.shape}")
    print(f"Expected random loss: {torch.log(torch.tensor(16.0)).item():.4f}")
    print(f"  (random = log(N) = log(16) — loss should decrease from here)")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
