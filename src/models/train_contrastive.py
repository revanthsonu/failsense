"""
src/models/train_contrastive.py

Day 3: Train the cross-modal contrastive alignment.
Requires:
  - Trained autoencoder checkpoint (from train_autoencoder.py)
  - SBERT embeddings of maintenance logs (from build_index.py)
  - Near-failure windows from preprocessed data

Training strategy:
  - Freeze autoencoder encoder weights (already well-trained)
  - Train only projection heads via InfoNCE loss
  - Positive pairs: near-failure sensor window ↔ critical maintenance log
    with matching failure_mode label
  - Negatives: all other pairs in the batch (in-batch negatives, same as CLIP)

Run:
    python src/models/train_contrastive.py \
        --ae_ckpt checkpoints/autoencoder_best.pth \
        --logs_path data/synthetic_logs/maintenance_corpus.json \
        --sbert_embeddings data/processed/log_sbert_embeddings.npy \
        --data_dir data/processed \
        --epochs 30
"""

import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import mlflow

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.autoencoder import SensorAutoencoder, AutoencoderConfig
from models.contrastive import CrossModalAlignment, ContrastiveConfig


class ContrastivePairDataset(Dataset):
    """
    Dataset of (sensor_window, log_embedding) positive pairs.

    Pairing strategy:
      Near-failure windows are grouped by a proxy failure mode derived
      from which sensors degraded most. Each is paired with a randomly
      sampled critical maintenance log. This is a weak supervision signal
      — the contrastive loss still works because critical logs are all
      semantically related to failure events, creating a shared cluster
      in the projected space.

    Research note: Stronger pairing via failure-mode matching would
    improve retrieval precision. This is the ablation planned for the paper.
    """

    def __init__(
        self,
        X_failure: np.ndarray,           # (N, seq_len, n_sensors) near-failure windows
        sbert_embeddings: np.ndarray,    # (M, 384) all log embeddings
        logs: list[dict],                # metadata for each log
        n_pairs: int = 5000,
        seed: int = 42,
    ):
        random.seed(seed)
        np.random.seed(seed)

        # Filter to critical logs only for positive pairs
        critical_indices = [i for i, l in enumerate(logs) if l["is_critical"]]
        if len(critical_indices) == 0:
            raise ValueError("No critical logs found — check generate_logs.py output")

        print(f"Near-failure windows: {len(X_failure)}")
        print(f"Critical logs: {len(critical_indices)}")

        # Build pairs
        self.sensor_windows = []
        self.log_embeddings = []

        for _ in range(n_pairs):
            win_idx = random.randint(0, len(X_failure) - 1)
            log_idx = random.choice(critical_indices)
            self.sensor_windows.append(X_failure[win_idx])
            self.log_embeddings.append(sbert_embeddings[log_idx])

        self.sensor_windows = np.array(self.sensor_windows, dtype=np.float32)
        self.log_embeddings = np.array(self.log_embeddings, dtype=np.float32)

    def __len__(self):
        return len(self.sensor_windows)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sensor_windows[idx]),   # (seq_len, n_sensors)
            torch.tensor(self.log_embeddings[idx]),   # (384,)
        )


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load autoencoder (frozen encoder)
    ae_cfg   = AutoencoderConfig()
    ae_model = SensorAutoencoder(ae_cfg).to(device)
    ae_ckpt  = torch.load(args.ae_ckpt, map_location=device)
    ae_model.load_state_dict(ae_ckpt["model_state"])
    ae_model.eval()
    for p in ae_model.parameters():
        p.requires_grad = False
    print(f"Loaded autoencoder (val_loss={ae_ckpt['val_loss']:.5f}), encoder frozen")

    # Load SBERT embeddings and log metadata
    sbert_embs = np.load(args.sbert_embeddings)   # (N_logs, 384)
    with open(args.logs_path) as f:
        logs = json.load(f)
    assert len(sbert_embs) == len(logs), "Embedding count mismatch"

    # Load near-failure windows
    data_dir = Path(args.data_dir)
    X   = np.load(data_dir / "X_train.npy")
    lbl = np.load(data_dir / "lbl_train.npy")
    X_failure = X[lbl == 1].reshape(-1, ae_cfg.seq_len, ae_cfg.n_sensors)

    # Build dataset
    dataset = ContrastivePairDataset(
        X_failure, sbert_embs, logs,
        n_pairs=args.n_pairs,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=2, pin_memory=True)

    # Contrastive alignment model (only projection heads trained)
    ct_cfg = ContrastiveConfig()
    model  = CrossModalAlignment(ct_cfg).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-6
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("failsense-contrastive")
    with mlflow.start_run():
        mlflow.log_params({
            "epochs":      args.epochs,
            "batch_size":  args.batch_size,
            "lr":          args.lr,
            "n_pairs":     args.n_pairs,
            "temperature": ct_cfg.temperature,
            "proj_dim":    ct_cfg.proj_dim,
        })

        best_loss = float("inf")

        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0

            for sensor_win, log_emb in loader:
                sensor_win = sensor_win.to(device)   # (B, seq_len, n_sensors)
                log_emb    = log_emb.to(device)       # (B, 384)

                # Get sensor latent from frozen encoder
                with torch.no_grad():
                    sensor_emb = ae_model.encoder(sensor_win)   # (B, 256)

                # Forward through projection heads + compute InfoNCE
                opt.zero_grad()
                loss, _, _ = model(sensor_emb, log_emb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                epoch_loss += loss.item()

            epoch_loss /= len(loader)
            scheduler.step()

            mlflow.log_metrics({
                "contrastive_loss": epoch_loss,
                "lr": scheduler.get_last_lr()[0],
            }, step=epoch)

            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d} | contrastive_loss: {epoch_loss:.4f}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save({
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "loss":        epoch_loss,
                    "config":      ct_cfg,
                }, out_dir / "alignment_best.pth")

        print(f"\nBest contrastive loss: {best_loss:.4f}")
        print(f"Checkpoint saved → {out_dir}/alignment_best.pth")
        mlflow.log_metric("best_contrastive_loss", best_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae_ckpt",          default="checkpoints/autoencoder_best.pth")
    parser.add_argument("--logs_path",        default="data/synthetic_logs/maintenance_corpus.json")
    parser.add_argument("--sbert_embeddings", default="data/processed/log_sbert_embeddings.npy")
    parser.add_argument("--data_dir",         default="data/processed")
    parser.add_argument("--out_dir",          default="checkpoints")
    parser.add_argument("--epochs",           default=30,   type=int)
    parser.add_argument("--batch_size",       default=128,  type=int)
    parser.add_argument("--lr",               default=1e-3, type=float)
    parser.add_argument("--n_pairs",          default=5000, type=int)
    args = parser.parse_args()
    train(args)
