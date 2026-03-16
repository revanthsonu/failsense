"""
src/models/train_autoencoder.py

Training loop for the sensor autoencoder.
Trains on healthy (non-near-failure) windows only — the model learns
what "normal" looks like, so reconstruction error on anomalous windows
is high by construction.

Run:
    python src/models/train_autoencoder.py \
        --data_dir data/processed \
        --epochs 50 \
        --batch_size 256 \
        --lr 1e-3
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
import mlflow

from autoencoder import SensorAutoencoder, AutoencoderConfig, reconstruction_loss


def load_data(data_dir: str, healthy_only: bool = True):
    """
    Load preprocessed windows. If healthy_only=True, train only on
    windows with label=0 (not near failure). This is the standard
    approach for reconstruction-based anomaly detection — the model
    learns normal patterns, so failures produce high reconstruction error.
    """
    data_dir = Path(data_dir)
    X   = np.load(data_dir / "X_train.npy")     # (N, 420)
    lbl = np.load(data_dir / "lbl_train.npy")   # (N,) — 1=near_failure

    if healthy_only:
        X = X[lbl == 0]
        print(f"Healthy windows: {len(X)} / {len(lbl)} total")

    # Reshape to (N, seq_len, n_sensors)
    cfg = AutoencoderConfig()
    X = X.reshape(-1, cfg.seq_len, cfg.n_sensors).astype(np.float32)
    return torch.tensor(X)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            _, x_hat = model(x)
            total_loss += reconstruction_loss(x, x_hat).item()
    return total_loss / len(loader)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    X = load_data(args.data_dir, healthy_only=True)
    dataset = TensorDataset(X)

    # 90/10 train/val split
    n_val   = int(0.1 * len(dataset))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2)

    # Model
    cfg   = AutoencoderConfig()
    model = SensorAutoencoder(cfg).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-5
    )

    # MLflow tracking
    mlflow.set_experiment("failsense-autoencoder")
    with mlflow.start_run():
        mlflow.log_params({
            "epochs":      args.epochs,
            "batch_size":  args.batch_size,
            "lr":          args.lr,
            "latent_dim":  cfg.latent_dim,
            "seq_len":     cfg.seq_len,
            "n_sensors":   cfg.n_sensors,
            "n_train":     n_train,
            "n_val":       n_val,
        })

        best_val = float("inf")
        out_dir  = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, args.epochs + 1):
            # Train
            model.train()
            train_loss = 0.0
            for (x,) in train_loader:
                x = x.to(device)
                opt.zero_grad()
                _, x_hat = model(x)
                loss = reconstruction_loss(x, x_hat)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss    = evaluate(model, val_loader, device)
            scheduler.step()

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "lr":         scheduler.get_last_lr()[0],
            }, step=epoch)

            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d} | train {train_loss:.5f} | val {val_loss:.5f}")

            # Save best checkpoint
            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "epoch":      epoch,
                    "model_state": model.state_dict(),
                    "val_loss":   val_loss,
                    "config":     cfg,
                }, out_dir / "autoencoder_best.pth")

        print(f"\nBest val loss: {best_val:.5f}")
        mlflow.log_metric("best_val_loss", best_val)
        mlflow.log_artifact(str(out_dir / "autoencoder_best.pth"))

    # Quick anomaly score sanity check on held-out windows
    print("\nAnomaly score check (healthy vs near-failure):")
    _check_anomaly_separation(model, args.data_dir, cfg, device)


def _check_anomaly_separation(model, data_dir, cfg, device):
    """
    Sanity check: reconstruction error should be noticeably higher
    on near-failure windows than healthy ones.
    If this fails, the autoencoder hasn't learned meaningful normal patterns.
    """
    data_dir = Path(data_dir)
    X   = np.load(data_dir / "X_train.npy").reshape(-1, cfg.seq_len, cfg.n_sensors).astype(np.float32)
    lbl = np.load(data_dir / "lbl_train.npy")

    X_healthy = torch.tensor(X[lbl == 0][:500])
    X_failure = torch.tensor(X[lbl == 1][:500])

    model.eval()
    with torch.no_grad():
        scores_h = model.anomaly_score(X_healthy.to(device)).cpu().numpy()
        scores_f = model.anomaly_score(X_failure.to(device)).cpu().numpy()

    print(f"  Healthy  — mean: {scores_h.mean():.5f}  std: {scores_h.std():.5f}")
    print(f"  Failure  — mean: {scores_f.mean():.5f}  std: {scores_f.std():.5f}")

    separation = scores_f.mean() / (scores_h.mean() + 1e-9)
    print(f"  Separation ratio: {separation:.2f}x  (target: >2.0x)")
    if separation < 1.5:
        print("  [WARN] Poor separation — consider more epochs or architecture changes")
    else:
        print("  [OK] Autoencoder is learning anomalous patterns")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  default="data/processed")
    parser.add_argument("--out_dir",   default="checkpoints")
    parser.add_argument("--epochs",    default=50,   type=int)
    parser.add_argument("--batch_size",default=256,  type=int)
    parser.add_argument("--lr",        default=1e-3, type=float)
    args = parser.parse_args()
    train(args)
