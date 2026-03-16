"""
eval/evaluate_retrieval.py

Day 4: Evaluate cross-modal retrieval quality.

Compares four retrieval strategies:
  1. Random baseline
  2. Text-only RAG (SBERT on sensor description text)
  3. Sensor kNN (raw autoencoder latent, no alignment projection)
  4. FailSense (aligned cross-modal projection → FAISS)

Metric: Precision@k
    A retrieved log is "correct" if its failure_mode matches the
    ground-truth failure mode assigned to the query sensor window.

    P@k = (# correct in top-k) / k

Note on ground truth:
    CMAPSS doesn't label failure modes per window. We use a proxy:
    each near-failure window's failure mode is derived from which
    sensors degraded most (top degrading sensor → mapped to failure mode).
    This is imperfect but consistent across all methods being compared,
    so relative rankings are valid.

Run:
    python eval/evaluate_retrieval.py \
        --ae_ckpt checkpoints/autoencoder_best.pth \
        --alignment_ckpt checkpoints/alignment_best.pth \
        --logs_path data/synthetic_logs/maintenance_corpus.json \
        --data_dir data/processed
"""

import json
import argparse
import numpy as np
import torch
import faiss
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.autoencoder import SensorAutoencoder, AutoencoderConfig
from models.contrastive import CrossModalAlignment, ContrastiveConfig
from retrieval.build_index import CrossModalRetriever


# Proxy failure mode assignment from top-degrading sensor
# Based on CMAPSS sensor-to-component mapping
SENSOR_TO_FAILURE = {
    "sensor_2":  "thermal degradation",
    "sensor_3":  "thermal degradation",
    "sensor_4":  "compressor stall",
    "sensor_7":  "compressor stall",
    "sensor_8":  "vibration-induced fatigue",
    "sensor_9":  "vibration-induced fatigue",
    "sensor_11": "seal degradation",
    "sensor_12": "lubrication failure",
    "sensor_13": "blade tip erosion",
    "sensor_14": "blade tip erosion",
    "sensor_15": "seal degradation",
    "sensor_17": "turbine blade damage",
    "sensor_20": "bearing wear",
    "sensor_21": "bearing wear",
}

SENSOR_COLS = [f"sensor_{i}" for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]


def get_proxy_failure_mode(window: np.ndarray) -> str:
    """
    Assign a proxy failure mode to a sensor window based on which
    sensor has the highest mean value (most activated = most degraded
    after min-max normalization).
    """
    mean_per_sensor = window.mean(axis=0)    # (n_sensors,)
    top_sensor_idx  = int(mean_per_sensor.argmax())
    top_sensor_name = SENSOR_COLS[top_sensor_idx]
    return SENSOR_TO_FAILURE.get(top_sensor_name, "bearing wear")


def precision_at_k(retrieved: list[dict], true_label: str, k: int) -> float:
    """Fraction of top-k retrieved logs with matching failure_mode."""
    hits = sum(
        1 for log in retrieved[:k]
        if log.get("failure_mode") == true_label
    )
    return hits / k


def evaluate_random(logs: list[dict], query_labels: list[str], k: int = 5) -> float:
    """Random baseline: retrieve k random logs."""
    import random
    total = 0.0
    for true_label in query_labels:
        retrieved = random.sample(logs, k)
        total += precision_at_k(retrieved, true_label, k)
    return total / len(query_labels)


def evaluate_text_rag(
    query_windows: np.ndarray,
    query_labels: list[str],
    sbert_embeddings: np.ndarray,
    logs: list[dict],
    k: int = 5,
) -> float:
    """
    Text-only RAG baseline.
    Convert sensor window to text description, encode with SBERT,
    retrieve nearest text embeddings. No cross-modal alignment.
    """
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    # Build flat FAISS index over raw SBERT embeddings
    d      = sbert_embeddings.shape[1]
    index  = faiss.IndexFlatIP(d)
    vecs   = sbert_embeddings.copy().astype(np.float32)
    faiss.normalize_L2(vecs)
    index.add(vecs)

    total = 0.0
    for window, true_label in zip(query_windows, query_labels):
        # Describe anomaly as text — impoverished signal but fair baseline
        mean_sensors = window.mean(axis=0)
        top3_idx = mean_sensors.argsort()[-3:][::-1]
        sensor_names = [SENSOR_COLS[i] for i in top3_idx]
        desc = f"Engine anomaly detected in {', '.join(sensor_names)}."

        query_vec = sbert.encode([desc], normalize_embeddings=True)
        _, indices = index.search(query_vec.astype(np.float32), k)

        retrieved = [logs[i] for i in indices[0] if i < len(logs)]
        total += precision_at_k(retrieved, true_label, k)

    return total / len(query_labels)


def evaluate_sensor_knn(
    query_embeddings: np.ndarray,     # raw autoencoder latents, no projection
    query_labels: list[str],
    log_sbert_embs: np.ndarray,       # raw SBERT, truncated to 256-d
    logs: list[dict],
    k: int = 5,
) -> float:
    """
    Sensor kNN baseline: AE latent directly against truncated SBERT.
    No contrastive projection — measures what alignment actually adds.
    """
    d      = query_embeddings.shape[1]
    index  = faiss.IndexFlatIP(d)
    vecs   = log_sbert_embs[:, :d].copy().astype(np.float32)
    faiss.normalize_L2(vecs)
    index.add(vecs)

    query_vecs = query_embeddings.copy().astype(np.float32)
    faiss.normalize_L2(query_vecs)

    total = 0.0
    for i, true_label in enumerate(query_labels):
        _, indices = index.search(query_vecs[i:i+1], k)
        retrieved  = [logs[j] for j in indices[0] if j < len(logs)]
        total += precision_at_k(retrieved, true_label, k)

    return total / len(query_labels)


def evaluate_failsense(
    retriever: CrossModalRetriever,
    query_embeddings: np.ndarray,
    query_labels: list[str],
    k: int = 5,
) -> float:
    """FailSense: aligned cross-modal retrieval."""
    total = 0.0
    for i, true_label in enumerate(query_labels):
        retrieved = retriever.retrieve(query_embeddings[i], k=k)
        total += precision_at_k(retrieved, true_label, k)
    return total / len(query_labels)


def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k_values = [1, 3, 5, 10]

    # Load data
    data_dir = Path(args.data_dir)
    X_test   = np.load(data_dir / "X_test.npy")
    lbl_test = np.load(data_dir / "lbl_test.npy")
    ae_cfg   = AutoencoderConfig()
    X_test_3d = X_test.reshape(-1, ae_cfg.seq_len, ae_cfg.n_sensors)

    # Use near-failure windows only for evaluation (most meaningful)
    near_fail_mask  = lbl_test == 1
    X_eval          = X_test_3d[near_fail_mask][:200]   # cap at 200 for speed
    proxy_labels    = [get_proxy_failure_mode(w) for w in X_eval]

    print(f"Evaluation set: {len(X_eval)} near-failure windows")
    print(f"Label distribution: {dict((l, proxy_labels.count(l)) for l in set(proxy_labels))}\n")

    # Load models
    ae_model = SensorAutoencoder(ae_cfg).to(device)
    ae_ckpt  = torch.load(args.ae_ckpt, map_location=device)
    ae_model.load_state_dict(ae_ckpt["model_state"])
    ae_model.eval()

    # Get raw AE latents for all eval windows
    with torch.no_grad():
        X_tensor    = torch.tensor(X_eval).to(device)
        ae_latents  = ae_model.encoder(X_tensor).cpu().numpy()   # (N, 256)

    # Load logs and SBERT embeddings
    with open(args.logs_path) as f:
        logs = json.load(f)
    sbert_embs = np.load(data_dir / "log_sbert_embeddings.npy")

    # Load FailSense retriever
    retriever = CrossModalRetriever(
        index_path=str(data_dir / "faiss_index.bin"),
        metadata_path=str(data_dir / "log_metadata.json"),
        alignment_ckpt=args.alignment_ckpt,
    )

    # Run all four methods
    results = {}
    print("Running evaluations...")

    print("  1/4 Random baseline...")
    results["Random"] = {
        k: evaluate_random(logs, proxy_labels, k) for k in k_values
    }

    print("  2/4 Text-only RAG...")
    results["Text RAG"] = {
        k: evaluate_text_rag(X_eval, proxy_labels, sbert_embs, logs, k)
        for k in k_values
    }

    print("  3/4 Sensor kNN (no alignment)...")
    results["Sensor kNN"] = {
        k: evaluate_sensor_knn(ae_latents, proxy_labels, sbert_embs, logs, k)
        for k in k_values
    }

    print("  4/4 FailSense (cross-modal alignment)...")
    results["FailSense"] = {
        k: evaluate_failsense(retriever, ae_latents, proxy_labels, k)
        for k in k_values
    }

    # Print results table
    print("\n" + "="*60)
    print(f"{'Method':<22} " + "  ".join(f"P@{k:<3}" for k in k_values))
    print("-"*60)
    for method, scores in results.items():
        row = "  ".join(f"{scores[k]:.3f}" for k in k_values)
        marker = " ← ours" if method == "FailSense" else ""
        print(f"{method:<22} {row}{marker}")
    print("="*60)

    # Save results
    out_path = Path("eval/retrieval_results.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")
    print("Copy P@5 column into README results table.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae_ckpt",        default="checkpoints/autoencoder_best.pth")
    parser.add_argument("--alignment_ckpt", default="checkpoints/alignment_best.pth")
    parser.add_argument("--logs_path",      default="data/synthetic_logs/maintenance_corpus.json")
    parser.add_argument("--data_dir",       default="data/processed")
    args = parser.parse_args()
    run_evaluation(args)
