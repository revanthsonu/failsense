"""
src/retrieval/build_index.py

Day 3: Embed maintenance logs with SBERT, project into shared space,
build FAISS index for fast cross-modal retrieval at inference time.

Two-stage pipeline:
  1. SBERT encodes raw log text → 384-d semantic vectors
  2. Trained text projection head maps 384-d → 256-d shared space
  3. FAISS IVF index built over projected vectors for kNN search

At inference:
  - New sensor anomaly → autoencoder encoder → sensor projection head → 256-d
  - FAISS.search(query_vec, k) → top-k maintenance logs

Run:
    python src/retrieval/build_index.py \
        --logs_path data/synthetic_logs/maintenance_corpus.json \
        --alignment_ckpt checkpoints/alignment_best.pth \
        --out_dir data/processed
"""

import json
import argparse
import numpy as np
import torch
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.contrastive import CrossModalAlignment, ContrastiveConfig


def embed_logs(
    logs: list[dict],
    sbert_model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    """
    Encode maintenance log texts with SBERT.

    Returns:
        embeddings: (N, 384) float32 array
    """
    print(f"Loading SBERT: {sbert_model_name}")
    model = SentenceTransformer(sbert_model_name, device=device)

    texts = [log["text"] for log in logs]
    print(f"Embedding {len(texts)} logs...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-normalize — consistent with projection head
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def project_text_embeddings(
    raw_embeddings: np.ndarray,
    alignment_ckpt: str | None,
    cfg: ContrastiveConfig,
    device: str,
) -> np.ndarray:
    """
    Project SBERT embeddings into shared 256-d space using trained
    text projection head.

    If no checkpoint is provided (early in training), returns raw
    SBERT embeddings truncated/padded to proj_dim — useful for
    baseline evaluation before contrastive training is complete.
    """
    if alignment_ckpt is None or not Path(alignment_ckpt).exists():
        print("[WARN] No alignment checkpoint — using raw SBERT embeddings as baseline")
        # Truncate 384→256 for dimension compatibility (baseline only)
        return raw_embeddings[:, :cfg.proj_dim]

    print(f"Loading alignment checkpoint: {alignment_ckpt}")
    ckpt  = torch.load(alignment_ckpt, map_location=device)
    model = CrossModalAlignment(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tensor = torch.tensor(raw_embeddings).to(device)
    with torch.no_grad():
        projected = model.project_text(tensor)

    return projected.cpu().numpy().astype(np.float32)


def build_faiss_index(
    vectors: np.ndarray,
    index_type: str = "IVF",
) -> faiss.Index:
    """
    Build FAISS index over projected text embeddings.

    Index selection:
        - IVF (Inverted File Index): fast approximate search, good for
          1k-1M vectors. Uses nlist=32 clusters.
        - Flat: exact search, slower but useful for debugging.

    For FailSense at 1,200 logs, IVF is overkill but demonstrates
    production-grade practice — the same index scales to 1M+ logs.

    All vectors should be L2-normalized before indexing so that
    inner product search = cosine similarity search.
    """
    d = vectors.shape[1]   # 256
    faiss.normalize_L2(vectors)   # in-place L2 normalization

    if index_type == "Flat":
        index = faiss.IndexFlatIP(d)   # inner product = cosine on normalized vecs
    else:
        # IVF with inner product metric
        nlist  = min(32, len(vectors) // 10)   # rule of thumb: sqrt(N) clusters
        quant  = faiss.IndexFlatIP(d)
        index  = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vectors)
        index.nprobe = 8   # search 8 of 32 clusters at query time (precision/speed tradeoff)

    index.add(vectors)
    print(f"FAISS index built: {index.ntotal} vectors, d={d}, type={index_type}")
    return index


def build_and_save(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg    = ContrastiveConfig()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load corpus
    print(f"Loading logs: {args.logs_path}")
    with open(args.logs_path) as f:
        logs = json.load(f)
    print(f"Loaded {len(logs)} maintenance logs")

    # Step 1: SBERT embeddings
    raw_embeddings = embed_logs(logs, device=device)
    np.save(out_dir / "log_sbert_embeddings.npy", raw_embeddings)
    print(f"Saved SBERT embeddings: {raw_embeddings.shape}")

    # Step 2: Project to shared space
    projected = project_text_embeddings(
        raw_embeddings, args.alignment_ckpt, cfg, device
    )
    np.save(out_dir / "log_projected_embeddings.npy", projected)
    print(f"Saved projected embeddings: {projected.shape}")

    # Step 3: Build FAISS index
    index = build_faiss_index(projected.copy(), index_type=args.index_type)
    faiss.write_index(index, str(out_dir / "faiss_index.bin"))
    print(f"Saved FAISS index → {out_dir}/faiss_index.bin")

    # Save log metadata for retrieval display (text + failure_mode + component)
    metadata = [{
        "log_id":       l["log_id"],
        "text":         l["text"],
        "failure_mode": l["failure_mode"],
        "component":    l["component"],
        "is_critical":  l["is_critical"],
        "unit_id":      l["unit_id"],
        "cycle":        l["cycle"],
    } for l in logs]

    import json
    with open(out_dir / "log_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved log metadata")


class CrossModalRetriever:
    """
    Wraps FAISS index for cross-modal retrieval at inference time.
    Query with a sensor anomaly embedding → get top-k maintenance logs.
    """

    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        alignment_ckpt: str | None = None,
    ):
        self.index    = faiss.read_index(index_path)
        self.cfg      = ContrastiveConfig()
        self.device   = "cuda" if torch.cuda.is_available() else "cpu"

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        # Load alignment model if available
        self.alignment = None
        if alignment_ckpt and Path(alignment_ckpt).exists():
            ckpt = torch.load(alignment_ckpt, map_location=self.device)
            self.alignment = CrossModalAlignment(self.cfg).to(self.device)
            self.alignment.load_state_dict(ckpt["model_state"])
            self.alignment.eval()

    def retrieve(
        self,
        sensor_embedding: np.ndarray,   # (256,) raw autoencoder latent
        k: int = 5,
    ) -> list[dict]:
        """
        Given a sensor anomaly embedding, retrieve k most similar
        maintenance log records.

        Returns:
            List of k dicts, each with text, failure_mode, similarity score
        """
        # Project sensor embedding into shared space
        if self.alignment is not None:
            t = torch.tensor(sensor_embedding).unsqueeze(0).to(self.device)
            with torch.no_grad():
                query = self.alignment.project_sensor(t).cpu().numpy()
        else:
            query = sensor_embedding[:self.cfg.proj_dim].reshape(1, -1)

        # L2-normalize query (must match index normalization)
        faiss.normalize_L2(query)

        # Search
        scores, indices = self.index.search(query.astype(np.float32), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:   # FAISS returns -1 for unfilled slots
                continue
            entry = self.metadata[idx].copy()
            entry["similarity"] = float(score)
            results.append(entry)

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_path",      default="data/synthetic_logs/maintenance_corpus.json")
    parser.add_argument("--alignment_ckpt", default=None)
    parser.add_argument("--out_dir",        default="data/processed")
    parser.add_argument("--index_type",     default="IVF", choices=["IVF", "Flat"])
    args = parser.parse_args()
    build_and_save(args)
