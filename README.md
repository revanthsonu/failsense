# FailSense

**Cross-modal RAG for industrial predictive maintenance** — aligning multivariate sensor anomaly embeddings with natural language maintenance records via contrastive learning, without paired supervision.

[![CI](https://github.com/revanthsonu/failsense/actions/workflows/ci.yml/badge.svg)](https://github.com/revanthsonu/failsense/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Active](https://img.shields.io/badge/status-active%20development-green)]()
[![Demo: Coming Soon](https://img.shields.io/badge/🤗%20demo-coming%20soon-lightgrey)]()

---

Industrial equipment generates two data streams that have never been connected: continuous **sensor telemetry** and free-text **maintenance logs** written by technicians. Current systems treat them independently — threshold alerts have no memory of past failures, and keyword search over logs has no awareness of the sensor context.

FailSense bridges this gap. A 1D CNN autoencoder encodes sliding sensor windows into 256-d anomaly vectors. SBERT encodes maintenance logs into 384-d text vectors. Two projection heads, trained with InfoNCE loss (same objective as CLIP), pull matching sensor–text pairs together in a shared 256-d latent space — **without requiring explicitly paired training data**. At inference, a live sensor anomaly queries a FAISS index cross-modally, retrieves the most semantically relevant past maintenance events, and an LLM reasoning agent synthesizes a structured diagnosis: failure mode, estimated remaining useful life, and recommended action.

CLIP-style contrastive alignment has been applied to image–text and audio–text pairs. **Aligning multivariate time-series anomaly embeddings with unstructured maintenance text is an open problem with no published benchmark.** FailSense defines the task, proposes a baseline, and evaluates on the NASA CMAPSS turbofan degradation dataset.

---

## How it works

```
┌───────────────────────────────────────────────────────┐
│                    FAILSENSE PIPELINE                 │
│                                                       │
│  Sensor Stream          Text Corpus                   │
│  (NASA CMAPSS)          (Maintenance Logs)            │
│       │                       │                       │
│  Sliding Window           SBERT Encoder               │
│  + Autoencoder            (all-MiniLM-L6-v2)          │
│       │                       │                       │
│  256-d anomaly vec        384-d text vec              │
│       │                       │                       │
│  ┌────▼───────────────────────▼────┐                  │
│  │   Contrastive Projection Heads  │                  │
│  │   InfoNCE Loss (shared 256-d)   │                  │
│  └────────────────┬────────────────┘                  │
│                   │                                   │
│            FAISS Index                                │
│         (cross-modal kNN)                             │
│                   │                                   │
│         Zero-Shot Classifier                          │
│         (DeBERTa NLI — failure modes)                 │
│                   │                                   │
│           LLM Reasoning Agent                         │
│      (GPT-4o / Llama-3 via Groq)                      │
│                   │                                   │
│    ┌──────────────▼──────────────────┐                │
│    │  Structured Output              │                │
│    │  · Failure mode + confidence    │                │
│    │  · Estimated RUL (cycles)       │                │
│    │  · Mechanistic explanation      │                │
│    │  · Recommended action           │                │
│    │  · Retrieved evidence citations │                │
│    └─────────────────────────────────┘                │
└───────────────────────────────────────────────────────┘
```

---

## Results

Training in progress. Baselines and FailSense results will be populated once contrastive alignment training completes (est. Day 7). See [research log](docs/research_log.md) for current status.

| Method | Retrieval P@5 | RUL MAE (cycles) | Lead Time | FP Rate |
|--------|--------------|-----------------|-----------|---------|
| Random baseline | — | — | — | — |
| Text-only RAG | — | — | — | — |
| Sensor kNN (no alignment) | — | — | — | — |
| **FailSense (ours)** | — | — | — | — |

---

## HuggingFace Tasks Used

| Task | Model | Role in pipeline |
|------|-------|-----------------|
| Feature Extraction | `all-MiniLM-L6-v2` | Maintenance log embedding |
| Feature Extraction | Custom autoencoder | Sensor window embedding |
| Zero-Shot Classification | `cross-encoder/nli-deberta-v3-base` | Failure mode labeling |
| Text Generation | `GPT-4o` / `llama-3-70b` | LLM reasoning agent |
| Document Question Answering | `deepset/deberta-v3-base-squad2` | Maintenance corpus QA |

---

## Dataset

**Sensors:** NASA CMAPSS Turbofan Engine Degradation Simulation Dataset
- 4 sub-datasets (FD001–FD004), 100 training engines per set
- 26 raw sensor channels; 7 near-zero-variance channels dropped → **14 informative sensors** used
- Download: [NASA Prognostics Data Repository](https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq)

**Maintenance Logs:** Synthetically generated using GPT-4o seeded with:
- OSHA maintenance terminology
- Real turbofan component taxonomy (HPT, LPT, HPC, fan, combustor)
- Failure mode distributions from CMAPSS ground truth labels
- 1,200 logs generated, available in `data/synthetic_logs/`

---

## Quickstart

```bash
git clone https://github.com/revanthsonu/failsense
cd failsense
pip install -r requirements.txt

# Set your OpenAI key (needed for log generation only)
cp .env.example .env
# edit .env: OPENAI_API_KEY=sk-...
```

**Step 1 — Data**
```bash
# Download NASA CMAPSS manually from:
# https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq
# Place train_FD001.txt, test_FD001.txt, RUL_FD001.txt into data/raw/

python src/ingestion/preprocess.py --data_dir data/raw --subset FD001
python src/ingestion/generate_logs.py --n_logs 1200
```

**Step 2 — Train autoencoder** *(Day 2, ~20 min on CPU)*
```bash
python src/models/train_autoencoder.py --epochs 50
# checkpoint → checkpoints/autoencoder_best.pth
# MLflow UI: mlflow ui
```

**Step 3 — Build baseline FAISS index** *(Day 3)*
```bash
python src/retrieval/build_index.py
# index → data/processed/faiss_index.bin
```

**Step 4 — Train contrastive alignment** *(Day 3)*
```bash
python src/models/train_contrastive.py --ae_ckpt checkpoints/autoencoder_best.pth
# checkpoint → checkpoints/alignment_best.pth
```

**Step 5 — Rebuild index with alignment**
```bash
python src/retrieval/build_index.py --alignment_ckpt checkpoints/alignment_best.pth
```

**Smoke-test models only (no data needed)**
```bash
python src/models/autoencoder.py
python src/models/contrastive.py
```

> Agent, API, and demo app are in active development (Days 8–10). See [research log](docs/research_log.md).

---

## Design decisions

*Key choices logged as they're made — this trail informs the paper's methodology section.*

### Day 1 — March 12, 2026
- Initialized repo and project structure
- Defined formal problem statement (see `docs/problem_statement.md`)
- Selected NASA CMAPSS FD001 as primary evaluation dataset
- Decision: drop 7 near-zero-variance sensors from CMAPSS (1,5,6,10,16,18,19) → 14 informative sensors remain
- Decision: sliding window of 30 time steps × 14 sensors = 420-dim input to autoencoder
- Decision: autoencoder bottleneck at 256-d; SBERT output 384-d; shared projection space 256-d
- Open question: optimal InfoNCE temperature parameter τ — will ablate over {0.07, 0.1, 0.2}

### Day 2 — March 13, 2026
- Built autoencoder training loop (`train_autoencoder.py`) with AdamW + cosine LR + MLflow tracking
- Added `_check_anomaly_separation()` post-training sanity check — failure windows should score >2× healthy
- Built synthetic maintenance log generator (`generate_logs.py`) via GPT-4o with turbofan taxonomy

### Day 3 — March 14, 2026
- Built SBERT embedding pipeline and FAISS IVF index (`build_index.py`)
- Added `CrossModalRetriever` class — wraps FAISS for cross-modal kNN at inference
- Built contrastive training loop (`train_contrastive.py`) — freezes AE encoder, trains projection heads only
- Decision: use in-batch negatives (same as CLIP) — no hard negative mining yet; add if P@5 < 0.35

### Day 4 — March 15, 2026
- Built zero-shot failure mode classifier (`classifier.py`) — DeBERTa NLI, no labelled data needed
- Added `SENSOR_TO_FAILURE` mapping for proxy ground-truth labels on CMAPSS windows
- Built retrieval evaluation script (`eval/evaluate_retrieval.py`) — compares all 4 methods at P@{1,3,5,10}
- Decision: use proxy failure mode labels derived from top-degrading sensor for evaluation consistency
- Open question: does failure-mode-matched pair construction in contrastive training improve P@5 by >10%?

### Day 5 — March 15, 2026
- Built LLM reasoning agent (`agent/agent.py`) with 3 tools: get_sensor_history, query_maintenance_db, estimate_rul
- Added `EngineMemory` — sliding deque of last 48 anomaly readings per engine; tracks trend
- Added `FailureDiagnosis` Pydantic schema — 8 required fields, structured JSON output
- Decision: temperature=0.1 for agent to ensure structured output consistency
- Decision: fallback to classification result if JSON parse fails — agent never returns empty

---

## Project structure

```
failsense/
├── data/
│   ├── raw/                         # NASA CMAPSS raw files (not committed)
│   ├── processed/                   # Windowed arrays, embeddings, FAISS index
│   └── synthetic_logs/              # Generated maintenance text corpus
├── src/
│   ├── ingestion/
│   │   ├── preprocess.py            # CMAPSS loader, sliding window extractor
│   │   └── generate_logs.py         # GPT-4o synthetic log generator
│   ├── models/
│   │   ├── autoencoder.py           # 1D CNN autoencoder architecture
│   │   ├── train_autoencoder.py     # Training loop + MLflow tracking
│   │   ├── contrastive.py           # Projection heads + InfoNCE loss
│   │   ├── train_contrastive.py     # Contrastive alignment training loop
│   │   └── classifier.py           # Zero-shot failure mode classifier (DeBERTa NLI)
│   ├── retrieval/
│   │   └── build_index.py           # SBERT embedding + FAISS index builder
│   ├── agent/
│   │   └── agent.py                 # LLM reasoning agent, 3 tools, Pydantic output
│   └── api/                         # FastAPI backend [in progress — Day 8]
├── eval/
│   └── evaluate_retrieval.py        # P@k evaluation, 4-method comparison
├── notebooks/                       # EDA, visualizations [in progress]
├── docs/
│   ├── problem_statement.md         # Formal problem formulation
│   └── research_log.md              # Daily design decisions
└── tests/
    └── test_models.py               # Smoke tests for autoencoder + contrastive
```

---

## Citation

If you find this work useful:

```bibtex
@misc{kethavath2026failsense,
  title={FailSense: Cross-Modal Contrastive Alignment for Industrial Failure Prediction},
  author={Kethavath, Revanth Naik},
  year={2026},
  note={Preprint / Workshop submission}
}
```

---

## Author

**Revanth Naik Kethavath** · [LinkedIn](https://linkedin.com/in/revanth-nayak) · [GitHub](https://github.com/revanthsonu) · krevanthnaik@gmail.com

BTech — IIT Hyderabad · MS CS — UT Dallas (May 2026) · GRE 326 (168Q/158V)
