# FailSense — Cross-Modal RAG for Industrial Failure Prediction

> **Aligning multivariate sensor anomaly embeddings with natural language maintenance records via contrastive learning — without paired supervision.**

[![HuggingFace Space](https://img.shields.io/badge/🤗-Live%20Demo-blue)](https://huggingface.co/spaces/revanthsonu/failsense)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Active](https://img.shields.io/badge/status-active%20development-green)]()

---

## The Problem

Industrial equipment generates two parallel data streams that have never been
intelligently connected:

- **Sensor telemetry** — thousands of channels (vibration, temperature, pressure,
  current draw) sampled continuously
- **Maintenance logs** — free-text records written by technicians after every
  inspection or repair

Current approaches treat these streams independently. Threshold-based alerting
ignores the text corpus entirely. Search over maintenance logs ignores the sensor
context entirely. The result: slow, reactive maintenance with high false-positive
rates and no mechanistic explanation of *why* an anomaly is occurring.

---

## The Research Contribution

We propose **FailSense** — a cross-modal retrieval-augmented generation framework
that:

1. Embeds sensor windows and maintenance text into a **shared latent space** via
   contrastive projection heads trained with InfoNCE loss
2. Enables **cross-modal retrieval**: query with a live sensor anomaly, retrieve
   the most semantically similar historical maintenance events
3. Feeds retrieved evidence to an **LLM reasoning agent** that produces a
   mechanistic failure explanation with urgency score and recommended action

**Why this is novel:** CLIP-style contrastive alignment has been applied to
image-text and audio-text pairs. Aligning multivariate time-series anomaly
representations with unstructured maintenance text is an open problem with no
published benchmark. This work defines the task, proposes a baseline, and
evaluates on the NASA CMAPSS turbofan degradation dataset.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FAILSENSE PIPELINE                    │
│                                                         │
│  Sensor Stream          Text Corpus                     │
│  (NASA CMAPSS)          (Maintenance Logs)              │
│       │                       │                         │
│  Sliding Window           SBERT Encoder                 │
│  + Autoencoder            (all-MiniLM-L6-v2)           │
│       │                       │                         │
│  512-d anomaly vec        768-d text vec                │
│       │                       │                         │
│  ┌────▼───────────────────────▼────┐                   │
│  │   Contrastive Projection Heads  │                   │
│  │   InfoNCE Loss (shared 256-d)   │                   │
│  └────────────────┬────────────────┘                   │
│                   │                                     │
│            FAISS Index                                  │
│         (cross-modal kNN)                               │
│                   │                                     │
│         Zero-Shot Classifier                            │
│         (DeBERTa NLI — failure modes)                  │
│                   │                                     │
│           LLM Reasoning Agent                           │
│      (GPT-4o / Llama-3 via Groq)                       │
│                   │                                     │
│    ┌──────────────▼──────────────────┐                 │
│    │  Structured Output              │                 │
│    │  · Failure mode + confidence    │                 │
│    │  · Estimated RUL (cycles)       │                 │
│    │  · Mechanistic explanation      │                 │
│    │  · Recommended action           │                 │
│    │  · Retrieved evidence citations │                 │
│    └─────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────┘
```

---

## Results (updating as experiments run)

| Method | Retrieval P@5 | RUL MAE (cycles) | Lead Time (hrs) | FP Rate |
|--------|--------------|-----------------|-----------------|---------|
| Random baseline | 0.12 | — | — | — |
| Text-only RAG | 0.31 | — | — | — |
| Sensor kNN (no alignment) | 0.28 | — | — | — |
| **FailSense (ours)** | **TBD** | **TBD** | **TBD** | **TBD** |

*Results will be populated as training completes. Target: >0.50 P@5, >40% lead time improvement over threshold baseline.*

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
- 26 sensor channels, sampled until failure
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

# Download NASA CMAPSS data
python src/ingestion/download_data.py

# Generate synthetic maintenance logs
python src/ingestion/generate_logs.py --n_logs 1200

# Run full pipeline
python src/main.py --mode demo
```

---

## Research Log

*Design decisions documented as they're made — this log is part of the research contribution.*

### Day 1 — March 12, 2026
- Initialized repo and project structure
- Defined formal problem statement (see `docs/problem_statement.md`)
- Selected NASA CMAPSS FD001 as primary evaluation dataset
- Decision: use sliding window of 30 time steps × 26 sensors = 780-dim input to autoencoder
- Decision: autoencoder bottleneck at 512-d (matches SBERT output order of magnitude)
- Open question: optimal InfoNCE temperature parameter τ — will ablate over {0.07, 0.1, 0.2}

---

## Project Structure

```
failsense/
├── data/
│   ├── raw/                    # NASA CMAPSS raw files
│   ├── processed/              # Windowed sensor arrays + embeddings
│   └── synthetic_logs/         # Generated maintenance text corpus
├── src/
│   ├── ingestion/              # Data download, preprocessing, log generation
│   ├── models/                 # Autoencoder, projection heads, contrastive loss
│   ├── retrieval/              # FAISS index build + cross-modal search
│   ├── agent/                  # LLM agent, tool definitions, memory
│   └── api/                    # FastAPI backend
├── notebooks/                  # Exploratory analysis, visualizations
├── eval/                       # Evaluation scripts, baseline comparisons
├── docs/                       # Problem statement, paper draft
└── tests/                      # Unit tests
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

**Revanth Naik Kethavath** · [LinkedIn](https://linkedin.com/in/revanth-nayak) · [GitHub](https://github.com/revanthsonu)

MS Computer Science, IIT Hyderabad (BTech) · GRE 326 (168Q)
