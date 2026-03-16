# FailSense: Formal Problem Statement

*This document forms the basis for the workshop paper introduction and
PhD application research statement.*

---

## Motivation

Industrial predictive maintenance systems generate two heterogeneous data
streams that have historically operated in isolation:

1. **Sensor telemetry** $\mathcal{X} = \{x_t \in \mathbb{R}^d\}$ — continuous
   multivariate measurements (vibration, temperature, pressure) at high frequency

2. **Maintenance records** $\mathcal{M} = \{m_i\}$ — unstructured natural
   language logs written by technicians, rich in mechanistic context

Current systems treat these streams independently. Threshold-based anomaly
detectors operate purely on $\mathcal{X}$, ignoring the accumulated institutional
knowledge encoded in $\mathcal{M}$. Keyword search over $\mathcal{M}$ ignores
the sensor context that would make retrieval semantically grounded.

The consequence: when an anomaly is detected, there is no principled way to
ask *"what does this sensor pattern mean, and what has happened when we've
seen it before?"*

---

## Problem Formulation

**Given:**
- A sensor corpus $\mathcal{X} = \{(x_t^{(k)}, y_k)\}$ where $x_t^{(k)}$ is
  a sliding window of sensor readings for engine $k$ at time $t$, and
  $y_k \in \mathbb{R}$ is the remaining useful life (RUL)
- A maintenance corpus $\mathcal{M} = \{(m_i, f_i)\}$ where $m_i$ is a
  free-text maintenance record and $f_i$ is its failure mode label

**Learn:**
- An encoder $\phi_s: \mathcal{X} \rightarrow \mathbb{R}^d$ mapping sensor
  windows to anomaly embeddings
- An encoder $\phi_t: \mathcal{M} \rightarrow \mathbb{R}^d$ mapping maintenance
  text to semantic embeddings
- Projection heads $g_s, g_t$ such that for matching pairs $(x, m)$:

$$\text{sim}(g_s(\phi_s(x)),\ g_t(\phi_t(m))) \gg \text{sim}(g_s(\phi_s(x)),\ g_t(\phi_t(m')))$$

for any non-matching $m' \neq m$.

**Such that:**
- Cross-modal retrieval: $\text{Retrieve}(x_{\text{new}}) = \text{kNN}(g_s(\phi_s(x_{\text{new}})),\ \{g_t(\phi_t(m_i))\})$
  returns maintenance records semantically relevant to the current anomaly

---

## Why This Is Novel

CLIP (Radford et al. 2021) demonstrated that contrastive alignment between
image and text embeddings enables powerful zero-shot transfer. Subsequent work
extended this to audio-text (CLAP), video-text (VideoCLIP), and
molecule-text (MolCLIP) pairs.

**To our knowledge, no published work has applied contrastive cross-modal
alignment to the (multivariate time-series, natural language) modality pair
in an industrial monitoring context.** This gap exists despite:

- The ubiquity of paired sensor + maintenance log data in industrial settings
- The practical value of grounding sensor anomalies in natural language context
- The theoretical interest of aligning a continuous, high-dimensional temporal
  signal with discrete natural language

FailSense addresses this gap by:
1. Defining a formal benchmark task and evaluation protocol
2. Proposing a contrastive projection head architecture adapted for this modality pair
3. Evaluating on the NASA CMAPSS dataset with a synthetic-but-grounded text corpus

---

## Evaluation Protocol

### Primary Metrics

| Metric | Definition | Why It Matters |
|--------|-----------|----------------|
| Retrieval P@k | Fraction of top-k retrieved logs with matching failure_mode | Direct measure of cross-modal alignment quality |
| RUL MAE | Mean absolute error in cycles vs ground truth | Downstream task performance |
| Lead time | Hours of warning before failure vs threshold baseline | Production utility |
| False positive rate | Alerts on healthy engines / total healthy windows | Deployment feasibility |

### Baselines

1. **Random** — retrieves k random maintenance logs (lower bound)
2. **Text-only RAG** — sensor anomaly described in text, SBERT search over corpus
3. **Sensor kNN** — nearest neighbor in raw sensor space (no alignment)
4. **FailSense (ours)** — cross-modal contrastive alignment

---

## Broader Impact

The alignment framework generalizes to any domain with paired sensor +
text streams: EV battery degradation + service records, wind turbine
SCADA data + field technician notes, aircraft ACARS messages +
maintenance deferrals. The benchmark and codebase serve as a foundation
for future work in industrial multimodal learning.
