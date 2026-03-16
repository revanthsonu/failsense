# FailSense Research Log

*Every design decision, experiment result, and open question logged here.
This log is part of the research contribution — it demonstrates scientific
process and will inform the paper's methodology section.*

---

## Day 1 — March 12, 2026

### Setup
- Initialized repo structure
- Created formal problem statement (`docs/problem_statement.md`)
- Selected NASA CMAPSS FD001 as primary dataset (single operating condition,
  cleanest for initial validation)

### Architecture Decisions

**Autoencoder:**
- 1D CNN encoder chosen over MLP: captures local temporal correlations
  in sensor data that flat MLP cannot see
- Bottleneck dim: 256 (will ablate vs 128, 512)
- Input: 30 steps × 14 sensors = 420-dim (after dropping 7 near-zero-variance sensors)

**Contrastive alignment:**
- InfoNCE temperature τ = 0.07 (following CLIP default — will ablate)
- Projection heads: 2-layer MLP with BN (following SimCLR)
- Shared projection dim: 256

**Text encoder:**
- `all-MiniLM-L6-v2` (384-d) chosen for speed/quality tradeoff
- Alternative: `all-mpnet-base-v2` (768-d) — will test if alignment quality suffers

### Open Questions
- [ ] How to construct high-quality training pairs without ground-truth
  (sensor window, log) matches? Current plan: match near-failure windows
  to critical logs by failure mode label.
- [ ] What temperature τ is optimal for this modality pair?
- [ ] Does the contrastive alignment actually improve over text-only RAG
  enough to justify the training cost?

### Next Steps (Day 2)
- Download NASA CMAPSS data, verify preprocessing pipeline
- Run autoencoder smoke test on real data
- Begin synthetic log generation (start with 200 logs to verify quality)

---

*[Days 2–15 will be added as work progresses]*
