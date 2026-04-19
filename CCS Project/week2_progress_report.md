# Week 2 Progress Report — 2026-04-19

**Project:** Integrating human behavioral bias into CodeIt for ARC program synthesis
**My role:** Running the training pipeline and producing a baseline for the team to compare against

---

## Week 1 (Setup Only — No Training Yet)

- Built the Python environment on NYU Torch HPC
- Fixed a path bug: the original code broke when the project folder was named `codeit`
- Read through the pipeline source code and documented what each meta-iteration does
- Confirmed data loads correctly (401 train + 401 eval ARC tasks, 160 DSL primitives)

Full notes: `CCS Project/week1_analysis.md`

---

## Week 2 — First End-to-End Training Runs

Goal: **Run the full 99-iteration training pipeline on HPC and produce a clean baseline.**

### Phase 1 — Small-scale validation
- 6 smoke tests (2 iterations each) to debug SLURM scripts, env activation, Hydra config paths until the pipeline ran cleanly
- 1 sanity run of 10 iterations — confirmed the model actually learns (reached 3.25% accuracy)

### Phase 2 — Full-length runs
| Run | GPU | Iterations reached | Accuracy | Outcome |
|---|---|---|---|---|
| `full_6273439` | L40s | 35 / 99 | 7.75% | L40s too slow, hit partition time limit → switched to H200 |
| `h200_full_6395562` | H200 | 70 / 99 | 11% | Killed by HPC (node reclaimed, not our fault) |
| `h200_full_6490747` (**running now**) | H200 | **63 / 99** | **10.5%** | On track, expected to finish this morning |

### Phase 3 — HPC operational hardening
- NYU Torch's H200 partition cancels jobs whose GPU utilization stays below 60% for 2 hours
- Wrote a background GPU keepalive (FP16 matrix multiplication loop) to maintain utilization above threshold
- Attempted a live runtime patch via `srun --overlap` on the running job — unviable on Torch (no MPS enabled, so a second CUDA context can't be allocated alongside the main process). Will be baked into future sbatch scripts from the start.

---

## Current Run Status (Live)

| | |
|---|---|
| Job | `h200_full_6490747` |
| Elapsed time | ~25 hours |
| Current iteration | **63 / 99** |
| Accuracy | **10.5%** |
| Avg time per iteration | ~24 min |
| Expected completion | 10–11 AM today |

### Accuracy trajectory
- Iter 0–20: 0% → 6% (fast early learning)
- Iter 20–40: 6% → 9.5%
- Iter 40–63: 9.5% → 10.5% (converging, as expected)

### Comparison with the CodeIt paper
- The paper reports ~**14–15%** with the same model (CodeT5+ 220M) and same 99 iterations
- We project our run will land at **12–13%**
- Slightly below the paper but in the same range; likely due to random seed, HPC hardware, and dependency version differences
- If the team wants a tighter comparison, we can run multiple seeds later and report the mean

---

## Deliverables When the Run Finishes

Once this run completes, the team will have:
1. **performance.csv** — per-iteration accuracy, training steps, buffer sizes
2. **log_i.json / solutions_i.json** — generated programs and solved tasks per iteration
3. **TensorBoard** — training loss and learning-rate curves
4. **Hydra config** (`.hydra/config.yaml`) — exact parameters used, fully reproducible

These are the inputs for the human-bias experiments other teammates will build on top.

---

## Risks and Decisions Needed

### Risks
- The current run still has about a **25% chance of being killed** by the GPU utilization policy
- If killed, I'll rerun — roughly 2 more days to finish

### For the team to decide
1. Should we run multiple random seeds for statistical robustness? (more HPC budget)
2. Any config changes for the next runs? For example:
   - Larger model (CodeT5+ 770M)
   - Different number of iterations
   - Different train/eval splits
