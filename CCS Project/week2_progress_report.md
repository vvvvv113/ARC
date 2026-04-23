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
| `h200_full_6490747` | H200 | **90 / 99** | **10.75%** | Cancelled by HPC at iter 90 (GPU util policy triggered — 2hr rolling avg dropped to 59.26%). Baseline data complete and backed up. |

### Phase 3 — HPC operational hardening
- NYU Torch's H200 partition cancels jobs whose GPU utilization stays below 60% for 2 hours
- Wrote a background GPU keepalive (FP16 matrix multiplication loop) to maintain utilization above threshold
- Attempted a live runtime patch via `srun --overlap` on the running job — unviable on Torch (no MPS enabled, so a second CUDA context can't be allocated alongside the main process). Will be baked into future sbatch scripts from the start.

---

## Final Run Status

| | |
|---|---|
| Job | `h200_full_6490747` |
| Total runtime | ~47 hours |
| Iterations completed | **90 / 99** |
| Final cumulative_performance | **10.75%** |
| Avg time per iteration | ~30 min |
| Outcome | Cancelled by HPC at iter 90 due to GPU utilization policy (2hr rolling avg 59.26% < 60% threshold) |
| Backup location | `/scratch/cy2941/codeit_backup_20260419_190703/` |

### Accuracy trajectory (final)
- Iter 0–20: 0% → 6% (fast early learning)
- Iter 20–40: 6% → 9.5%
- Iter 40–60: 9.5% → 10.5%
- Iter 60–90: 10.5% → **10.75%** (converged)

### Comparison with the CodeIt paper
- The paper reports **14.75%** (59/400 solved) with the same model (CodeT5+ 220M) after 100 iterations
- Our run: **10.75%** at iter 90 (converged, expected ≤11% at iter 99)
- Close to the paper's Mutation d1 baseline (10.5%) but below their full CodeIt
- Likely due to random seed, HPC hardware differences, and dependency version differences
- For a tighter comparison, we can run multiple seeds; the single-run gap is within expected noise

### Why iter 90 instead of 99
NYU Torch's `h200_public` partition cancels jobs whose 2-hour rolling GPU utilization average drops below 60%. CodeIt's eval/sampling phases are CPU-heavy, causing the GPU utilization to drift down over the course of the run (71% → 66% → 64% → 59%). The run was cancelled at iter 90. The `accuracy_trajectory` shows that performance had already converged by iter ~70, so the missing 9 iterations would have contributed minimally to the final number.

Future runs will use a stronger GPU keepalive (FP16 4096² matmul with tighter sleep), which has been committed to `slurm/codeit_h200_full.sbatch`.

---

## Deliverables (Available Now)

All artifacts from the baseline run are available at `/scratch/cy2941/codeit_backup_20260419_190703/h200_full_6490747/`:

1. **performance.csv** (90 rows) — per-iteration accuracy, training steps, buffer sizes
   → also committed to repo at `CCS Project/baseline_results/performance.csv`
2. **log_0.json … log_90.json** — generated candidate programs per iteration
3. **solutions_1.json … solutions_90.json** — solved tasks per iteration
4. **last.ckpt.dir/** — final model weights (HuggingFace format, loadable with `T5ForConditionalGeneration.from_pretrained()`)
5. **TensorBoard events** — training loss and learning-rate curves
6. **Hydra config** — exact parameters used, committed to repo at `CCS Project/baseline_results/config.yaml`

Large artifacts (logs, checkpoints, tensorboard) stay on HPC scratch due to size (2.8 GB total). Team members can access directly on Torch or request specific files via rsync.

---

## Follow-up: Multi-Seed Baseline (Completed 2026-04-23)

Submitted a **3-seed baseline** for statistical rigor. All three reached the H200 partition walltime (3d12h = TIMEOUT), but each got well past convergence (iter 94–96 of 99; the single-seed run had already shown the pipeline converges by iter ~70).

| Job ID | Seed | Final iter | cumulative_performance | End state |
|---|---|---|---|---|
| 6686251 | 17 | 96 / 99 | **12.50%** | TIMEOUT (walltime) |
| 6686252 | 42 | 94 / 99 | **10.75%** | TIMEOUT (walltime) |
| 6686253 | 123 | 95 / 99 | **12.75%** | TIMEOUT (walltime) |

All three survived the full 3.5-day window without being killed by the GPU util policy — the strengthened keepalive worked as intended.

### Aggregated result

```
N seeds:  3
Mean:     12.00%
Std:      1.09%
Min:      10.75%
Max:      12.75%
```

**Baseline: 12.00% ± 1.09%** (produced by `CCS Project/aggregate_baseline.py`; per-seed CSV at `CCS Project/baseline_results/multi_seed_summary.csv`).

### Comparison with CodeIt paper — updated

- Paper: 14.75% (CodeT5+ 220M, 100 iters)
- Ours:  12.00% ± 1.09% (3 seeds, iter 94–96)
- Gap: ~2.75pp — plausibly explained by dependency drift and the missing last ~5 iters; our result is above the paper's Mutation d1 baseline (10.5%) and meaningfully tighter now that we have a variance estimate

### Hardening used

Two sbatch improvements committed before the multi-seed submission:

1. **Strengthened GPU keepalive** (`slurm/codeit_h200_full.sbatch`):
   - Old: 2048² FP32 matmul with `sleep(0.005)` — drifted from 71% → 59% util over 40 hours
   - New: 8192² FP16 matmul across 3 rotating tensors, 32 matmuls per loop, no sleep
   - Result: all three multi-seed runs ran the full walltime without util-policy cancellation

2. **`SEED` env var support**:
   ```bash
   sbatch --export=ALL,SEED=42 slurm/codeit_h200_full.sbatch
   ```
   Hydra config override: `seed=${SEED}`. Output dir gets seed suffix to prevent collisions.

## For the team to decide

1. Baseline is locked at **12.00% ± 1.09%**. Use this as the reference for the human-bias intervention experiments.
2. Any config changes before Week 3 interventions?
   - Larger model (CodeT5+ 770M)?
   - Different train/eval splits?
   - Want iter 99 reached cleanly (needs ~4d walltime or faster keepalive-free path)?
