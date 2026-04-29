# How to use the HPC results

A short guide for teammates on what's in the repo (and on `/scratch`),
what each file means, and how to run the analyses.

---

## TL;DR

The 6 multi-seed runs we did this term are **already aggregated and
plotted**. Everything you need is in **`CCS Project/baseline_results/`**.

| If you want… | Read this |
|---|---|
| The headline numbers (baseline + method 2) | `multi_seed_summary.csv`, `method2_vs_baseline_summary.csv` |
| Curves with mean ± σ across seeds | `baseline_multiseed_curves.png`, `method2_vs_baseline_curves.png` |
| Per-seed raw training curves | `performance_seed{17,42,123}.csv` (baseline), `method2_performance_seed{17,42,123}.csv` |
| Human-likeness comparison (DTW / Wasserstein) | `method2_dtw_comparison_summary.csv`, `method2_dtw_comparison_raw.log` |
| The Week 2 / Week 3 write-ups | `week2_progress_report.md`, `week3_progress_report.md` |

You should not need to touch `/scratch/` for most analyses — just open the
CSVs in `CCS Project/baseline_results/`. Heavy artifacts (logs, candidate
solutions, checkpoints) live on HPC scratch — see "On HPC scratch" below.

---

## What runs exist

### Baseline (CodeIt as published, no human bias)
3 seeds, all hit walltime at iter 94–96 (converged by iter ~70):

| Job ID | Seed | Final iter | Final cum pass rate |
|---|---|---|---|
| 6686251 | 17  | 96 | 12.50% |
| 6686252 | 42  | 94 | 10.75% |
| 6686253 | 123 | 95 | 12.75% |

Aggregate: **12.00% ± 1.09%** (compare to paper's 14.75%).

### Method 2 (DTW-biased mutation, human_lambda=0.5)
3 seeds, same regime:

| Job ID | Seed | Final iter | Final cum pass rate |
|---|---|---|---|
| 7064997 | 17  | 93 | 11.75% |
| 7064998 | 42  | 95 | 11.50% |
| 7065000 | 123 | 93 | 11.25% |

Aggregate: **11.50% ± 0.25%** — null effect vs baseline. See Week 3
report for full analysis.

---

## In the repo (read these first)

```
CCS Project/baseline_results/
├── performance_seed{17,42,123}.csv         # baseline raw curves (per-seed)
├── method2_performance_seed{17,42,123}.csv # method 2 raw curves (per-seed)
├── multi_seed_summary.csv                  # baseline final values per seed
├── method2_vs_baseline_summary.csv         # final values + delta per seed
├── method2_dtw_comparison_summary.csv      # DTW similarity / Wasserstein per seed (20 cols)
├── method2_dtw_comparison_raw.log          # full stdout from 07_method2_eval.py
├── baseline_multiseed_curves.png           # all 3 baseline seeds + mean ± σ
├── method2_vs_baseline_curves.png          # baseline vs method 2 mean ± σ
├── method2_dtw_per_seed.png                # per-seed mean DTW bar chart
├── method2_wasserstein_per_seed.png        # per-seed Wasserstein bar chart
├── method2_per_task_improved.png           # per-seed per-task improvement rate
└── config.yaml                             # baseline run hyperparameters
```

### CSV schemas

**`performance_seed*.csv` and `method2_performance_seed*.csv`** (raw training curves):

| col | meaning |
|---|---|
| `meta_iteration` | training iteration index (0-based) |
| `cumulative_performance` | fraction of ARC eval tasks solved by *any* iter ≤ this one |
| `performance` | fraction solved at *this* iter only (per-iter window) |
| `step` | total optimizer steps so far |
| `num_mutated_tasks` | size of mutated-task buffer |
| `num_policy_tasks` | size of policy-replay buffer |

**`method2_dtw_comparison_summary.csv`**:

| col | meaning |
|---|---|
| `seed` | seed id |
| `baseline_mean_dtw` / `method2_mean_dtw` | mean DTW similarity (1/(1+d)) of generated programs to human curves; higher = more human-like |
| `baseline_wasserstein` / `method2_wasserstein` | Wasserstein-1 distance between AI vs human curve-AUC distributions; lower = more human-like |
| `delta_*` | method 2 − baseline (positive = method 2 better, except for Wasserstein where positive = worse) |
| `mw_u_stat`, `mw_p_value` | Mann-Whitney U test on per-program DTW similarities (note: treats each program as independent — over-powered) |
| `tasks_improved / tasks_total_pertask` | count of tasks where method 2's mean DTW > baseline's |

---

## On HPC scratch (only if you need raw artifacts)

```
/scratch/cy2941/codeit_outputs/
├── h200_full_6686251_seed17/                          # baseline runs
├── h200_full_6686252_seed42/
├── h200_full_6686253_seed123/
├── method2_h200_full_7064997_seed17_lambda0.5/        # method 2 runs
├── method2_h200_full_7064998_seed42_lambda0.5/
└── method2_h200_full_7065000_seed123_lambda0.5/
```

Each run dir contains:

| file | size | content |
|---|---|---|
| `performance.csv` | ~3 KB | per-iter pass rate (already mirrored to repo) |
| `solutions_<N>.json` | ~30 MB each | candidate programs the model produced at iter N (per-task list of program strings + per-demonstration-example performance) |
| `log_<N>.json` | ~9 MB each | full sampling log for iter N |
| `last.ckpt.dir/` | ~1 GB | final HuggingFace checkpoint (loadable with `T5ForConditionalGeneration.from_pretrained()`) |

Slurm logs:
```
/scratch/cy2941/codeit_logs/{method2_,}h200_full_<jobid>.{out,err}
```

If you only have `cy2941`-permissioned access, ask Harvey to `rsync` what
you need (one solutions JSON ~30 MB, one full run dir ~3 GB). Don't pull
checkpoints unless you specifically need to fine-tune from them.

---

## How to reproduce / extend the analyses

All scripts live in `CCS Project/`. Run from the repo root after activating
the env (`source setup_env.sh`).

| Script | What it does | Inputs | Outputs |
|---|---|---|---|
| `aggregate_baseline.py` | Aggregate 3-seed baseline finals → mean±σ | baseline run dirs on `/scratch` | `multi_seed_summary.csv` |
| `plot_baseline.py` | 3-seed cumulative curves + convergence check | per-seed CSVs in `baseline_results/` | `baseline_multiseed_curves.png` |
| `compare_method2_vs_baseline.py` | Side-by-side method 2 vs baseline pass-rate | both run-dir sets on `/scratch` | `method2_vs_baseline_{summary.csv, curves.png}` |
| `make_method2_summary_plots.py` | Per-seed bar charts (DTW + Wasserstein + per-task) | `method2_dtw_comparison_summary.csv` | three `method2_*_per_seed.png` files |
| `analysis/07_method2_eval.py` | DTW similarity / Wasserstein on solutions | `--baseline X.json --biased Y.json` (per-seed solutions JSON) | per-seed stdout (we collected to `method2_dtw_comparison_raw.log`) |

### Common tasks

**"Re-aggregate the 3-seed baseline":**
```bash
python "CCS Project/aggregate_baseline.py"
python "CCS Project/plot_baseline.py"
```

**"Re-plot method 2 vs baseline at a different epoch":**
Edit the `common_max` line in `compare_method2_vs_baseline.py` (currently
auto-picks the latest iter shared by all 6 runs = 93), then rerun.

**"Compare method 2 at a different seed/epoch on the DTW side":**
```bash
python analysis/07_method2_eval.py \
  --baseline /scratch/cy2941/codeit_outputs/h200_full_6686251_seed17/solutions_93.json \
  --biased   /scratch/cy2941/codeit_outputs/method2_h200_full_7064997_seed17_lambda0.5/solutions_93.json
```

**"Add a new seed":**
```bash
sbatch --export=ALL,SEED=999,LAMBDA=0.5 slurm/method2_h200_full.sbatch
# wait ~3.5 days, then mirror performance.csv into baseline_results/
```

---

## Known gotchas

1. **`07_method2_eval.py` has hard-coded data paths** that broke after the
   repo restructure. Local symlinks fix them in this checkout:
   - `codelt/data/evaluation` → `data/evaluation`
   - `analysis/processed/human_traces.json` → parent repo's
     `codeit-3/analysis/processed/04_human_traces/human_traces.json`

   If you check out fresh, recreate these or pass `--human-traces` and
   patch `EVAL_DIR` at the top of the script.

2. **`07_method2_eval.py` plot dir is a single shared path**, so per-seed
   runs overwrite each other. Use the saved CSV summary instead of the
   PNGs (or add an `--out-dir` flag).

3. **No `config.yaml` is auto-saved by the method 2 runs** (baseline ones
   do). Hyperparameters are captured by `slurm/method2_h200_full.sbatch`
   for now.

4. **All 6 runs hit walltime, none reached iter 99.** The numbers in the
   reports are at iter 93 (the latest shared by all 6) or per-seed final
   (iter 93–96). Performance had plateaued by iter ~70, so this does not
   meaningfully shift the headline.

5. **`/scratch` is purgable** — NYU Torch's scratch can purge files older
   than 60 days. If you plan to use HPC artifacts months from now, copy
   the needed files into `/home` or external storage.

---

## Who to ask

| Question | Person |
|---|---|
| Anything about the HPC pipeline / how to extend training | Harvey |
| What the project is about, biasing strategy direction | Solim / project lead |
| ARC dataset / human trace format | check `codeit-3/analysis/` parent repo |
