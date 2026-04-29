"""
Mean-trace vs mean-trace DTW comparison (professor's framing, Week 3+).

Instead of comparing each CodeIT program against the *set* of human curves
(what 07_method2_eval.py does), this script collapses each side to a single
mean trajectory per task and computes one DTW distance per task between
the two means.

Per-task pipeline:
  1. mean_human_curve(t)  = element-wise mean of all success human curves on task t
  2. mean_codeit_curve(t) = element-wise mean of all CodeIT program curves on task t
                           (from the run's solutions_<N>.json)
  3. d(t) = DTW_distance( mean_human_curve, mean_codeit_curve )

Bootstrap CI on the human side (B=200): resample human curves per task with
replacement → recompute mean → redo DTW. Quantifies the wide-CI limitation
the professor flagged.

Aggregation: mean ± SD of d(t) across tasks. Run for baseline + method 2
across all 3 seeds; report delta.

Output (CCS Project/baseline_results/):
  - mean_trace_comparison_per_task.csv   long format: seed, condition, task, dtw, ci_lo, ci_hi
  - mean_trace_comparison_summary.csv    per-seed/per-condition aggregate
  - mean_trace_comparison.png            box plot of per-task DTW distances by condition

Run from repo root:
    python "CCS Project/compare_mean_traces.py"
"""
from pathlib import Path
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = "/home/cy2941/codeit-3/codeit-1"
sys.path.insert(0, REPO)
sys.path.insert(1, os.path.join(REPO, "codelt"))

from codeit.human_trajectories import (
    build_human_curves,
    get_program_curve,
    _dtw,
)

HERE = Path(__file__).parent
RESULTS = HERE / "baseline_results"
RESULTS.mkdir(parents=True, exist_ok=True)

HUMAN_TRACES = "/home/cy2941/codeit-3/analysis/processed/04_human_traces/human_traces.json"
EVAL_DIR = Path(REPO) / "data" / "evaluation"
EPOCH = 93  # last common iter across all 6 runs

BASELINE_DIRS = {
    17:  "/scratch/cy2941/codeit_outputs/h200_full_6686251_seed17",
    42:  "/scratch/cy2941/codeit_outputs/h200_full_6686252_seed42",
    123: "/scratch/cy2941/codeit_outputs/h200_full_6686253_seed123",
}
METHOD2_DIRS = {
    17:  "/scratch/cy2941/codeit_outputs/method2_h200_full_7064997_seed17_lambda0.5",
    42:  "/scratch/cy2941/codeit_outputs/method2_h200_full_7064998_seed42_lambda0.5",
    123: "/scratch/cy2941/codeit_outputs/method2_h200_full_7065000_seed123_lambda0.5",
}

N_BOOTSTRAP = 200
RNG = np.random.default_rng(0)


def load_input_target(task_id):
    path = EVAL_DIR / f"{task_id}.json"
    with open(path) as f:
        data = json.load(f)
    inp = tuple(tuple(r) for r in data["test_examples"][0]["input"])
    tgt = tuple(tuple(r) for r in data["test_examples"][0]["output"])
    return inp, tgt


def codeit_curves_for_task(programs, input_grid, target_grid):
    """List of program curves for one task (skips programs that fail to execute)."""
    out = []
    for program_str in programs:
        c = get_program_curve(program_str, input_grid, target_grid)
        if c is not None:
            out.append(c)
    return out


def per_task_dtw(mean_human_curve, codeit_curves, n_bootstrap=N_BOOTSTRAP):
    """
    Returns (dtw_point_estimate, ci_lo_human_bootstrap, ci_hi_human_bootstrap).
    Bootstrap is over the codeit side — but professor's framing wants the
    *human* side bootstrapped. We accept human curves precomputed; bootstrap
    here is on the codeit side (resample codeit curves with replacement,
    recompute mean, redo DTW). Symmetric; both are valid CIs.
    """
    if not codeit_curves:
        return np.nan, np.nan, np.nan
    mean_codeit = np.mean(codeit_curves, axis=0)
    point = _dtw(mean_human_curve, mean_codeit)

    # bootstrap on codeit side (cheaper, codeit has more samples)
    n = len(codeit_curves)
    boots = np.empty(n_bootstrap)
    arr = np.asarray(codeit_curves)
    for b in range(n_bootstrap):
        idx = RNG.integers(0, n, size=n)
        boot_mean = arr[idx].mean(axis=0)
        boots[b] = _dtw(mean_human_curve, boot_mean)
    return float(point), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def per_task_dtw_human_bootstrap(human_curves_list, mean_codeit, n_bootstrap=N_BOOTSTRAP):
    """Bootstrap on the *human* side — what the professor's "wide CI" caveat refers to."""
    if not human_curves_list:
        return np.nan
    n = len(human_curves_list)
    arr = np.asarray(human_curves_list)
    boots = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = RNG.integers(0, n, size=n)
        boot_mean_human = arr[idx].mean(axis=0)
        boots[b] = _dtw(boot_mean_human, mean_codeit)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def evaluate_run(run_dir, human_curves):
    """Return list of dict rows: one per task."""
    sol_path = Path(run_dir) / f"solutions_{EPOCH}.json"
    with open(sol_path) as f:
        data = json.load(f)
    tasks = data["policy"]["task_demonstration"]

    rows = []
    for i, (task_id, programs) in enumerate(tasks.items()):
        if task_id not in human_curves:
            continue
        try:
            inp, tgt = load_input_target(task_id)
        except Exception:
            continue

        human_curves_list = human_curves[task_id]
        mean_human = np.mean(human_curves_list, axis=0)

        codeit_curves = codeit_curves_for_task(list(programs.keys()), inp, tgt)
        if not codeit_curves:
            continue
        mean_codeit = np.mean(codeit_curves, axis=0)
        point = float(_dtw(mean_human, mean_codeit))

        # bootstrap on human side (the side with low n)
        ci_lo_h, ci_hi_h = per_task_dtw_human_bootstrap(human_curves_list, mean_codeit)

        rows.append({
            "task_id": task_id,
            "n_human": len(human_curves_list),
            "n_codeit": len(codeit_curves),
            "dtw_mean_vs_mean": point,
            "human_bootstrap_ci_lo": ci_lo_h,
            "human_bootstrap_ci_hi": ci_hi_h,
        })

    return rows


def main():
    print(f"Loading human curves from {HUMAN_TRACES}")
    human_curves = build_human_curves(HUMAN_TRACES)
    print(f"  loaded curves for {len(human_curves)} tasks")

    all_rows = []
    summary_rows = []

    for cond, dirs in [("baseline", BASELINE_DIRS), ("method2", METHOD2_DIRS)]:
        for seed, run_dir in dirs.items():
            print(f"\n[{cond} / seed {seed}] evaluating {run_dir}...")
            rows = evaluate_run(run_dir, human_curves)
            for r in rows:
                r["seed"] = seed
                r["condition"] = cond
                all_rows.append(r)
            dtws = [r["dtw_mean_vs_mean"] for r in rows]
            summary_rows.append({
                "condition": cond,
                "seed": seed,
                "n_tasks": len(dtws),
                "dtw_mean": float(np.mean(dtws)),
                "dtw_std": float(np.std(dtws, ddof=1)) if len(dtws) > 1 else 0.0,
                "dtw_median": float(np.median(dtws)),
                "ci_width_mean": float(np.mean([r["human_bootstrap_ci_hi"] - r["human_bootstrap_ci_lo"] for r in rows])),
            })
            print(f"  -> n_tasks={len(dtws)}  mean DTW = {np.mean(dtws):.4f} ± {np.std(dtws, ddof=1):.4f}")

    long_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summary_rows)

    long_df.to_csv(RESULTS / "mean_trace_comparison_per_task.csv", index=False)
    summary_df.to_csv(RESULTS / "mean_trace_comparison_summary.csv", index=False)

    print("\n=== Mean-trace-vs-mean-trace DTW (per seed) ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # paired delta per seed
    print("\n=== Paired delta (method 2 − baseline) per seed ===")
    for s in sorted(BASELINE_DIRS):
        b = summary_df.query("condition == 'baseline' and seed == @s")["dtw_mean"].iat[0]
        m = summary_df.query("condition == 'method2'  and seed == @s")["dtw_mean"].iat[0]
        print(f"  seed {s}: baseline={b:.4f}  method2={m:.4f}  Δ={m-b:+.4f}  (negative=method2 closer to humans)")

    # box plot
    fig, ax = plt.subplots(figsize=(8, 5))
    by_cond_seed = []
    labels = []
    for cond in ["baseline", "method2"]:
        for s in sorted(BASELINE_DIRS):
            sub = long_df.query("condition == @cond and seed == @s")["dtw_mean_vs_mean"].values
            by_cond_seed.append(sub)
            labels.append(f"{cond}\nseed {s}")
    bp = ax.boxplot(by_cond_seed, labels=labels, showmeans=True, patch_artist=True)
    colors = ["#1f77b4", "#1f77b4", "#1f77b4", "#d62728", "#d62728", "#d62728"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.3)
    ax.set_ylabel("DTW distance: mean human curve vs mean CodeIT curve\n(per task; lower = more human-like)")
    ax.set_title(f"Per-task mean-trace DTW distance (epoch {EPOCH}, 3 seeds × 2 conditions)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "mean_trace_comparison.png", dpi=150)
    print(f"\nSaved -> {RESULTS/'mean_trace_comparison_per_task.csv'}")
    print(f"Saved -> {RESULTS/'mean_trace_comparison_summary.csv'}")
    print(f"Saved -> {RESULTS/'mean_trace_comparison.png'}")


if __name__ == "__main__":
    main()
