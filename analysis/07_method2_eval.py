"""
Evaluate Method 2 (DTW trajectory bias) by comparing baseline vs biased solutions.

Usage:
    python analysis/07_method2_eval.py \
        --baseline data/solutions_baseline.json \
        --biased   data/solutions_biased.json

If only --baseline is provided, prints stats for that run alone.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wasserstein_distance

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(1, os.path.join(REPO, "codelt"))

from codeit.human_trajectories import (
    build_human_curves,
    compute_dtw_similarity,
    get_program_curve,
)

HUMAN_TRACES = os.path.join(REPO, "analysis/processed/human_traces.json")
EVAL_DIR = os.path.join(REPO, "codelt/data/evaluation")
DIFFICULTY_CSV = os.path.join(REPO, "analysis/processed/01_difficulty/task_difficulty.csv")
PLOTS_DIR = os.path.join(REPO, "analysis/processed/method2_eval")


def load_input_target(task_id):
    path = os.path.join(EVAL_DIR, f"{task_id}.json")
    with open(path) as f:
        task_data = json.load(f)
    input_grid = tuple(tuple(r) for r in task_data["test_examples"][0]["input"])
    target_grid = tuple(tuple(r) for r in task_data["test_examples"][0]["output"])
    return input_grid, target_grid


def evaluate_solutions(solutions_path, human_curves):
    """
    For each task in solutions, compute:
      - solve rate (task_demonstration_performance)
      - mean DTW similarity of all programs for that task

    Returns per-task results and aggregate stats.
    """
    with open(solutions_path) as f:
        data = json.load(f)

    tasks = data.get("policy", data).get("task_demonstration", {})

    per_task = {}
    all_dtw_sims = []
    n_tasks = 0
    n_solved = 0

    for task_id, programs in tasks.items():
        if task_id not in human_curves:
            continue

        curves = human_curves[task_id]
        try:
            input_grid, target_grid = load_input_target(task_id)
        except Exception:
            continue

        n_tasks += 1
        task_solved = False
        task_sims = []

        for program_str, meta in programs.items():
            perf = meta.get("task_demonstration_performance", [])
            if isinstance(perf, list) and any(perf):
                task_solved = True
            elif perf:
                task_solved = True

            curve = get_program_curve(program_str, input_grid, target_grid)
            sim = compute_dtw_similarity(curve, curves)
            task_sims.append(sim)
            all_dtw_sims.append(sim)

        if task_solved:
            n_solved += 1

        per_task[task_id] = {
            "solved": task_solved,
            "mean_dtw_sim": float(np.mean(task_sims)) if task_sims else 0.0,
            "n_programs": len(task_sims),
        }

    solve_rate = n_solved / n_tasks if n_tasks > 0 else 0.0
    mean_dtw = float(np.mean(all_dtw_sims)) if all_dtw_sims else 0.0
    median_dtw = float(np.median(all_dtw_sims)) if all_dtw_sims else 0.0

    # collect all program curves (recompute for Wasserstein)
    all_program_curves = []
    for task_id, programs in tasks.items():
        if task_id not in human_curves:
            continue
        try:
            input_grid, target_grid = load_input_target(task_id)
        except Exception:
            continue
        for program_str in programs:
            curve = get_program_curve(program_str, input_grid, target_grid)
            if curve is not None:
                all_program_curves.append(curve)

    return per_task, solve_rate, mean_dtw, median_dtw, all_dtw_sims, all_program_curves


def compute_wasserstein(program_curves, human_curves_dict):
    """
    Compute Wasserstein distance between the distribution of all program curves
    and the distribution of all human curves, using mean curve value as a
    1-D projection (AUC proxy).
    """
    ai_aucs = [float(np.mean(c)) for c in program_curves]
    human_aucs = [float(np.mean(c)) for curves in human_curves_dict.values() for c in curves]
    if not ai_aucs or not human_aucs:
        return float("nan")
    return float(wasserstein_distance(ai_aucs, human_aucs))


def print_report(label, per_task, solve_rate, mean_dtw, median_dtw, wass):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Tasks evaluated:        {len(per_task)}")
    print(f"  Solve rate:             {solve_rate:.3f}")
    print(f"  Mean DTW similarity:    {mean_dtw:.4f}")
    print(f"  Median DTW similarity:  {median_dtw:.4f}")
    print(f"  Wasserstein distance:   {wass:.4f}  (lower = more human-like distribution)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--biased", default=None)
    parser.add_argument("--human-traces", default=HUMAN_TRACES)
    args = parser.parse_args()

    print("Loading human trajectory curves...")
    human_curves = build_human_curves(args.human_traces)
    print(f"Loaded curves for {len(human_curves)} tasks")

    print("\nEvaluating baseline...")
    b_per_task, b_solve, b_mean_dtw, b_med_dtw, b_sims, b_curves = evaluate_solutions(
        args.baseline, human_curves
    )
    b_wass = compute_wasserstein(b_curves, human_curves)
    print_report("Baseline (λ=0)", b_per_task, b_solve, b_mean_dtw, b_med_dtw, b_wass)

    if args.biased:
        print("\nEvaluating biased...")
        h_per_task, h_solve, h_mean_dtw, h_med_dtw, h_sims, h_curves = evaluate_solutions(
            args.biased, human_curves
        )
        h_wass = compute_wasserstein(h_curves, human_curves)
        print_report("Human-biased (λ>0)", h_per_task, h_solve, h_mean_dtw, h_med_dtw, h_wass)

        print(f"\n{'='*50}")
        print("  Delta (biased - baseline)")
        print(f"{'='*50}")
        print(f"  Solve rate delta:       {h_solve - b_solve:+.3f}")
        print(f"  Mean DTW sim delta:     {h_mean_dtw - b_mean_dtw:+.4f}  ({'better' if h_mean_dtw > b_mean_dtw else 'worse'})")
        print(f"  Median DTW sim delta:   {h_med_dtw - b_med_dtw:+.4f}  ({'better' if h_med_dtw > b_med_dtw else 'worse'})")
        print(f"  Wasserstein delta:      {h_wass - b_wass:+.4f}  ({'better' if h_wass < b_wass else 'worse'})")

        # hypothesis test: Mann-Whitney U on DTW similarity distributions
        print(f"\n{'='*50}")
        print("  Hypothesis Test (Mann-Whitney U)")
        print(f"{'='*50}")
        stat, p_value = mannwhitneyu(h_sims, b_sims, alternative="greater")
        print(f"  H0: biased DTW sim <= baseline DTW sim")
        print(f"  U statistic: {stat:.1f}")
        print(f"  p-value:     {p_value:.4f}")
        if p_value < 0.05:
            print(f"  Result:      SIGNIFICANT (p < 0.05) — bias improved human-likeness")
        else:
            print(f"  Result:      NOT significant (p >= 0.05)")

        # per-task breakdown
        difficulty_map = load_difficulty()
        print(f"\n{'='*50}")
        print("  Per-task DTW sim (biased - baseline)")
        print(f"{'='*50}")
        common_tasks = set(b_per_task) & set(h_per_task)
        deltas = []
        for task_id in sorted(common_tasks):
            delta = h_per_task[task_id]["mean_dtw_sim"] - b_per_task[task_id]["mean_dtw_sim"]
            deltas.append(delta)
            cat = difficulty_map.get(task_id, "Unknown")
            print(f"  {task_id}  {delta:+.4f}  [{cat}]")
        print(f"\n  Mean per-task delta: {np.mean(deltas):+.4f}")
        print(f"  Tasks improved: {sum(d > 0 for d in deltas)}/{len(deltas)}")

        # plots
        os.makedirs(PLOTS_DIR, exist_ok=True)
        print(f"\nGenerating plots -> {PLOTS_DIR}/")
        plot_dtw_distributions(b_sims, h_sims, PLOTS_DIR)
        plot_per_task_delta(b_per_task, h_per_task, difficulty_map, PLOTS_DIR)
        plot_mean_curves(human_curves, b_curves, h_curves, PLOTS_DIR)


def load_difficulty():
    if os.path.exists(DIFFICULTY_CSV):
        df = pd.read_csv(DIFFICULTY_CSV)[["task_id", "difficulty_category"]]
        return dict(zip(df["task_id"], df["difficulty_category"]))
    return {}


def plot_dtw_distributions(b_sims, h_sims, out_dir):
    """Plot 2: DTW similarity distribution — baseline vs biased histogram."""
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 30)
    ax.hist(b_sims, bins=bins, alpha=0.6, label="Baseline (λ=0)", color="tab:blue")
    ax.hist(h_sims, bins=bins, alpha=0.6, label="Human-biased (λ>0)", color="tab:orange")
    ax.axvline(np.mean(b_sims), color="tab:blue", linestyle="--", linewidth=1.5)
    ax.axvline(np.mean(h_sims), color="tab:orange", linestyle="--", linewidth=1.5)
    ax.set_xlabel("DTW Similarity")
    ax.set_ylabel("Count")
    ax.set_title("DTW Similarity Distribution: Baseline vs Human-Biased")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "dtw_distribution.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved -> {path}")


def plot_per_task_delta(b_per_task, h_per_task, difficulty_map, out_dir):
    """Plot 1: Per-task DTW delta bar chart, coloured by difficulty category."""
    common_tasks = sorted(set(b_per_task) & set(h_per_task))
    deltas = [h_per_task[t]["mean_dtw_sim"] - b_per_task[t]["mean_dtw_sim"] for t in common_tasks]
    categories = [difficulty_map.get(t, "Unknown") for t in common_tasks]

    cat_colors = {
        "Easy for both": "tab:green",
        "Only hard for AI": "tab:orange",
        "Only hard for humans": "tab:blue",
        "Hard for both": "tab:red",
        "Unknown": "tab:gray",
    }
    colors = [cat_colors[c] for c in categories]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(common_tasks)), deltas, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(common_tasks)))
    ax.set_xticklabels(common_tasks, rotation=90, fontsize=7)
    ax.set_ylabel("DTW Similarity Delta (biased - baseline)")
    ax.set_title("Per-task DTW Similarity Improvement by Difficulty Category")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cat_colors.values()]
    ax.legend(handles, cat_colors.keys(), fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "per_task_delta.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved -> {path}")


def plot_mean_curves(human_curves, b_curves, h_curves, out_dir):
    """Plot 3: Mean progress curves — human vs baseline vs biased."""
    x = np.linspace(0, 1, 100)
    human_mean = np.mean([c for curves in human_curves.values() for c in curves], axis=0)
    b_mean = np.mean(b_curves, axis=0) if b_curves else None
    h_mean = np.mean(h_curves, axis=0) if h_curves else None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, human_mean, color="tab:blue", label="Human", linewidth=2)
    if b_mean is not None:
        ax.plot(x, b_mean, color="tab:gray", linestyle="--", label="CodeIt baseline (λ=0)", linewidth=2)
    if h_mean is not None:
        ax.plot(x, h_mean, color="tab:orange", label="CodeIt biased (λ>0)", linewidth=2)
    ax.set_xlabel("Normalised step (0=start, 1=end)")
    ax.set_ylabel("Progress (fraction of cells correct)")
    ax.set_title("Mean Progress Curves: Human vs CodeIt")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "mean_progress_curves.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved -> {path}")


if __name__ == "__main__":
    main()
