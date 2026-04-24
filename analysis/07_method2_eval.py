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

import numpy as np

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

    return per_task, solve_rate, mean_dtw, median_dtw, all_dtw_sims


def print_report(label, per_task, solve_rate, mean_dtw, median_dtw):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Tasks evaluated:        {len(per_task)}")
    print(f"  Solve rate:             {solve_rate:.3f}")
    print(f"  Mean DTW similarity:    {mean_dtw:.4f}")
    print(f"  Median DTW similarity:  {median_dtw:.4f}")


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
    b_per_task, b_solve, b_mean_dtw, b_med_dtw, b_sims = evaluate_solutions(
        args.baseline, human_curves
    )
    print_report("Baseline (λ=0)", b_per_task, b_solve, b_mean_dtw, b_med_dtw)

    if args.biased:
        print("\nEvaluating biased...")
        h_per_task, h_solve, h_mean_dtw, h_med_dtw, h_sims = evaluate_solutions(
            args.biased, human_curves
        )
        print_report("Human-biased (λ>0)", h_per_task, h_solve, h_mean_dtw, h_med_dtw)

        print(f"\n{'='*50}")
        print("  Delta (biased - baseline)")
        print(f"{'='*50}")
        print(f"  Solve rate delta:       {h_solve - b_solve:+.3f}")
        print(f"  Mean DTW sim delta:     {h_mean_dtw - b_mean_dtw:+.4f}  ({'better' if h_mean_dtw > b_mean_dtw else 'worse'})")
        print(f"  Median DTW sim delta:   {h_med_dtw - b_med_dtw:+.4f}  ({'better' if h_med_dtw > b_med_dtw else 'worse'})")

        # per-task breakdown
        print(f"\n{'='*50}")
        print("  Per-task DTW sim (biased - baseline)")
        print(f"{'='*50}")
        common_tasks = set(b_per_task) & set(h_per_task)
        deltas = []
        for task_id in sorted(common_tasks):
            delta = h_per_task[task_id]["mean_dtw_sim"] - b_per_task[task_id]["mean_dtw_sim"]
            deltas.append(delta)
            print(f"  {task_id}  {delta:+.4f}")
        print(f"\n  Mean per-task delta: {np.mean(deltas):+.4f}")
        print(f"  Tasks improved: {sum(d > 0 for d in deltas)}/{len(deltas)}")


if __name__ == "__main__":
    main()
