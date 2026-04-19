"""
Q2: Does human solving effort correlate with how hard it was for CodeIt?

Human effort per task:
  - avg_actions_to_solve : average total actions a participant spent before solving
    (sum of num_actions across all attempts up to and including the first solved attempt;
     only participants who eventually solved the task are counted)
  - avg_attempts_to_solve: average number of attempts before first solve
    (among participants who solved it)
  - overall_avg_actions  : average num_actions per attempt (all participants, all attempts)

CodeIt effort:
  - codeit_first_iter: the iteration when the first solution was found (from Q1 output)

Then compute Spearman correlation between human effort metrics and CodeIt iteration.
"""

import os, re
import pandas as pd
import numpy as np
from scipy import stats

# ── paths ─────────────────────────────────────────────────────────────────────
REPO    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMM    = os.path.join(REPO, "human_data/data/summary_data.csv")
OUT     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")
DIFF_IN = os.path.join(OUT, "task_difficulty.csv")   # produced by script 01

# ── load data ─────────────────────────────────────────────────────────────────
summ = pd.read_csv(SUMM)
summ["task_id"] = summ["task_name"].str.replace(".json", "", regex=False)

diff = pd.read_csv(DIFF_IN)[["task_id", "codeit_first_iter"]]
solved_tasks = set(diff["task_id"])
summ59 = summ[summ["task_id"].isin(solved_tasks)].copy()

# ── metric 1: effort among participants who eventually solved the task ─────────
# Sort by (task_id, participant, attempt_number) so attempts are in order
summ59 = summ59.sort_values(["task_id", "hashed_id", "attempt_number"])

# For each participant-task, find their first solved attempt index (1-based)
first_solve = (
    summ59[summ59["solved"]]
    .groupby(["task_id", "hashed_id"])["attempt_number"]
    .min()
    .reset_index()
    .rename(columns={"attempt_number": "first_solve_attempt"})
)

# Merge back to get all attempts up to and including the first solve
effort_rows = summ59.merge(first_solve, on=["task_id", "hashed_id"])
effort_rows = effort_rows[effort_rows["attempt_number"] <= effort_rows["first_solve_attempt"]]

# Sum actions per participant-task (total actions spent before/on solving)
total_effort = (
    effort_rows.groupby(["task_id", "hashed_id"])
    .agg(
        total_actions=("num_actions", "sum"),
        attempts_to_solve=("first_solve_attempt", "first"),
    )
    .reset_index()
)

# Average across participants who solved the task
human_effort = (
    total_effort.groupby("task_id")
    .agg(
        avg_actions_to_solve=("total_actions", "mean"),
        avg_attempts_to_solve=("attempts_to_solve", "mean"),
        n_solvers=("hashed_id", "count"),
    )
    .reset_index()
)

# ── metric 2: overall avg actions per attempt (all participants) ───────────────
overall = (
    summ59.groupby("task_id")
    .agg(
        overall_avg_actions=("num_actions", "mean"),
        total_attempts=("num_actions", "count"),
    )
    .reset_index()
)

# ── merge everything ──────────────────────────────────────────────────────────
effort = human_effort.merge(overall, on="task_id").merge(diff, on="task_id")

# ── Spearman correlations ─────────────────────────────────────────────────────
metrics = ["avg_actions_to_solve", "avg_attempts_to_solve", "overall_avg_actions"]
print("=== Q2: Human Effort vs CodeIt Iteration (Spearman correlation) ===\n")
for m in metrics:
    valid = effort[["codeit_first_iter", m]].dropna()
    r, p = stats.spearmanr(valid["codeit_first_iter"], valid[m])
    print(f"{m:30s}  rho={r:+.3f}  p={p:.4f}  (n={len(valid)})")

print("\nTask-level effort summary:")
print(effort[["task_id", "avg_actions_to_solve", "avg_attempts_to_solve",
              "overall_avg_actions", "codeit_first_iter"]].describe().to_string())

# ── save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT, "solving_effort.csv")
effort.to_csv(out_path, index=False)
print(f"\nSaved -> {out_path}")
