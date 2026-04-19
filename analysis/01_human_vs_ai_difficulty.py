"""
Q1: Of the 59 tasks CodeIt solved, what fraction did humans also solve?
    Which tasks are hard for both, only for humans, or only for the AI?

Approach:
  - Human solve rate per task = (# unique participants who solved the task at least once)
                                 / (# unique participants who attempted the task)
  - CodeIt difficulty proxy = the earliest iteration at which any solution was found
    (extracted from the `new_task_key` field, e.g. "00576224_it81__21" -> iteration 81)
  - Classify each of the 59 tasks into four quadrants using median thresholds:
      "Easy for both"   : high human rate  + early CodeIt iteration
      "Only hard for AI": high human rate  + late  CodeIt iteration
      "Only hard for humans": low human rate + early CodeIt iteration
      "Hard for both"   : low human rate  + late  CodeIt iteration
"""

import json, re, os
import pandas as pd
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
REPO   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMM   = os.path.join(REPO, "human_data/data/summary_data.csv")
SOL    = os.path.join(REPO, "codelt/data/solutions_100.json")
OUT    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")
os.makedirs(OUT, exist_ok=True)

# ── load CodeIt solutions ─────────────────────────────────────────────────────
with open(SOL) as f:
    test_solutions = json.load(f)["policy"]["test"]   # {task_id -> {program -> {...}}}

def first_iteration(programs: dict) -> int:
    """Return the smallest iteration number across all programs found for a task."""
    iters = []
    for meta in programs.values():
        m = re.search(r"_it(\d+)__", meta.get("new_task_key", ""))
        if m:
            iters.append(int(m.group(1)))
    return min(iters) if iters else 99   # default to 99 if parsing fails

codeit_iters = {tid: first_iteration(progs) for tid, progs in test_solutions.items()}

# ── load human summary data (attempt level) ───────────────────────────────────
summ = pd.read_csv(SUMM)
summ["task_id"] = summ["task_name"].str.replace(".json", "", regex=False)

# Keep only the 59 tasks
solved_tasks = set(test_solutions.keys())
summ59 = summ[summ["task_id"].isin(solved_tasks)].copy()

# ── compute human solve rate per task ─────────────────────────────────────────
# For each (task_id, participant) pair: did they solve it in ANY attempt?
participant_solved = (
    summ59.groupby(["task_id", "hashed_id"])["solved"]
    .any()                     # True if any attempt was solved
    .reset_index()
)

human_stats = (
    participant_solved
    .groupby("task_id")
    .agg(
        n_participants=("hashed_id", "count"),
        n_solved=("solved", "sum"),
    )
    .assign(human_solve_rate=lambda d: d["n_solved"] / d["n_participants"])
    .reset_index()
)

# ── merge with CodeIt iteration ───────────────────────────────────────────────
ai_df = pd.DataFrame(
    [(tid, it) for tid, it in codeit_iters.items()],
    columns=["task_id", "codeit_first_iter"]
)
merged = human_stats.merge(ai_df, on="task_id", how="outer")

# ── classify difficulty ───────────────────────────────────────────────────────
med_human = merged["human_solve_rate"].median()
med_iter  = merged["codeit_first_iter"].median()

def classify(row):
    human_easy = row["human_solve_rate"] >= med_human
    ai_easy    = row["codeit_first_iter"] <= med_iter
    if human_easy and ai_easy:
        return "Easy for both"
    elif human_easy and not ai_easy:
        return "Only hard for AI"
    elif not human_easy and ai_easy:
        return "Only hard for humans"
    else:
        return "Hard for both"

merged["difficulty_category"] = merged.apply(classify, axis=1)

# ── print summary ─────────────────────────────────────────────────────────────
print("=== Q1: Human vs AI Difficulty (59 tasks) ===")
print(f"Median human solve rate : {med_human:.2%}")
print(f"Median CodeIt first iter: {med_iter:.0f}\n")
print("Difficulty category counts:")
print(merged["difficulty_category"].value_counts().to_string())
print("\nPer-category examples (task_id, human_rate, first_iter):")
for cat in merged["difficulty_category"].unique():
    subset = merged[merged["difficulty_category"] == cat][
        ["task_id", "human_solve_rate", "codeit_first_iter"]
    ].head(5)
    print(f"\n[{cat}]")
    print(subset.to_string(index=False))

# ── save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT, "task_difficulty.csv")
merged.to_csv(out_path, index=False)
print(f"\nSaved -> {out_path}")
