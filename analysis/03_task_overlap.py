"""
Q3: Are the tasks humans solve and the tasks CodeIt solves the same set?

Compare across all tasks that appear in the human dataset (800 tasks total):
  - human_solved_tasks : tasks where at least one participant solved it
  - codeit_solved_tasks: the 59 tasks from solutions_100.json (test split)

Then classify each task into one of four groups:
  A. Both solved    : human ✓, CodeIt ✓
  B. Only humans    : human ✓, CodeIt ✗
  C. Only CodeIt    : human ✗, CodeIt ✓
  D. Neither solved : human ✗, CodeIt ✗  (not in our 800-task human set)

Note: CodeIt only ran on the ARC *evaluation* set (~400 tasks);
      human data covers both training and evaluation tasks.
      We separate task_type to avoid mixing the two pools.
"""

import json, os
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMM = os.path.join(REPO, "human_data/data/summary_data.csv")
SOL  = os.path.join(REPO, "codelt/data/solutions_100.json")
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")
os.makedirs(OUT, exist_ok=True)

# ── CodeIt solved tasks ───────────────────────────────────────────────────────
with open(SOL) as f:
    all_solutions = json.load(f)["policy"]
codeit_test   = set(all_solutions["test"].keys())         # 59 tasks (passed test)
codeit_seenex = set(all_solutions["seen_example"].keys()) # 114 tasks (seen examples only)
codeit_any    = set(all_solutions["test"].keys())         # we focus on true solved = test

# ── human solved tasks (from full summary data) ───────────────────────────────
summ = pd.read_csv(SUMM)
summ["task_id"] = summ["task_name"].str.replace(".json", "", regex=False)

# Per task: did ANY participant solve it?
human_task_solved = (
    summ.groupby(["task_id", "task_type"])["solved"]
    .any()
    .reset_index()
    .rename(columns={"solved": "human_solved"})
)

# ── build overlap table for evaluation tasks only ────────────────────────────
eval_human = human_task_solved[human_task_solved["task_type"] == "evaluation"].copy()
eval_human["codeit_solved"] = eval_human["task_id"].isin(codeit_test)

def categorize(row):
    if row["human_solved"] and row["codeit_solved"]:
        return "Both solved"
    elif row["human_solved"] and not row["codeit_solved"]:
        return "Only humans solved"
    elif not row["human_solved"] and row["codeit_solved"]:
        return "Only CodeIt solved"
    else:
        return "Neither solved"

eval_human["overlap_category"] = eval_human.apply(categorize, axis=1)

# ── also include any CodeIt-solved tasks not in human eval data ───────────────
in_human_eval = set(eval_human["task_id"])
codeit_only_extra = codeit_test - in_human_eval
extra_rows = pd.DataFrame({
    "task_id": list(codeit_only_extra),
    "task_type": "evaluation",
    "human_solved": False,
    "codeit_solved": True,
    "overlap_category": "Only CodeIt solved",
})
overlap = pd.concat([eval_human, extra_rows], ignore_index=True)

# ── print results ─────────────────────────────────────────────────────────────
print("=== Q3: Task Overlap — Humans vs CodeIt (evaluation tasks) ===\n")
counts = overlap["overlap_category"].value_counts()
print(counts.to_string())
total_eval = len(overlap)
print(f"\nTotal evaluation tasks in analysis: {total_eval}")

# Human solve rate across all eval tasks
n_human = int(eval_human["human_solved"].sum())
print(f"Evaluation tasks humans solved (any participant): {n_human}")
print(f"Evaluation tasks CodeIt solved (test split):      {len(codeit_test)}")
print(f"Overlap (both solved):                            {int((overlap['overlap_category']=='Both solved').sum())}")

# Show tasks only CodeIt solved (humans struggled on all of them)
codeit_only = overlap[overlap["overlap_category"] == "Only CodeIt solved"]["task_id"].tolist()
print(f"\nTasks only CodeIt solved ({len(codeit_only)} tasks):")
print(sorted(codeit_only))

# Chi-square / Fisher test: are CodeIt solutions correlated with human solutions?
from scipy.stats import fisher_exact
both = int((overlap["overlap_category"] == "Both solved").sum())
only_human = int((overlap["overlap_category"] == "Only humans solved").sum())
only_ai = int((overlap["overlap_category"] == "Only CodeIt solved").sum())
neither = int((overlap["overlap_category"] == "Neither solved").sum())

table = [[both, only_human], [only_ai, neither]]
odds, p = fisher_exact(table)
print(f"\nFisher's exact test (CodeIt solved ~ Human solved):")
print(f"  Odds ratio = {odds:.3f},  p = {p:.4f}")
if p < 0.05:
    print("  -> Significant association: tasks CodeIt solves tend to overlap with human-solved tasks.")
else:
    print("  -> No significant association: the two sets are largely independent.")

# ── save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT, "task_overlap.csv")
overlap.to_csv(out_path, index=False)
print(f"\nSaved -> {out_path}")
