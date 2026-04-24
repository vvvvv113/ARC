"""
Extract human grid trajectory sequences for the 59 CodeIt-solved tasks.

For each (task_id, participant):
  - Keep only their last attempt (highest attempt_number)
  - Collect test_output_grid per action_id in order — this is the working output
    grid being built, which changes with each edit/floodfill/reset/paste action
  - Deduplicate consecutive identical grids (actions like change_color do not
    modify the grid and would add redundant frames)
  - Label success=True if solved==True appears anywhere in that last attempt

Output: analysis/processed/human_traces.json
  { task_id: [ {hashed_id, success, grids: [str, ...]}, ... ], ... }
"""

import json, os
import pandas as pd

REPO    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA    = os.path.join(REPO, "analysis/processed/00_shared_tasks/human_data_solved_tasks.csv")
OUT     = os.path.join(REPO, "analysis/processed/04_human_traces/human_traces.json")

df = pd.read_csv(DATA)

# Keep only last attempt per (task_id, participant)
last_attempt = (
    df.groupby(["task_id", "hashed_id"])["attempt_number"]
    .max()
    .reset_index()
    .rename(columns={"attempt_number": "last_attempt"})
)
df = df.merge(last_attempt, on=["task_id", "hashed_id"])
df = df[df["attempt_number"] == df["last_attempt"]]

# Sort actions within each attempt
df = df.sort_values(["task_id", "hashed_id", "action_id"])

traces = {}

for (task_id, hashed_id), group in df.groupby(["task_id", "hashed_id"]):
    grids_raw = group["test_output_grid"].tolist()
    success   = bool(group["solved"].any())

    # Deduplicate consecutive identical grids
    grids_dedup = []
    for g in grids_raw:
        if not grids_dedup or g != grids_dedup[-1]:
            grids_dedup.append(g)

    entry = {"hashed_id": hashed_id, "success": success, "grids": grids_dedup}
    traces.setdefault(task_id, []).append(entry)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(traces, f)

# Summary
total_traces = sum(len(v) for v in traces.values())
success_count = sum(e["success"] for v in traces.values() for e in v)
print(f"Tasks: {len(traces)}")
print(f"Total trajectories: {total_traces}  (success={success_count}, failed={total_traces-success_count})")
avg_len = sum(len(e['grids']) for v in traces.values() for e in v) / total_traces
print(f"Avg trajectory length (deduped): {avg_len:.1f} grids")
print(f"Saved -> {OUT}")
