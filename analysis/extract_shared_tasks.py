import json
import os
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLUTIONS_PATH = os.path.join(REPO_ROOT, "codelt/data/solutions_100.json")
HUMAN_DATA_PATH = os.path.join(REPO_ROOT, "human_data/data/data.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed", "00_shared_tasks")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the 59 CodeIt-solved task IDs
with open(SOLUTIONS_PATH) as f:
    solutions = json.load(f)
solved_tasks = set(solutions["policy"]["test"].keys())
print(f"CodeIt solved tasks: {len(solved_tasks)}")

# Save task ID list for reuse
task_ids_path = os.path.join(OUTPUT_DIR, "solved_task_ids.json")
with open(task_ids_path, "w") as f:
    json.dump(sorted(solved_tasks), f, indent=2)
print(f"Saved task IDs -> {task_ids_path}")

# Load human data and filter to the 59 tasks
print("Loading human data (this may take a moment)...")
df = pd.read_csv(HUMAN_DATA_PATH)
df["task_id"] = df["task_name"].str.replace(".json", "", regex=False)

human_on_solved = df[df["task_id"].isin(solved_tasks)].copy()

# Summary
print(f"\n--- Summary ---")
print(f"Total rows in human data:          {len(df):,}")
print(f"Rows for CodeIt-solved tasks:      {len(human_on_solved):,}")
print(f"Unique tasks in filtered data:     {human_on_solved['task_id'].nunique()}")
print(f"Unique participants:               {human_on_solved['hashed_id'].nunique()}")
print(f"Tasks with human data:             {sorted(human_on_solved['task_id'].unique())}")

out_path = os.path.join(OUTPUT_DIR, "human_data_solved_tasks.csv")
human_on_solved.to_csv(out_path, index=False)
print(f"\nSaved filtered human data -> {out_path}")
