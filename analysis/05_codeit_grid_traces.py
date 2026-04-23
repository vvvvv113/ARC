"""
Execute CodeIt DSL programs line-by-line and capture intermediate grid states
for the 59 tasks in the test split of solutions_100.json.

Programs are collected from ALL splits (seen_example, task_demonstration, test)
for these 59 tasks, then classified:
  "success" : test_performance[0] == True
  "failed"  : test_performance[0] == False (or test_performance missing/empty)

Uses execute_candidate_program_with_trace() added to environment.py.
The task input grid comes from codelt/data/evaluation/{task_id}.json -> test[0]["input"].

Usage:
  python3 analysis/05_codeit_grid_traces.py                  # full run
  python3 analysis/05_codeit_grid_traces.py --test-task 00576224  # single task

Output: analysis/processed/codeit_traces.json
  { task_id: [ {program, class, grids: [...]}, ... ], ... }
"""

import json, os, sys, argparse

# Make codeit package importable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "codelt"))

from codeit.policy.environment import execute_candidate_program_with_trace
from codeit.augment.mutate_grid import valid_grid

SOL_PATH  = os.path.join(REPO, "codelt/data/solutions_100.json")
EVAL_DIR  = os.path.join(REPO, "codelt/data/evaluation")
OUT_PATH  = os.path.join(REPO, "analysis/processed/codeit_traces.json")

# Parse optional single-task flag
parser = argparse.ArgumentParser()
parser.add_argument("--test-task", default=None, help="Run only this task_id")
args = parser.parse_args()

# Load solutions
with open(SOL_PATH) as f:
    policy = json.load(f)["policy"]

test_task_ids = set(policy["test"].keys())

# Collect all programs for the 59 tasks across all splits
def collect_programs(task_id):
    """Return list of {program, class, test_performance} for a given task."""
    programs = {}
    for split in ("seen_example", "task_demonstration", "test"):
        if task_id not in policy[split]:
            continue
        for prog_str, meta in policy[split][task_id].items():
            if prog_str in programs:
                continue  # same program may appear in multiple splits
            tp = meta.get("test_performance", [])
            success = bool(tp and tp[0] is True)
            programs[prog_str] = {"program": prog_str, "class": "success" if success else "failed"}
    return list(programs.values())

# Load task input grid from evaluation JSON
def load_input_grid(task_id):
    path = os.path.join(EVAL_DIR, f"{task_id}.json")
    with open(path) as f:
        task = json.load(f)
    # codelt preprocessed format uses "test_examples" not "test"
    raw = task["test_examples"][0]["input"]
    return tuple(tuple(row) for row in raw)

# Serialise grid (tuple of tuples) to pipe-delimited string for JSON storage
def grid_to_str(grid):
    return "|" + "|".join("".join(str(c) for c in row) for row in grid) + "|"

# Determine which tasks to run
if args.test_task:
    tasks_to_run = [args.test_task]
    print(f"Test mode: running single task {args.test_task}")
else:
    tasks_to_run = sorted(test_task_ids)

results = {}
total_programs = sum(len(collect_programs(t)) for t in tasks_to_run)

try:
    from tqdm import tqdm
    use_tqdm = True
except ImportError:
    use_tqdm = False

processed = 0
errors = 0

for task_id in tasks_to_run:
    programs = collect_programs(task_id)
    input_grid = load_input_grid(task_id)
    task_traces = []

    for prog_meta in programs:
        prog_str   = prog_meta["program"]
        prog_class = prog_meta["class"]

        output, trace = execute_candidate_program_with_trace(prog_str, input_grid)

        if isinstance(output, str) and output.startswith("Error"):
            errors += 1
            processed += 1
            continue  # skip programs that error during execution

        # Convert tuple grids in trace to pipe-delimited strings for JSON
        grids_serialised = [grid_to_str(g) for (_, g) in trace if valid_grid(g)]

        task_traces.append({
            "program": prog_str,
            "class":   prog_class,
            "grids":   grids_serialised,
        })
        processed += 1

        if use_tqdm:
            pass  # tqdm handles display
        elif processed % 500 == 0:
            print(f"  {processed}/{total_programs} programs done, {errors} errors")

    results[task_id] = task_traces
    print(f"[{task_id}] {len(task_traces)} programs traced "
          f"(success={sum(1 for t in task_traces if t['class']=='success')}, "
          f"failed={sum(1 for t in task_traces if t['class']=='failed')})")

with open(OUT_PATH, "w") as f:
    json.dump(results, f)

print(f"\nTotal programs traced: {processed}  Errors skipped: {errors}")
print(f"Saved -> {OUT_PATH}")
