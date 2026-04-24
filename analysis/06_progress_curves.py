"""
Compute progress curves for all 59 tasks and save plots + raw data.

Progress(grid, target) = fraction of cells with matching colour.
  - Returns 0.0 if grid sizes differ.
  - Both grids are pipe-delimited strings: "|012|345|678|"

X-axis is normalised to [0, 1] by resampling each trajectory to 100 points
via linear interpolation, so curves of different lengths are comparable.

For each task, per-trajectory curves are collected in two forms:
  - raw      : resampled progress ∈ [0, 1], reflects absolute cell-match rate
  - normalised: (raw - raw[0]) / (1 - raw[0]), so every trajectory starts at 0

Both forms are aggregated via element-wise median across trajectories within
each of four groups (human success / failed, codeit success / failed).

Outputs:
  06_curves/progress_curves.json     — median normalised + raw median curves
  06_curves/curve_summary.csv        — per-task summary statistics
  06_curves/curves/{task_id}.png     — normalised median curves (y starts at 0)
  06_curves/origin_curves/{task_id}.png — raw median curves (y = absolute progress)
"""

import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUMAN_PATH = os.path.join(REPO, "analysis/processed/04_human_traces/human_traces.json")
CODEIT_PATH= os.path.join(REPO, "analysis/processed/05_codeit_traces/codeit_traces.json")
EVAL_DIR   = os.path.join(REPO, "codelt/data/evaluation")
CURVES_DIR  = os.path.join(REPO, "analysis/processed/06_curves/curves")
ORIGIN_DIR  = os.path.join(REPO, "analysis/processed/06_curves/origin_curves")
OUT_JSON    = os.path.join(REPO, "analysis/processed/06_curves/progress_curves.json")
OUT_SUMMARY = os.path.join(REPO, "analysis/processed/06_curves/curve_summary.csv")
N_POINTS    = 100   # x-axis resolution after resampling

os.makedirs(CURVES_DIR, exist_ok=True)
os.makedirs(ORIGIN_DIR, exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────────

def parse_grid(grid_str):
    """Convert pipe-delimited string to list of lists of ints."""
    rows = grid_str.strip("|").split("|")
    return [[int(c) for c in row] for row in rows]

def progress(grid_str, target_str):
    """Fraction of cells matching target. 0.0 if sizes differ or parse error."""
    try:
        g = parse_grid(grid_str)
        t = parse_grid(target_str)
        if len(g) != len(t) or any(len(gr) != len(tr) for gr, tr in zip(g, t)):
            return 0.0
        total = sum(len(row) for row in t)
        match = sum(g[r][c] == t[r][c] for r in range(len(t)) for c in range(len(t[r])))
        return match / total if total > 0 else 0.0
    except Exception:
        return 0.0

def resample(curve, n=N_POINTS):
    """Linearly resample a list of floats to exactly n points."""
    x_old = np.linspace(0, 1, len(curve))
    x_new = np.linspace(0, 1, n)
    return np.interp(x_new, x_old, curve).tolist()

def normalise_curve(curve):
    """Shift and scale so curve starts at 0 and max possible end is 1.
    norm(t) = (curve(t) - curve(0)) / (1 - curve(0))
    If curve(0) == 1.0, the task is already solved from the start — return all-ones.
    """
    start = curve[0]
    if start >= 1.0:
        return [1.0] * len(curve)
    denom = 1.0 - start
    return [(v - start) / denom for v in curve]

def mean_curves(curves):
    """Element-wise mean of a list of length-n lists. Returns None if empty."""
    if not curves:
        return None
    arr = np.array(curves)
    return arr.mean(axis=0).tolist()

def median_curves(curves):
    """Element-wise median of a list of length-n lists. Robust to outlier trajectories."""
    if not curves:
        return None
    arr = np.array(curves)
    return np.median(arr, axis=0).tolist()

def load_target_grid(task_id):
    path = os.path.join(EVAL_DIR, f"{task_id}.json")
    with open(path) as f:
        task = json.load(f)
    rows = task["test_examples"][0]["output"]
    return "|" + "|".join("".join(str(c) for c in row) for row in rows) + "|"

# ── load traces ────────────────────────────────────────────────────────────────

with open(HUMAN_PATH)  as f: human_traces  = json.load(f)
with open(CODEIT_PATH) as f: codeit_traces = json.load(f)

all_task_ids = sorted(set(human_traces) | set(codeit_traces))
all_curves   = {}

for task_id in all_task_ids:
    target = load_target_grid(task_id)

    # ── collect raw + normalised curves for each group ────────────────────────
    # raw       : resampled progress in [0,1] — absolute cell-match rate
    # normalised: (raw - raw[0]) / (1 - raw[0]) — starts at 0 for every traj
    human_success_norm, human_success_raw = [], []
    human_failed_norm,  human_failed_raw  = [], []
    for traj in human_traces.get(task_id, []):
        raw = resample([progress(g, target) for g in traj["grids"]])
        norm = normalise_curve(raw)
        if traj["success"]:
            human_success_raw.append(raw)
            human_success_norm.append(norm)
        else:
            human_failed_raw.append(raw)
            human_failed_norm.append(norm)

    codeit_success_norm, codeit_success_raw = [], []
    codeit_failed_norm,  codeit_failed_raw  = [], []
    for traj in codeit_traces.get(task_id, []):
        if not traj["grids"]:
            continue
        raw = resample([progress(g, target) for g in traj["grids"]])
        norm = normalise_curve(raw)
        if traj["class"] == "success":
            codeit_success_raw.append(raw)
            codeit_success_norm.append(norm)
        else:
            codeit_failed_raw.append(raw)
            codeit_failed_norm.append(norm)

    entry = {
        # normalised median curves (primary analysis)
        "human_success_median":  median_curves(human_success_norm),
        "human_failed_median":   median_curves(human_failed_norm),
        "codeit_success_median": median_curves(codeit_success_norm),
        "codeit_failed_median":  median_curves(codeit_failed_norm),
        # raw median curves (absolute progress, for reference)
        "human_success_raw_median":  median_curves(human_success_raw),
        "human_failed_raw_median":   median_curves(human_failed_raw),
        "codeit_success_raw_median": median_curves(codeit_success_raw),
        "codeit_failed_raw_median":  median_curves(codeit_failed_raw),
        # trajectory counts
        "human_success_n":  len(human_success_norm),
        "human_failed_n":   len(human_failed_norm),
        "codeit_success_n": len(codeit_success_norm),
        "codeit_failed_n":  len(codeit_failed_norm),
    }
    all_curves[task_id] = entry

    # ── plot ──────────────────────────────────────────────────────────────────
    x = np.linspace(0, 1, N_POINTS)
    fig, ax = plt.subplots(figsize=(7, 4))

    plot_spec = [
        ("human_success_median",  "human_success_n",  "tab:blue",   "Human success"),
        ("human_failed_median",   "human_failed_n",   "tab:cyan",   "Human failed"),
        ("codeit_success_median", "codeit_success_n", "tab:orange", "CodeIt success"),
        ("codeit_failed_median",  "codeit_failed_n",  "tab:red",    "CodeIt failed"),
    ]
    any_plotted = False
    y_min = 0.0
    for key, n_key, color, label in plot_spec:
        curve = entry[key]
        n     = entry[n_key]
        if curve is not None and n > 0:
            ax.plot(x, curve, color=color, label=f"{label} (n={n})", linewidth=2)
            any_plotted = True
            y_min = min(y_min, min(curve))

    if any_plotted:
        ax.set_xlabel("Normalised step (0=start, 1=end)")
        ax.set_ylabel("Normalised progress (median across trajectories)")
        ax.set_title(f"Task {task_id}")
        bottom = min(y_min - 0.05, -0.05) if y_min < 0 else -0.05
        ax.set_ylim(bottom, 1.05)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(CURVES_DIR, f"{task_id}.png"), dpi=120)
    plt.close()

    # ── origin (raw) curves plot ──────────────────────────────────────────────
    origin_spec = [
        ("human_success_raw_median",  "human_success_n",  "tab:blue",   "Human success"),
        ("human_failed_raw_median",   "human_failed_n",   "tab:cyan",   "Human failed"),
        ("codeit_success_raw_median", "codeit_success_n", "tab:orange", "CodeIt success"),
        ("codeit_failed_raw_median",  "codeit_failed_n",  "tab:red",    "CodeIt failed"),
    ]
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    any_origin = False
    for key, n_key, color, label in origin_spec:
        curve = entry[key]
        n     = entry[n_key]
        if curve is not None and n > 0:
            ax2.plot(x, curve, color=color, label=f"{label} (n={n})", linewidth=2)
            any_origin = True
    if any_origin:
        ax2.set_xlabel("Normalised step (0=start, 1=end)")
        ax2.set_ylabel("Raw progress (fraction of cells correct)")
        ax2.set_title(f"Task {task_id} — raw (no normalisation)")
        ax2.set_ylim(-0.02, 1.05)
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(ORIGIN_DIR, f"{task_id}.png"), dpi=120)
    plt.close()

# Save JSON
with open(OUT_JSON, "w") as f:
    json.dump(all_curves, f, indent=2)

# ── per-task summary CSV ───────────────────────────────────────────────────────
summary_rows = []
for task_id, entry in all_curves.items():
    row = {"task_id": task_id}
    for group, median_key, n_key in [
        ("human_success",  "human_success_median",  "human_success_n"),
        ("human_failed",   "human_failed_median",   "human_failed_n"),
        ("codeit_success", "codeit_success_median", "codeit_success_n"),
        ("codeit_failed",  "codeit_failed_median",  "codeit_failed_n"),
    ]:
        n = entry.get(n_key, 0)
        curve = entry.get(median_key)
        row[f"{group}_n"] = n
        row[f"{group}_final_progress"]  = round(curve[-1], 4)           if curve and n > 0 else float("nan")
        row[f"{group}_median_progress"] = round(float(np.median(curve)), 4) if curve and n > 0 else float("nan")
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
col_order = [
    "task_id",
    "human_success_n", "human_success_final_progress", "human_success_median_progress",
    "human_failed_n",  "human_failed_final_progress",  "human_failed_median_progress",
    "codeit_success_n","codeit_success_final_progress","codeit_success_median_progress",
    "codeit_failed_n", "codeit_failed_final_progress", "codeit_failed_median_progress",
]
summary_df[col_order].to_csv(OUT_SUMMARY, index=False)

print(f"Normalised plots  -> {CURVES_DIR}/")
print(f"Raw origin plots  -> {ORIGIN_DIR}/")
print(f"Curve data        -> {OUT_JSON}")
print(f"Curve summary     -> {OUT_SUMMARY}")
print(f"Tasks processed: {len(all_curves)}")
