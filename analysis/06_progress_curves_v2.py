"""
Recompute all progress curves using a unified task-level normalization baseline (v2).

Problem with v1:
  norm_v1(t) = (progress(t) - progress(0)) / (1 - progress(0))
  Each agent's progress(0) means something different:
    - CodeIt : progress(input_grid, target)  — task property
    - Human  : progress(blank_output, target) — fraction of target that is background color
  Forcing both to 0 erases a meaningful asymmetry.

v2 formula (unified baseline):
  norm_v2(t) = (progress(t) - baseline) / (1 - baseline)
  baseline   = progress(input_grid, target_grid)  — same for both agents

Interpretation:
  norm = 0   →  input_grid  (unified zero point for all agents)
  norm = 1   →  target_grid (goal)
  norm > 0   →  better than input
  norm < 0   →  worse than input (e.g. human's blank output on a task where
               input already overlaps target, or backtracking past the baseline)

Aggregation:  per-trajectory normalization → element-wise median across group.
              Also stores p25 and p75 (IQR) for each group.

Outputs (analysis/processed/06_curves/):
  progress_curves_v2.json    — v2 curves: metadata + per-task baseline/p25/median/p75
  curve_summary_v2.csv       — per-task flat summary (start, end, AUC, monotonic flag)
  curve_v2/{task_id}.png     — plots with IQR shading, baseline reference line
  validation_report.txt      — automated sanity checks
  migration_summary.md       — v1 vs v2 starting-point comparison for key tasks
"""

import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUMAN_PATH  = os.path.join(REPO, "analysis/processed/04_human_traces/human_traces.json")
CODEIT_PATH = os.path.join(REPO, "analysis/processed/05_codeit_traces/codeit_traces.json")
EVAL_DIR    = os.path.join(REPO, "codelt/data/evaluation")
OUT_DIR     = os.path.join(REPO, "analysis/processed/06_curves")
CURVES_V1   = os.path.join(OUT_DIR, "progress_curves.json")

OUT_JSON    = os.path.join(OUT_DIR, "progress_curves_v2.json")
OUT_SUMMARY = os.path.join(OUT_DIR, "curve_summary_v2.csv")
CURVE_V2_DIR= os.path.join(OUT_DIR, "curve_v2")
VAL_REPORT  = os.path.join(OUT_DIR, "validation_report.txt")
MIG_SUMMARY = os.path.join(OUT_DIR, "migration_summary.md")

N_POINTS = 100

os.makedirs(CURVE_V2_DIR, exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────────

def parse_grid(grid_str):
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
    x_old = np.linspace(0, 1, len(curve))
    x_new = np.linspace(0, 1, n)
    return np.interp(x_new, x_old, curve).tolist()

def normalise_v2(raw_curve, baseline):
    """norm(t) = (progress(t) - baseline) / (1 - baseline)"""
    denom = 1.0 - baseline
    if denom <= 0:
        # input_grid already equals target — everything is at goal
        return [1.0] * len(raw_curve)
    return [(v - baseline) / denom for v in raw_curve]

def load_grids(task_id):
    """Return (input_str, target_str) for a task."""
    path = os.path.join(EVAL_DIR, f"{task_id}.json")
    with open(path) as f:
        task = json.load(f)
    inp_rows = task["test_examples"][0]["input"]
    out_rows = task["test_examples"][0]["output"]
    inp_str = "|" + "|".join("".join(str(c) for c in row) for row in inp_rows) + "|"
    tgt_str = "|" + "|".join("".join(str(c) for c in row) for row in out_rows) + "|"
    return inp_str, tgt_str

def compute_group(raw_curves, baseline):
    """Normalise all raw curves with task baseline, return stats dict or None."""
    if not raw_curves:
        return None
    norm_arr = np.array([normalise_v2(c, baseline) for c in raw_curves])  # (N, 100)
    median      = np.median(norm_arr, axis=0).tolist()
    p25         = np.percentile(norm_arr, 25, axis=0).tolist()
    p75         = np.percentile(norm_arr, 75, axis=0).tolist()
    is_monotonic = all(median[t] <= median[t + 1] for t in range(len(median) - 1))
    return {
        "median_curve": median,
        "p25_curve":    p25,
        "p75_curve":    p75,
        "sample_size":  len(raw_curves),
        "is_monotonic": is_monotonic,
    }

# ── load traces ────────────────────────────────────────────────────────────────

with open(HUMAN_PATH)  as f: human_traces  = json.load(f)
with open(CODEIT_PATH) as f: codeit_traces = json.load(f)

all_task_ids = sorted(set(human_traces) | set(codeit_traces))
all_tasks    = {}
x            = np.linspace(0, 1, N_POINTS)

# ── main loop ──────────────────────────────────────────────────────────────────

for task_id in all_task_ids:
    inp_str, tgt_str = load_grids(task_id)
    baseline = progress(inp_str, tgt_str)
    denom    = 1.0 - baseline

    # collect raw resampled progress curves per group
    human_success_raw, human_failed_raw   = [], []
    codeit_success_raw, codeit_failed_raw = [], []

    for traj in human_traces.get(task_id, []):
        raw = resample([progress(g, tgt_str) for g in traj["grids"]])
        if traj["success"]:
            human_success_raw.append(raw)
        else:
            human_failed_raw.append(raw)

    for traj in codeit_traces.get(task_id, []):
        if not traj["grids"]:
            continue
        raw = resample([progress(g, tgt_str) for g in traj["grids"]])
        if traj["class"] == "success":
            codeit_success_raw.append(raw)
        else:
            codeit_failed_raw.append(raw)

    human_success  = compute_group(human_success_raw,  baseline)
    human_failed   = compute_group(human_failed_raw,   baseline)
    codeit_success = compute_group(codeit_success_raw, baseline)
    codeit_failed  = compute_group(codeit_failed_raw,  baseline)

    task_entry = {
        "baseline":    round(baseline, 6),
        "denominator": round(denom, 6),
    }
    for name, grp in [
        ("human_success",  human_success),
        ("human_failed",   human_failed),
        ("codeit_success", codeit_success),
        ("codeit_failed",  codeit_failed),
    ]:
        if grp is not None:
            task_entry[name] = grp

    all_tasks[task_id] = task_entry

    # ── curve_v2 plot (IQR shading, baseline reference) ───────────────────────
    plot_spec = [
        ("human_success",  "tab:blue",   "Human success"),
        ("human_failed",   "tab:cyan",   "Human failed"),
        ("codeit_success", "tab:orange", "CodeIt success"),
        ("codeit_failed",  "tab:red",    "CodeIt failed"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    any_plotted = False
    y_min, y_max = 0.0, 1.0

    for key, color, label in plot_spec:
        grp = task_entry.get(key)
        if grp is None:
            continue
        n    = grp["sample_size"]
        med  = np.array(grp["median_curve"])
        p25a = np.array(grp["p25_curve"])
        p75a = np.array(grp["p75_curve"])
        ax.fill_between(x, p25a, p75a, color=color, alpha=0.15)
        ax.plot(x, med, color=color, label=f"{label} (n={n})", linewidth=2)
        any_plotted = True
        y_min = min(y_min, float(p25a.min()))
        y_max = max(y_max, float(p75a.max()))

    if any_plotted:
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5,
                   label=f"input baseline (={baseline:.2f})")
        ax.axhline(1, color="gray",  linewidth=0.6, linestyle=":",  alpha=0.4)
        ax.set_ylim(min(y_min - 0.05, -0.05), max(y_max + 0.05, 1.05))
        ax.set_xlabel("Normalised step (0=start, 1=end)")
        ax.set_ylabel("Normalised progress v2 (IQR shaded)")
        ax.set_title(f"Task {task_id}  [baseline={baseline:.3f}]")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(CURVE_V2_DIR, f"{task_id}.png"), dpi=120)
    plt.close()

# ── save JSON ─────────────────────────────────────────────────────────────────

output = {
    "metadata": {
        "normalization_method":  "task_baseline_v2",
        "normalization_formula": "norm(t) = (progress(t) - baseline) / (1 - baseline)",
        "baseline_definition":   "progress(input_grid, target_grid)",
        "description": (
            "Unified zero point for all agents: norm=0 corresponds to the task "
            "input_grid, norm=1 to the target. Human blank-output advantage and "
            "CodeIt input-start are now measured against the same reference."
        ),
    },
    "tasks": all_tasks,
}
with open(OUT_JSON, "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved v2 JSON -> {OUT_JSON}")

# ── curve_summary_v2.csv ──────────────────────────────────────────────────────

summary_rows = []
for task_id, td in all_tasks.items():
    row = {"task_id": task_id, "baseline": td["baseline"], "denominator": td["denominator"]}
    for grp_name in ["human_success", "human_failed", "codeit_success", "codeit_failed"]:
        grp = td.get(grp_name)
        if grp:
            med = grp["median_curve"]
            row[f"{grp_name}_n"]            = grp["sample_size"]
            row[f"{grp_name}_start"]        = round(med[0], 4)
            row[f"{grp_name}_end"]          = round(med[-1], 4)
            row[f"{grp_name}_auc"]          = round(float(np.trapezoid(med, dx=1.0/(N_POINTS-1))), 4)
            row[f"{grp_name}_is_monotonic"] = grp["is_monotonic"]
        else:
            for sfx in ["_n", "_start", "_end", "_auc", "_is_monotonic"]:
                row[f"{grp_name}{sfx}"] = float("nan")
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_SUMMARY, index=False)
print(f"Saved summary   -> {OUT_SUMMARY}")

# ── validation ────────────────────────────────────────────────────────────────

issues = []
for task_id, td in all_tasks.items():
    b = td["baseline"]

    # 1. baseline in valid range
    if not (0.0 <= b < 1.0):
        issues.append(f"BASELINE_RANGE   {task_id}: baseline={b:.4f}")

    # 2. CodeIt first point must be 0 (input grid = baseline by definition)
    for grp_name in ["codeit_success", "codeit_failed"]:
        grp = td.get(grp_name)
        if grp:
            first = grp["median_curve"][0]
            if abs(first) > 0.005:
                issues.append(f"CODEIT_NONZERO   {task_id}/{grp_name}: start={first:.4f}")

    # 3. p25 <= median <= p75 at all time steps
    for grp_name in ["human_success", "human_failed", "codeit_success", "codeit_failed"]:
        grp = td.get(grp_name)
        if grp:
            med = np.array(grp["median_curve"])
            p25 = np.array(grp["p25_curve"])
            p75 = np.array(grp["p75_curve"])
            eps = 1e-9
            bad = np.sum((p25 > med + eps) | (med > p75 + eps))
            if bad > 0:
                issues.append(f"QUANTILE_ORDER   {task_id}/{grp_name}: {bad} violations")

with open(VAL_REPORT, "w") as f:
    f.write("Validation report — progress_curves_v2\n")
    f.write("=" * 50 + "\n")
    f.write(f"Tasks checked : {len(all_tasks)}\n")
    f.write(f"Issues found  : {len(issues)}\n\n")
    if issues:
        f.write("ISSUES:\n")
        for issue in issues:
            f.write(f"  {issue}\n")
    else:
        f.write("All checks passed.\n")

status = "✅ All checks passed" if not issues else f"⚠️  {len(issues)} issues"
print(f"Validation: {status}  -> {VAL_REPORT}")

# ── migration summary ─────────────────────────────────────────────────────────

with open(CURVES_V1) as f:
    v1_data = json.load(f)

sample_ids = ["0c9aba6e", "195ba7dc", "009d5c81", "00576224", "3194b014"]
sample_ids = [t for t in sample_ids if t in all_tasks]

lines = [
    "# Migration Summary: Normalization v1 → v2\n",
    "## Formula change\n",
    "| | v1 | v2 |",
    "|---|---|---|",
    "| Baseline | `progress(0)` of each individual trajectory | `progress(input_grid, target_grid)` — shared task property |",
    "| Human zero point | After participant's 1st edit | Same as CodeIt: input_grid level |",
    "| CodeIt zero point | Input grid | Input grid (unchanged) |",
    "| Aggregation | Median of per-trajectory normalised curves | Same |",
    "| IQR bands | Not stored | p25 / p75 stored per group |",
    "",
    "## Starting-point comparison (v1 vs v2 median_curve[0])\n",
    "| task_id | baseline | human_success v1→v2 | human_failed v1→v2 | codeit_success v1→v2 |",
    "|---|---|---|---|---|",
]

for tid in sample_ids:
    td_v2 = all_tasks[tid]
    td_v1 = v1_data[tid]
    b = td_v2["baseline"]

    def fmt(v1_key, v2_key):
        v1c = td_v1.get(v1_key)
        v2c = td_v2.get(v2_key, {}).get("median_curve")
        v1s = f"{v1c[0]:.3f}" if v1c else "—"
        v2s = f"{v2c[0]:.3f}" if v2c else "—"
        return f"{v1s} → {v2s}"

    hs = fmt("human_success_median", "human_success")
    hf = fmt("human_failed_median",  "human_failed")
    cs = fmt("codeit_success_median","codeit_success")
    lines.append(f"| {tid} | {b:.3f} | {hs} | {hf} | {cs} |")

lines += [
    "",
    "## Interpretation of v2 starting points\n",
    "- **CodeIt always starts at 0**: first trace grid is the input_grid, so norm=0 by construction.",
    "- **Human start > 0**: participant's blank output is partially correct (target has background-colored cells).",
    "- **Human start < 0**: participant's 1st edit was worse than the input grid (rare).",
    "- **AUC may exceed v1 values** for human groups on tasks with high blank-output overlap.",
    "",
    "## Downstream impact\n",
    "Scripts 07 and 08 updated to read `progress_curves_v2.json` with new key structure.",
]

with open(MIG_SUMMARY, "w") as f:
    f.write("\n".join(lines))
print(f"Saved migration summary -> {MIG_SUMMARY}")

print(f"\nv2 plots -> {CURVE_V2_DIR}/  ({len(all_tasks)} tasks)")
