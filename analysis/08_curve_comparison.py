"""
Pairwise curve comparison across the four trajectory groups for each task.

For each task where both curves in a pair are available, computes:

  L2 distance : sqrt( sum_{t=0}^{99} (curve_A(t) - curve_B(t))^2 )
                Absolute path dissimilarity. Larger => curves are farther apart
                in progress space over the entire trajectory.

  Pearson r   : Pearson correlation between the two 100-point curves.
                Shape / trend similarity. r=1 means identical learning rate
                profile; r=0 means unrelated; r<0 means one rises as the
                other falls.

Four pairs compared:
  1. human_success vs codeit_success   — same outcome, different agent
  2. human_failed  vs codeit_failed    — same outcome, different agent
  3. human_success vs human_failed     — different outcome, same agent
  4. codeit_success vs codeit_failed   — different outcome, same agent

Input:
  analysis/processed/06_curves/progress_curves.json
  analysis/processed/01_difficulty/task_difficulty.csv

Outputs (analysis/processed/08_curve_comparison/):
  pair_metrics_per_task.csv  — per-task long-format: task_id, pair,
                               difficulty_category, l2_distance, pearson_r,
                               pearson_p, n_a, n_b
  pair_metrics_summary.csv   — median L2 and Pearson r grouped by
                               pair × difficulty_category
  l2_stripplot.png           — per-task L2, coloured by difficulty_category
  pearson_stripplot.png      — per-task Pearson r, coloured by difficulty_category
"""

import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

REPO        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURVES_PATH = os.path.join(REPO, "analysis/processed/06_curves/progress_curves.json")
DIFF_PATH   = os.path.join(REPO, "analysis/processed/01_difficulty/task_difficulty.csv")
OUT_DIR     = os.path.join(REPO, "analysis/processed/08_curve_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PER_TASK = os.path.join(OUT_DIR, "pair_metrics_per_task.csv")
OUT_SUMMARY  = os.path.join(OUT_DIR, "pair_metrics_summary.csv")
OUT_L2       = os.path.join(OUT_DIR, "l2_stripplot.png")
OUT_PEARSON  = os.path.join(OUT_DIR, "pearson_stripplot.png")

N_POINTS = 100

PAIRS = {
    "human_success_vs_codeit_success": ("human_success_median", "codeit_success_median"),
    "human_failed_vs_codeit_failed":   ("human_failed_median",  "codeit_failed_median"),
    "human_success_vs_human_failed":   ("human_success_median", "human_failed_median"),
    "codeit_success_vs_codeit_failed": ("codeit_success_median","codeit_failed_median"),
}

PAIR_LABELS = {
    "human_success_vs_codeit_success": "Human succ\nvs CodeIt succ",
    "human_failed_vs_codeit_failed":   "Human fail\nvs CodeIt fail",
    "human_success_vs_human_failed":   "Human succ\nvs Human fail",
    "codeit_success_vs_codeit_failed": "CodeIt succ\nvs CodeIt fail",
}

CAT_COLORS = {
    "Easy for both":        "tab:green",
    "Hard for both":        "tab:red",
    "Only hard for AI":     "tab:blue",
    "Only hard for humans": "tab:orange",
}

# ── load data ──────────────────────────────────────────────────────────────────

with open(CURVES_PATH) as f:
    curves = json.load(f)

diff_df  = pd.read_csv(DIFF_PATH)[["task_id", "difficulty_category"]]
diff_map = dict(zip(diff_df["task_id"], diff_df["difficulty_category"]))

# ── per-task metrics ───────────────────────────────────────────────────────────

rows = []
for task_id, entry in curves.items():
    diff_cat = diff_map.get(task_id, "Unknown")
    for pair_name, (key_a, key_b) in PAIRS.items():
        ca = entry.get(key_a)
        cb = entry.get(key_b)
        n_a = entry.get(key_a.replace("_median", "_n"), 0)
        n_b = entry.get(key_b.replace("_median", "_n"), 0)
        if ca is None or cb is None or n_a == 0 or n_b == 0:
            continue
        ca_arr = np.array(ca)
        cb_arr = np.array(cb)
        l2  = float(np.sqrt(np.sum((ca_arr - cb_arr) ** 2)))
        # Pearson r is undefined when either curve is constant (std = 0)
        if ca_arr.std() == 0 or cb_arr.std() == 0:
            r, p = float("nan"), float("nan")
        else:
            r, p = stats.pearsonr(ca_arr, cb_arr)
        rows.append({
            "task_id":             task_id,
            "difficulty_category": diff_cat,
            "pair":                pair_name,
            "l2_distance":         round(l2, 4),
            "pearson_r":           round(float(r), 4),
            "pearson_p":           round(float(p), 4),
            "n_a":                 n_a,
            "n_b":                 n_b,
        })

per_task_df = pd.DataFrame(rows)
per_task_df.to_csv(OUT_PER_TASK, index=False)
print(f"Saved per-task metrics -> {OUT_PER_TASK}  ({len(per_task_df)} rows)")

# ── summary: median by pair × difficulty_category ─────────────────────────────

summary = (
    per_task_df.groupby(["pair", "difficulty_category"])
    .agg(
        task_count    = ("task_id",      "count"),
        median_l2     = ("l2_distance",  "median"),
        median_pearson_r = ("pearson_r", "median"),
    )
    .round(4)
    .reset_index()
)
summary.to_csv(OUT_SUMMARY, index=False)
print(f"\nSaved summary -> {OUT_SUMMARY}")
print(summary.to_string(index=False))

# ── strip plots ────────────────────────────────────────────────────────────────

def strip_plot(metric, ylabel, title, out_path):
    pair_names = list(PAIRS.keys())
    fig, ax = plt.subplots(figsize=(11, 5))
    rng = np.random.default_rng(42)

    legend_added = set()
    for xi, pair in enumerate(pair_names):
        sub = per_task_df[per_task_df["pair"] == pair]
        for cat, color in CAT_COLORS.items():
            pts = sub[sub["difficulty_category"] == cat][metric].values
            if len(pts) == 0:
                continue
            jitter = rng.uniform(-0.18, 0.18, len(pts))
            label  = cat if cat not in legend_added else "_nolegend_"
            ax.scatter(xi + jitter, pts, c=color, s=35, alpha=0.75, label=label)
            legend_added.add(cat)
        # median line across all categories
        med = sub[metric].median()
        ax.plot([xi - 0.28, xi + 0.28], [med, med],
                color="black", linewidth=2.0, zorder=5)

    ax.set_xticks(range(len(pair_names)))
    ax.set_xticklabels([PAIR_LABELS[p] for p in pair_names], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"Saved plot -> {out_path}")

strip_plot(
    "l2_distance",
    "L2 distance",
    "L2 distance between paired median curves (each point = one task; bar = median)",
    OUT_L2,
)
strip_plot(
    "pearson_r",
    "Pearson r",
    "Pearson r between paired median curves (each point = one task; bar = median)",
    OUT_PEARSON,
)
