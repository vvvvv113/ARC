"""
Extract AUC and steps_to_90pct from normalised mean progress curves for each task.

Curves in progress_curves.json are already normalised (start=0, end=1 for success groups).
Metrics are computed from these normalised curves and saved to curve_metrics.csv.

Two scatter plots compare human_success vs codeit_success across 59 tasks,
coloured by difficulty_category.
"""

import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

REPO       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURVES_PATH = os.path.join(REPO, "analysis/processed/06_curves/progress_curves.json")
DIFF_PATH   = os.path.join(REPO, "analysis/processed/01_difficulty/task_difficulty.csv")
OUT_CSV     = os.path.join(REPO, "analysis/processed/07_metrics/curve_metrics.csv")
OUT_AUC     = os.path.join(REPO, "analysis/processed/07_metrics/auc_scatter.png")
OUT_S90     = os.path.join(REPO, "analysis/processed/07_metrics/steps90_scatter.png")
OUT_DIFF    = os.path.join(REPO, "analysis/processed/07_metrics/auc_by_difficulty.csv")
OUT_CORR    = os.path.join(REPO, "analysis/processed/07_metrics/correlation_stats.csv")
N_POINTS    = 100

# ── helpers ────────────────────────────────────────────────────────────────────

def auc(curve):
    """Area under a 100-point normalised curve via trapezoidal integration."""
    return float(np.trapezoid(curve, dx=1.0 / (N_POINTS - 1)))

def steps_to_90(curve):
    """Normalised x position where curve first reaches 0.9; NaN if never."""
    x = np.linspace(0, 1, N_POINTS)
    hits = [x[i] for i, v in enumerate(curve) if v >= 0.9]
    return hits[0] if hits else float("nan")

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────────────

with open(CURVES_PATH) as f:
    curves = json.load(f)

diff_df = pd.read_csv(DIFF_PATH)[["task_id", "difficulty_category"]]

# ── compute metrics ────────────────────────────────────────────────────────────

rows = []
groups = [
    ("human_success",  "human_success_mean",  "human_success_n"),
    ("human_failed",   "human_failed_mean",   "human_failed_n"),
    ("codeit_success", "codeit_success_mean", "codeit_success_n"),
    ("codeit_failed",  "codeit_failed_mean",  "codeit_failed_n"),
]

for task_id, entry in curves.items():
    row = {"task_id": task_id}
    for name, mean_key, n_key in groups:
        c = entry.get(mean_key)
        n = entry.get(n_key, 0)
        row[f"{name}_n"] = n
        if c is not None and n > 0:
            row[f"{name}_auc"]     = auc(c)
            row[f"{name}_steps90"] = steps_to_90(c)
        else:
            row[f"{name}_auc"]     = float("nan")
            row[f"{name}_steps90"] = float("nan")
    rows.append(row)

metrics = pd.DataFrame(rows)
metrics = metrics.merge(diff_df, on="task_id", how="left")

# ── save CSV ───────────────────────────────────────────────────────────────────

col_order = [
    "task_id", "difficulty_category",
    "human_success_auc",  "codeit_success_auc",
    "human_success_steps90", "codeit_success_steps90",
    "human_failed_auc",   "codeit_failed_auc",
    "human_success_n", "codeit_success_n",
    "human_failed_n",  "codeit_failed_n",
]
metrics[col_order].to_csv(OUT_CSV, index=False)
print(f"Saved metrics -> {OUT_CSV}")

# ── Spearman correlation: human_success_auc ~ codeit_success_auc ───────────────

valid = metrics[["human_success_auc", "codeit_success_auc"]].dropna()
rho, p = stats.spearmanr(valid["human_success_auc"], valid["codeit_success_auc"])
print(f"\nSpearman rho (human_success_auc ~ codeit_success_auc): {rho:+.3f}  p={p:.4f}  n={len(valid)}")

corr_df = pd.DataFrame([{
    "metric_x": "human_success_auc",
    "metric_y": "codeit_success_auc",
    "spearman_rho": round(rho, 4),
    "p_value": round(p, 4),
    "n": len(valid),
}])
corr_df.to_csv(OUT_CORR, index=False)
print(f"Saved correlation stats -> {OUT_CORR}")

# ── summary by difficulty_category ────────────────────────────────────────────

print("\nMean AUC by difficulty category:")
summary = (
    metrics.groupby("difficulty_category")[
        ["human_success_auc", "codeit_success_auc",
         "human_failed_auc",  "codeit_failed_auc",
         "human_success_n",   "codeit_success_n",
         "human_failed_n",    "codeit_failed_n"]
    ]
    .agg({"human_success_auc": "mean", "codeit_success_auc": "mean",
          "human_failed_auc":  "mean", "codeit_failed_auc":  "mean",
          "human_success_n":   "sum",  "codeit_success_n":   "sum",
          "human_failed_n":    "sum",  "codeit_failed_n":    "sum"})
    .round(3)
)
summary.to_csv(OUT_DIFF)
print(summary.to_string())
print(f"Saved difficulty summary -> {OUT_DIFF}")

# ── scatter plots ──────────────────────────────────────────────────────────────

cat_colors = {
    "Easy for both":        "tab:green",
    "Hard for both":        "tab:red",
    "Only hard for AI":     "tab:blue",
    "Only hard for humans": "tab:orange",
}

def scatter_plot(x_col, y_col, xlabel, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_df = metrics[[x_col, y_col, "difficulty_category", "task_id"]].dropna()
    for cat, color in cat_colors.items():
        sub = plot_df[plot_df["difficulty_category"] == cat]
        ax.scatter(sub[x_col], sub[y_col], c=color, label=cat, s=60, alpha=0.8)
    # diagonal reference line
    lo = min(plot_df[x_col].min(), plot_df[y_col].min()) - 0.02
    hi = max(plot_df[x_col].max(), plot_df[y_col].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5, label="y = x")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"Saved plot -> {out_path}")

scatter_plot(
    "human_success_auc", "codeit_success_auc",
    "Human success AUC", "CodeIt success AUC",
    "Convergence speed: Human vs CodeIt (success groups)",
    OUT_AUC,
)
scatter_plot(
    "human_success_steps90", "codeit_success_steps90",
    "Human success steps-to-90%", "CodeIt success steps-to-90%",
    "Steps to 90% progress: Human vs CodeIt (success groups)",
    OUT_S90,
)
