"""
Aggregate plots for Week 3 method 2 vs baseline comparison.

Produces (under CCS Project/baseline_results/):
  - method2_dtw_per_seed.png        bar chart of mean DTW similarity per seed
  - method2_wasserstein_per_seed.png bar chart of Wasserstein distance per seed
  - method2_per_task_improved.png    fraction of tasks improved per seed

Reads:
  CCS Project/baseline_results/method2_dtw_comparison_summary.csv

Run from repo root:
    python "CCS Project/make_method2_summary_plots.py"
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RESULTS = HERE / "baseline_results"
df = pd.read_csv(RESULTS / "method2_dtw_comparison_summary.csv").sort_values("seed").reset_index(drop=True)

seeds = df["seed"].astype(str).tolist()
x = np.arange(len(seeds))
W = 0.36

def grouped_bar(ax, base_vals, m2_vals, ylabel, title, fmt="{:.4f}"):
    bb = ax.bar(x - W/2, base_vals, W, label="Baseline", color="#1f77b4")
    bm = ax.bar(x + W/2, m2_vals, W, label="Method 2 (λ=0.5)", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bars in (bb, bm):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    fmt.format(b.get_height()), ha="center", va="bottom", fontsize=8)

# 1. Mean DTW similarity (higher = more human-like)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
grouped_bar(ax, df["baseline_mean_dtw"].values, df["method2_mean_dtw"].values,
            "Mean DTW similarity to humans (higher = more human-like)",
            "Mean DTW similarity per seed (epoch 93)")
# annotate cross-seed means
b_mean = df["baseline_mean_dtw"].mean(); m_mean = df["method2_mean_dtw"].mean()
ax.axhline(b_mean, color="#1f77b4", ls="--", alpha=0.6, lw=1)
ax.axhline(m_mean, color="#d62728", ls="--", alpha=0.6, lw=1)
ax.text(len(seeds)-0.5, b_mean, f"  baseline mean {b_mean:.4f}", color="#1f77b4", va="bottom", fontsize=8)
ax.text(len(seeds)-0.5, m_mean, f"  method2 mean {m_mean:.4f}",  color="#d62728", va="top",    fontsize=8)
fig.tight_layout(); fig.savefig(RESULTS / "method2_dtw_per_seed.png", dpi=150); plt.close(fig)

# 2. Wasserstein distance to human curve-AUC distribution (lower = more human-like)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
grouped_bar(ax, df["baseline_wasserstein"].values, df["method2_wasserstein"].values,
            "Wasserstein distance to human AUC distribution (lower = more human-like)",
            "Wasserstein distance per seed (epoch 93)")
fig.tight_layout(); fig.savefig(RESULTS / "method2_wasserstein_per_seed.png", dpi=150); plt.close(fig)

# 3. Per-task improvement: tasks where method 2 has higher mean DTW than baseline
fig, ax = plt.subplots(figsize=(7.5, 4.5))
frac = df["tasks_improved"] / df["tasks_total_pertask"]
bars = ax.bar(x, frac.values, 0.55, color="#9467bd")
ax.axhline(0.5, color="gray", ls="--", lw=1, label="50% (no effect)")
ax.set_xticks(x); ax.set_xticklabels([f"seed {s}" for s in seeds])
ax.set_ylabel("Fraction of tasks where method 2 > baseline (DTW sim)")
ax.set_title("Per-task improvement rate under method 2")
ax.set_ylim(0, 1.0)
for i, b in enumerate(bars):
    ax.text(b.get_x() + b.get_width()/2, b.get_height(),
            f"{df['tasks_improved'].iat[i]}/{df['tasks_total_pertask'].iat[i]}\n({b.get_height()*100:.0f}%)",
            ha="center", va="bottom", fontsize=9)
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(RESULTS / "method2_per_task_improved.png", dpi=150); plt.close(fig)

print("Saved:")
for p in ["method2_dtw_per_seed.png", "method2_wasserstein_per_seed.png", "method2_per_task_improved.png"]:
    print(f"  {RESULTS/p}")
