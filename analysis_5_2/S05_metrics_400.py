"""
S05_metrics_400.py
AUC、steps_to_90pct 指标 + Spearman 相关检验

指标定义：
  AUC         = np.trapz(median_curve, dx=1/99)，∈(−∞,1]；负值有效（整体低于基线）
  steps_to_90pct = median curve 第一次 ≥ 0.90 的归一化时间点（0–1）；
                   从未达到则 NaN（包括 human_failed 和大多数 codeit_failed 任务）

分层报告：
  所有统计量按 baseline_group A（baseline=0）和 B（baseline>0）分层报告。
  合并报告会混淆两类认知结构不同的任务，存在 Simpson's paradox 风险。

Spearman 检验（FDR family 定义）：
  问题：human_success AUC 与 codeit_success AUC 是否在任务层面正相关？
  检验：对有两组数据的 task，计算 Spearman ρ(human_success AUC, codeit_success AUC)
  Family：Group A 和 Group B 各一个检验，共 2 个 p 值，作为同一 family 进行 BH 校正。
  理由：两个检验回答同一科学问题（AUC 相关性），应视为同一 family；
        S06/S07 的检验回答不同科学问题，不混入此 family。
  效应量：报告 ρ 本身 + bootstrap 95% CI（B=1000），不仅报告 p 值。

协变量分析：
  散点图：x = n_wrong_cells，y = human_success median AUC 的 bootstrap CI 宽度（B=200）
  预期：n_wrong_cells 越小（高 baseline 任务），progress 粒度越粗，CI 越宽。

输出：
  S05_metrics/curve_metrics.csv           — 每 task × group 的 AUC + steps_to_90pct
  S05_metrics/spearman_summary.csv        — Spearman ρ、CI、raw p、BH-adjusted p
  S05_metrics/auc_distribution.png        — AUC 分布箱线图（按 group × baseline_group）
  S05_metrics/spearman_scatter_A.png      — Group A: human_success vs codeit_success AUC
  S05_metrics/spearman_scatter_B.png      — Group B: 同上
  S05_metrics/auc_ci_width_scatter.png    — n_wrong_cells vs bootstrap CI 宽度（协变量图）
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# BH 校正（不依赖 statsmodels）
def bh_correction(pvals):
    """
    Benjamini-Hochberg FDR 校正。
    输入：p 值列表（同一 family）
    返回：调整后的 p 值列表（顺序与输入一致）
    """
    n  = len(pvals)
    idx_sorted = np.argsort(pvals)
    sorted_p   = np.array(pvals)[idx_sorted]
    adj = np.minimum.accumulate((sorted_p * n / (np.arange(n) + 1))[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    result = np.empty(n)
    result[idx_sorted] = adj
    return result.tolist()

# ── 路径配置 ───────────────────────────────────────────────────────────────────

REPO        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURVES_JSON = os.path.join(REPO, "analysis_5_2/processed/S04_curves/progress_curves_400.json")
SUMMARY_CSV = os.path.join(REPO, "analysis_5_2/processed/S04_curves/curve_summary_400.csv")
OUT_DIR     = os.path.join(REPO, "analysis_5_2/processed/S05_metrics")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 加载数据 ───────────────────────────────────────────────────────────────────

print("Loading S04 outputs...")
with open(CURVES_JSON) as f:
    curves = json.load(f)
summary_df = pd.read_csv(SUMMARY_CSV)
print(f"  Tasks in JSON: {len(curves)}")
print(f"  Rows in summary CSV: {len(summary_df)}")

# ── 计算 steps_to_90pct ───────────────────────────────────────────────────────

print("Computing steps_to_90pct...")
x100 = np.linspace(0, 1, 100)

def steps_to_threshold(median_list, threshold=0.90):
    """
    返回 median curve 第一次 >= threshold 的归一化时间点（x100 中的值）。
    从未达到返回 NaN。
    """
    arr = np.array(median_list)
    idx = np.where(arr >= threshold)[0]
    if len(idx) == 0:
        return float("nan")
    return float(x100[idx[0]])

# 构建 metrics 表：per task × group
metrics_rows = []
GROUP_ORDER = ["human_success", "human_failed", "codeit_success", "codeit_failed"]

for task_id, task_data in curves.items():
    meta = {
        "baseline":           task_data["baseline"],
        "baseline_group":     task_data["baseline_group"],
        "n_wrong_cells":      task_data["n_wrong_cells"],
        "numerically_coarse": task_data["numerically_coarse"],
    }
    for gname in GROUP_ORDER:
        if gname not in task_data:
            continue
        ge  = task_data[gname]
        s90 = steps_to_threshold(ge["median"])
        metrics_rows.append({
            "task_id":             task_id,
            "group":               gname,
            "n_traces":            ge["n"],
            "auc_median":          ge["auc_median"],
            "steps_to_90pct":      s90,
            **meta,
        })

metrics_df = pd.DataFrame(metrics_rows)
out_metrics = os.path.join(OUT_DIR, "curve_metrics.csv")
metrics_df.to_csv(out_metrics, index=False)
print(f"Saved -> {out_metrics}  ({len(metrics_df)} rows)")

# ── Spearman ρ：human_success AUC vs codeit_success AUC ───────────────────────

print("\nComputing Spearman correlations (human_success AUC vs codeit_success AUC)...")

# 取有两组数据的任务：merge human_success 和 codeit_success AUC
hs = metrics_df[metrics_df["group"] == "human_success"][["task_id", "auc_median", "baseline_group"]].rename(columns={"auc_median": "hs_auc"})
cs = metrics_df[metrics_df["group"] == "codeit_success"][["task_id", "auc_median"]].rename(columns={"auc_median": "cs_auc"})
paired = hs.merge(cs, on="task_id")

rng = np.random.default_rng(42)
B_rho = 1000

spearman_rows = []
raw_pvals     = []
group_labels  = []

for bg in ["A", "B"]:
    sub = paired[paired["baseline_group"] == bg]
    n   = len(sub)
    if n < 5:
        print(f"  Group {bg}: n={n} too small, skip")
        continue

    hs_arr = sub["hs_auc"].values
    cs_arr = sub["cs_auc"].values

    rho_obs, p_obs = stats.spearmanr(hs_arr, cs_arr)

    # Bootstrap CI for ρ
    rho_boot = []
    for _ in range(B_rho):
        idx     = rng.integers(0, n, size=n)
        rho_b, _ = stats.spearmanr(hs_arr[idx], cs_arr[idx])
        rho_boot.append(rho_b)
    ci_lo = float(np.percentile(rho_boot, 2.5))
    ci_hi = float(np.percentile(rho_boot, 97.5))

    spearman_rows.append({
        "baseline_group": bg,
        "n_tasks":        n,
        "rho":            round(float(rho_obs), 4),
        "ci_lo_95":       round(ci_lo, 4),
        "ci_hi_95":       round(ci_hi, 4),
        "p_raw":          round(float(p_obs), 6),
        "p_adj_BH":       None,  # 填入 BH 校正后
    })
    raw_pvals.append(float(p_obs))
    group_labels.append(bg)
    print(f"  Group {bg}: n={n}, ρ={rho_obs:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], p={p_obs:.4f}")

# BH 校正（family = 这 2 个检验）
if raw_pvals:
    adj_pvals = bh_correction(raw_pvals)
    for i, row in enumerate(spearman_rows):
        row["p_adj_BH"] = round(adj_pvals[i], 6)
    print(f"\n  BH-adjusted p-values: {[round(p, 4) for p in adj_pvals]}")

spearman_df = pd.DataFrame(spearman_rows)
out_spearman = os.path.join(OUT_DIR, "spearman_summary.csv")
spearman_df.to_csv(out_spearman, index=False)
print(f"Saved -> {out_spearman}")

# ── Bootstrap CI 宽度（协变量分析）─────────────────────────────────────────────

print("\nComputing bootstrap CI widths for human_success AUC (B=200)...")
B_ci = 200

ci_width_rows = []
for task_id, task_data in curves.items():
    if "human_success" not in task_data:
        continue
    ge = task_data["human_success"]
    n  = ge["n"]
    if n < 3:
        continue
    traces = np.array(ge["traces"], dtype=float)  # shape (n, 100)
    auc_boot = []
    for _ in range(B_ci):
        idx     = rng.integers(0, n, size=n)
        med     = np.median(traces[idx], axis=0)
        auc_b   = float(np.trapz(med, dx=1 / 99))
        auc_boot.append(auc_b)
    ci_width = float(np.percentile(auc_boot, 97.5) - np.percentile(auc_boot, 2.5))
    ci_width_rows.append({
        "task_id":        task_id,
        "n_traces":       n,
        "n_wrong_cells":  task_data["n_wrong_cells"],
        "baseline_group": task_data["baseline_group"],
        "auc_median":     ge["auc_median"],
        "ci_width_95":    round(ci_width, 4),
    })

ci_df = pd.DataFrame(ci_width_rows)
print(f"  Tasks with CI computed: {len(ci_df)}")

# ── 绘图 1：AUC 分布箱线图 ─────────────────────────────────────────────────────

print("\nGenerating plots...")

GROUP_COLORS = {
    "human_success":  "tab:green",
    "human_failed":   "tab:red",
    "codeit_success": "tab:blue",
    "codeit_failed":  "dimgray",
}
GROUP_LABELS = {
    "human_success":  "Human\nsuccess",
    "human_failed":   "Human\nfailed",
    "codeit_success": "CodeIt\nsuccess",
    "codeit_failed":  "CodeIt\nfailed",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("S05: AUC Distribution by Group and Baseline Group", fontsize=11, fontweight="bold")

for ax_idx, bg in enumerate(["A", "B"]):
    ax  = axes[ax_idx]
    sub = metrics_df[metrics_df["baseline_group"] == bg]
    plot_data, plot_labels, plot_colors = [], [], []
    for gname in ["human_success", "human_failed", "codeit_success", "codeit_failed"]:
        gsub = sub[sub["group"] == gname]["auc_median"].dropna()
        if len(gsub) < 2:
            continue
        plot_data.append(gsub.values)
        plot_labels.append(GROUP_LABELS[gname] + f"\n(n={len(gsub)})")
        plot_colors.append(GROUP_COLORS[gname])

    if not plot_data:
        continue

    bp = ax.boxplot(plot_data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], plot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticklabels(plot_labels, fontsize=9)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axhline(1, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Median AUC per task")
    bg_desc = "baseline=0, abstract" if bg == "A" else "baseline>0, perceptual"
    ax.set_title(f"Baseline Group {bg}  ({bg_desc})")
    ax.grid(alpha=0.25, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "auc_distribution.png"), dpi=130)
plt.close()
print("  Saved: auc_distribution.png")

# ── 绘图 2：Spearman 散点图（分 Group A / B）──────────────────────────────────

for bg in ["A", "B"]:
    sub = paired[paired["baseline_group"] == bg]
    if len(sub) < 3:
        continue

    row = spearman_df[spearman_df["baseline_group"] == bg]
    if row.empty:
        continue
    row = row.iloc[0]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(sub["hs_auc"], sub["cs_auc"], alpha=0.65, edgecolors="white",
               s=55, color="steelblue")
    lim = [min(sub["hs_auc"].min(), sub["cs_auc"].min()) - 0.08,
           max(sub["hs_auc"].max(), sub["cs_auc"].max()) + 0.08]
    ax.plot(lim, lim, color="gray", linewidth=1, linestyle="--", label="y=x")
    ax.set_xlabel("Human success AUC (per task)")
    ax.set_ylabel("CodeIt success AUC (per task)")
    ax.set_title(
        f"Group {bg}: Human success vs CodeIt success AUC\n"
        f"ρ={row['rho']:.3f}  95% CI [{row['ci_lo_95']:.3f}, {row['ci_hi_95']:.3f}]  "
        f"p_raw={row['p_raw']:.4f}  p_BH={row['p_adj_BH']:.4f}  (n={int(row['n_tasks'])})",
        fontsize=8.5,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fname = f"spearman_scatter_{bg}.png"
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=130)
    plt.close()
    print(f"  Saved: {fname}")

# ── 绘图 3：n_wrong_cells vs bootstrap CI 宽度（协变量）─────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "S05: Human success AUC uncertainty vs n_wrong_cells\n"
    "(CI width = bootstrap 95% CI, B=200; tasks with n_traces ≥ 3)",
    fontsize=10, fontweight="bold"
)

for ax_idx, bg in enumerate(["A", "B"]):
    ax  = axes[ax_idx]
    sub = ci_df[ci_df["baseline_group"] == bg]
    if sub.empty:
        ax.set_visible(False)
        continue

    ax.scatter(sub["n_wrong_cells"], sub["ci_width_95"],
               alpha=0.55, edgecolors="white", s=45,
               c=sub["n_traces"], cmap="viridis")
    ax.set_xlabel("n_wrong_cells (= total_cells × (1 − baseline))")
    ax.set_ylabel("Bootstrap 95% CI width of median AUC")
    bg_desc = "baseline=0, abstract" if bg == "A" else "baseline>0, perceptual"
    ax.set_title(f"Group {bg}  ({bg_desc})\n(color = n_traces per task)")
    ax.grid(alpha=0.25)

    # Spearman ρ between n_wrong_cells and CI width
    if len(sub) >= 5:
        rho_cov, p_cov = stats.spearmanr(sub["n_wrong_cells"], sub["ci_width_95"])
        ax.annotate(
            f"ρ={rho_cov:.2f}, p={p_cov:.3f}",
            xy=(0.05, 0.92), xycoords="axes fraction",
            fontsize=9, color="firebrick",
        )

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "auc_ci_width_scatter.png"), dpi=130)
plt.close()
print("  Saved: auc_ci_width_scatter.png")

# ── 打印汇总统计 ───────────────────────────────────────────────────────────────

print("\n=== AUC Summary by group × baseline_group ===")
for bg in ["A", "B"]:
    sub = metrics_df[metrics_df["baseline_group"] == bg]
    print(f"\n  Baseline Group {bg}:")
    for gname in ["human_success", "human_failed", "codeit_success", "codeit_failed"]:
        gsub = sub[sub["group"] == gname]
        if len(gsub) == 0:
            continue
        aucs  = gsub["auc_median"]
        s90s  = gsub["steps_to_90pct"]
        pct90 = (~s90s.isna()).mean() * 100
        print(f"    {gname:20s}  n={len(gsub):3d}  "
              f"AUC median={aucs.median():.3f}  IQR=[{aucs.quantile(.25):.3f},{aucs.quantile(.75):.3f}]  "
              f"steps_to_90pct: {pct90:.0f}% reached,  "
              f"median={s90s.median():.3f}" if not s90s.isna().all() else
              f"    {gname:20s}  n={len(gsub):3d}  "
              f"AUC median={aucs.median():.3f}  "
              f"steps_to_90pct: 0% reached"
              )

print("\n=== Spearman Results ===")
print(spearman_df.to_string(index=False))

print("\n✓ S05 complete.")
print(f"  Metrics:  {out_metrics}")
print(f"  Spearman: {out_spearman}")
print(f"  Plots:    {OUT_DIR}/")
