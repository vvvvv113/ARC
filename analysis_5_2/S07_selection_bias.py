"""
S07_selection_bias.py
Selection Bias 检验：有 CodeIt 数据的任务 vs 无 CodeIt 数据的任务

背景：
  CodeIt 能产生 traces 的 134 个任务是有选择性的子集——
  这些是 CodeIt 至少尝试过并被存入 replay buffer 的任务。
  如果这些任务系统性地对人类也更容易（更高的 human AUC 或更高的解题率），
  则 human vs CodeIt 比较分析的结论不能推广到全部 400 个任务。
  266 个没有 CodeIt 数据的任务是自然的对照组。

分析内容：
  1. 两组任务的 human_success AUC 分布对比
     → Mann-Whitney U 检验 + 效应量 rank-biserial r
  2. 两组任务的 human success rate（人类解题率）对比
     → 同上
  3. 两组任务的 baseline 分布对比（任务结构差异）
  4. 两组任务的 n_wrong_cells 分布对比（任务难度代理指标）
  5. 全部比较均按 baseline_group A / B 分层报告

效应量：rank-biserial r = 2·U/(n1·n2) − 1
  r ≈ 0 → 无差异；r > 0 → 有 CodeIt 的任务更高；r < 0 → 无 CodeIt 的任务更高
  |r| < 0.1 small, 0.1–0.3 medium, > 0.3 large

输出：
  S07_selection_bias/selection_bias_summary.csv   — 各指标的检验结果汇总
  S07_selection_bias/human_auc_comparison.png      — AUC 箱线图（有/无 CodeIt × baseline_group）
  S07_selection_bias/human_success_rate_comparison.png
  S07_selection_bias/baseline_comparison.png
  S07_selection_bias/n_wrong_cells_comparison.png
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ── 路径配置 ───────────────────────────────────────────────────────────────────

REPO          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURVES_JSON   = os.path.join(REPO, "analysis_5_2/processed/S04_curves/progress_curves_400.json")
METRICS_CSV   = os.path.join(REPO, "analysis_5_2/processed/S05_metrics/curve_metrics.csv")
HUMAN_SUM_CSV = os.path.join(REPO, "analysis_5_2/processed/S02_human_traces/human_traces_summary.csv")
OUT_DIR       = os.path.join(REPO, "analysis_5_2/processed/S07_selection_bias")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 加载数据 ───────────────────────────────────────────────────────────────────

print("Loading data...")

with open(CURVES_JSON) as f:
    curves = json.load(f)

metrics_df  = pd.read_csv(METRICS_CSV)
human_sum   = pd.read_csv(HUMAN_SUM_CSV)   # task_id, n_success, n_failed, n_total, avg_grids_per_trace

# 从 curves JSON 提取每任务的 baseline 元数据
meta_rows = []
for task_id, td in curves.items():
    meta_rows.append({
        "task_id":           task_id,
        "baseline":          td["baseline"],
        "baseline_group":    td["baseline_group"],
        "n_wrong_cells":     td["n_wrong_cells"],
        "numerically_coarse": td["numerically_coarse"],
        "has_codeit_success": "codeit_success" in td,
        "has_codeit_failed":  "codeit_failed"  in td,
        "has_any_codeit":     ("codeit_success" in td) or ("codeit_failed" in td),
    })
meta_df = pd.DataFrame(meta_rows)
print(f"  Total tasks: {len(meta_df)}")
print(f"  Has any CodeIt: {meta_df['has_any_codeit'].sum()}")
print(f"  No CodeIt:      {(~meta_df['has_any_codeit']).sum()}")

# 合并 human_success AUC（来自 S05）
hs_auc = (
    metrics_df[metrics_df["group"] == "human_success"]
    [["task_id", "auc_median", "n_traces"]]
    .rename(columns={"auc_median": "hs_auc", "n_traces": "n_human_success"})
)
hf_auc = (
    metrics_df[metrics_df["group"] == "human_failed"]
    [["task_id", "auc_median"]]
    .rename(columns={"auc_median": "hf_auc"})
)

# 合并 human success rate（来自 S02）
human_sum["success_rate"] = human_sum["n_success"] / human_sum["n_total"]

df = (
    meta_df
    .merge(hs_auc,    on="task_id", how="left")
    .merge(hf_auc,    on="task_id", how="left")
    .merge(human_sum[["task_id", "n_success", "n_failed", "n_total", "success_rate"]],
           on="task_id", how="left")
)

print(f"\nMerged dataset: {len(df)} tasks")
print(df[["has_any_codeit","baseline_group"]].value_counts().sort_index())

# ── Mann-Whitney U 检验 + 效应量 ───────────────────────────────────────────────

def mann_whitney_test(group_with, group_without, label):
    """
    比较 has_codeit vs no_codeit 两组的某指标分布。
    返回结果字典，包含 U、p、rank-biserial r、两组中位数。
    """
    a = group_with.dropna().values
    b = group_without.dropna().values
    if len(a) < 3 or len(b) < 3:
        return None
    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b)
    # rank-biserial r：正值 = has_codeit 组更高
    r = 2 * U / (n1 * n2) - 1
    return {
        "metric":        label,
        "n_with_codeit": n1,
        "n_no_codeit":   n2,
        "median_with":   round(float(np.median(a)), 4),
        "median_without": round(float(np.median(b)), 4),
        "U":             round(float(U), 1),
        "p":             round(float(p), 6),
        "rank_biserial_r": round(float(r), 4),
        "effect_size":   "small" if abs(r) < 0.1 else ("medium" if abs(r) < 0.3 else "large"),
    }

# ── 全量检验（不分 baseline_group）─────────────────────────────────────────────

print("\n── Overall (all tasks) ──")
summary_rows = []

for metric, col in [
    ("human_success AUC",   "hs_auc"),
    ("human failed AUC",    "hf_auc"),
    ("human success rate",  "success_rate"),
    ("baseline",            "baseline"),
    ("n_wrong_cells",       "n_wrong_cells"),
]:
    w = df[df["has_any_codeit"]][col]
    n = df[~df["has_any_codeit"]][col]
    res = mann_whitney_test(w, n, metric)
    if res:
        res["baseline_group"] = "all"
        summary_rows.append(res)
        print(f"  {metric:30s}  with={res['median_with']:.3f}  without={res['median_without']:.3f}  "
              f"r={res['rank_biserial_r']:.3f}  p={res['p']:.4f}  [{res['effect_size']}]")

# ── 分层检验（按 baseline_group）──────────────────────────────────────────────

for bg in ["A", "B"]:
    print(f"\n── Baseline Group {bg} ──")
    sub = df[df["baseline_group"] == bg]
    for metric, col in [
        ("human_success AUC",   "hs_auc"),
        ("human success rate",  "success_rate"),
        ("n_wrong_cells",       "n_wrong_cells"),
    ]:
        w = sub[sub["has_any_codeit"]][col]
        n = sub[~sub["has_any_codeit"]][col]
        res = mann_whitney_test(w, n, metric)
        if res:
            res["baseline_group"] = bg
            summary_rows.append(res)
            print(f"  {metric:30s}  with={res['median_with']:.3f}  without={res['median_without']:.3f}  "
                  f"r={res['rank_biserial_r']:.3f}  p={res['p']:.4f}  [{res['effect_size']}]")

summary_out = pd.DataFrame(summary_rows)
summary_out.to_csv(os.path.join(OUT_DIR, "selection_bias_summary.csv"), index=False)
print(f"\nSaved -> selection_bias_summary.csv")

# ── 绘图辅助 ───────────────────────────────────────────────────────────────────

def make_comparison_boxplot(df_all, col, ylabel, title, fname,
                             groups=("A", "B"), ylim=None):
    """
    双面板（Group A / Group B）的箱线图：有 CodeIt vs 无 CodeIt 的某指标分布。
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=11, fontweight="bold")
    for ax_idx, bg in enumerate(groups):
        ax  = axes[ax_idx]
        sub = df_all[df_all["baseline_group"] == bg]
        w   = sub[sub["has_any_codeit"]][col].dropna().values
        n   = sub[~sub["has_any_codeit"]][col].dropna().values
        bp  = ax.boxplot([w, n], patch_artist=True, widths=0.5,
                         medianprops=dict(color="black", linewidth=2))
        bp["boxes"][0].set_facecolor("steelblue"); bp["boxes"][0].set_alpha(0.65)
        bp["boxes"][1].set_facecolor("tomato");    bp["boxes"][1].set_alpha(0.65)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"Has CodeIt\n(n={len(w)})", f"No CodeIt\n(n={len(n)})"], fontsize=9)
        ax.set_ylabel(ylabel)
        bg_desc = "baseline=0, abstract" if bg == "A" else "baseline>0, perceptual"
        ax.set_title(f"Group {bg}  ({bg_desc})")
        ax.grid(alpha=0.25, axis="y")
        if ylim:
            ax.set_ylim(*ylim)
        # 在图上标注 p 值
        row = summary_out[(summary_out["metric"] == col.replace("_", " ").replace("hs auc","human_success AUC")) &
                          (summary_out["baseline_group"] == bg)]
        # 重新从 summary_rows 找
        matched = [r for r in summary_rows
                   if r["baseline_group"] == bg and col in r["metric"].lower().replace(" ","_")]
        if matched:
            r_val = matched[0]["rank_biserial_r"]
            p_val = matched[0]["p"]
            ax.annotate(f"r={r_val:.3f}\np={p_val:.4f}", xy=(0.72, 0.88),
                        xycoords="axes fraction", fontsize=9, color="navy")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=130)
    plt.close()

# ── 图1：Human Success AUC 对比 ──────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("S07: Human Success AUC — Tasks with vs without CodeIt Data",
             fontsize=11, fontweight="bold")
for ax_idx, bg in enumerate(["A", "B"]):
    ax  = axes[ax_idx]
    sub = df[df["baseline_group"] == bg]
    w   = sub[sub["has_any_codeit"]]["hs_auc"].dropna().values
    n   = sub[~sub["has_any_codeit"]]["hs_auc"].dropna().values
    bp  = ax.boxplot([w, n], patch_artist=True, widths=0.5,
                     medianprops=dict(color="black", linewidth=2))
    bp["boxes"][0].set_facecolor("steelblue"); bp["boxes"][0].set_alpha(0.65)
    bp["boxes"][1].set_facecolor("tomato");    bp["boxes"][1].set_alpha(0.65)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Has CodeIt\n(n={len(w)})", f"No CodeIt\n(n={len(n)})"], fontsize=9)
    ax.set_ylabel("Human success AUC (per task)")
    bg_desc = "baseline=0, abstract" if bg == "A" else "baseline>0, perceptual"
    ax.set_title(f"Group {bg}  ({bg_desc})")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axhline(1, color="gray", linewidth=0.8, linestyle="--")
    ax.grid(alpha=0.25, axis="y")
    matched = [r for r in summary_rows
               if r["baseline_group"] == bg and "human_success AUC" in r["metric"]]
    if matched:
        r_val = matched[0]["rank_biserial_r"]
        p_val = matched[0]["p"]
        ax.annotate(f"rank-biserial r = {r_val:.3f}\np = {p_val:.4f}",
                    xy=(0.60, 0.88), xycoords="axes fraction", fontsize=9, color="navy")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "human_auc_comparison.png"), dpi=130)
plt.close()
print("Saved -> human_auc_comparison.png")

# ── 图2：Human Success Rate 对比 ──────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("S07: Human Success Rate — Tasks with vs without CodeIt Data",
             fontsize=11, fontweight="bold")
for ax_idx, bg in enumerate(["A", "B"]):
    ax  = axes[ax_idx]
    sub = df[df["baseline_group"] == bg]
    w   = sub[sub["has_any_codeit"]]["success_rate"].dropna().values
    n   = sub[~sub["has_any_codeit"]]["success_rate"].dropna().values
    bp  = ax.boxplot([w, n], patch_artist=True, widths=0.5,
                     medianprops=dict(color="black", linewidth=2))
    bp["boxes"][0].set_facecolor("steelblue"); bp["boxes"][0].set_alpha(0.65)
    bp["boxes"][1].set_facecolor("tomato");    bp["boxes"][1].set_alpha(0.65)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Has CodeIt\n(n={len(w)})", f"No CodeIt\n(n={len(n)})"], fontsize=9)
    ax.set_ylabel("Human success rate (n_success / n_total)")
    bg_desc = "baseline=0, abstract" if bg == "A" else "baseline>0, perceptual"
    ax.set_title(f"Group {bg}  ({bg_desc})")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25, axis="y")
    matched = [r for r in summary_rows
               if r["baseline_group"] == bg and "human success rate" in r["metric"]]
    if matched:
        r_val = matched[0]["rank_biserial_r"]
        p_val = matched[0]["p"]
        ax.annotate(f"rank-biserial r = {r_val:.3f}\np = {p_val:.4f}",
                    xy=(0.60, 0.88), xycoords="axes fraction", fontsize=9, color="navy")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "human_success_rate_comparison.png"), dpi=130)
plt.close()
print("Saved -> human_success_rate_comparison.png")

# ── 图3：Baseline 分布对比 ─────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4))
w_base = df[df["has_any_codeit"]]["baseline"].dropna().values
n_base = df[~df["has_any_codeit"]]["baseline"].dropna().values
bins = np.linspace(0, 1, 30)
ax.hist(w_base, bins=bins, alpha=0.65, color="steelblue",
        label=f"Has CodeIt (n={len(w_base)})", edgecolor="white")
ax.hist(n_base, bins=bins, alpha=0.65, color="tomato",
        label=f"No CodeIt (n={len(n_base)})", edgecolor="white")
ax.set_xlabel("Baseline (= progress(input, target))")
ax.set_ylabel("Number of tasks")
ax.set_title("S07: Baseline Distribution — Has CodeIt vs No CodeIt")
ax.legend(fontsize=9)
ax.grid(alpha=0.25, axis="y")
overall_base = [r for r in summary_rows if r["baseline_group"] == "all" and "baseline" in r["metric"]]
if overall_base:
    ax.annotate(f"rank-biserial r={overall_base[0]['rank_biserial_r']:.3f}, p={overall_base[0]['p']:.4f}",
                xy=(0.55, 0.88), xycoords="axes fraction", fontsize=9, color="navy")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "baseline_comparison.png"), dpi=130)
plt.close()
print("Saved -> baseline_comparison.png")

# ── 图4：n_wrong_cells 分布对比 ───────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("S07: n_wrong_cells Distribution — Has CodeIt vs No CodeIt",
             fontsize=11, fontweight="bold")
for ax_idx, bg in enumerate(["A", "B"]):
    ax  = axes[ax_idx]
    sub = df[df["baseline_group"] == bg]
    w   = sub[sub["has_any_codeit"]]["n_wrong_cells"].dropna().values
    n   = sub[~sub["has_any_codeit"]]["n_wrong_cells"].dropna().values
    mx  = max(np.max(w) if len(w) else 0, np.max(n) if len(n) else 0)
    bins = np.linspace(0, mx + 1, 25)
    ax.hist(w, bins=bins, alpha=0.65, color="steelblue",
            label=f"Has CodeIt (n={len(w)})", edgecolor="white")
    ax.hist(n, bins=bins, alpha=0.65, color="tomato",
            label=f"No CodeIt (n={len(n)})", edgecolor="white")
    ax.set_xlabel("n_wrong_cells")
    ax.set_ylabel("Number of tasks")
    bg_desc = "baseline=0, abstract" if bg == "A" else "baseline>0, perceptual"
    ax.set_title(f"Group {bg}  ({bg_desc})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "n_wrong_cells_comparison.png"), dpi=130)
plt.close()
print("Saved -> n_wrong_cells_comparison.png")

# ── 最终汇总打印 ──────────────────────────────────────────────────────────────

print("\n=== Selection Bias Summary ===")
print(summary_out.to_string(index=False))

print("\n=== Task Count by has_codeit × baseline_group ===")
print(df.groupby(["has_any_codeit", "baseline_group"]).size().unstack(fill_value=0))

# 额外信息：有 CodeIt 的任务里，有多少只有 failed（无 success）
only_failed = df[df["has_any_codeit"] & ~df["has_codeit_success"]]
print(f"\n  Tasks with CodeIt (failed only, no success): {len(only_failed)}")
print(f"  Tasks with CodeIt success: {df['has_codeit_success'].sum()}")

print("\n✓ S07 complete.")
print(f"  Output: {OUT_DIR}/")
