"""
S04_progress_curves_400.py
计算全部 400 个 ARC evaluation 任务的 progress curve

归一化公式（v2）：
  baseline(task) = progress(input_grid, target_grid)
  norm(t) = (progress(grid_at_t, target_grid) - baseline) / (1 - baseline)

  norm = 0  →  input grid（所有轨迹的起点，t=0 时 norm 恒为 0）
  norm = 1  →  target grid（完全解决）
  norm < 0  →  比 input grid 更差（有实质含义，不 clip）
  baseline = 1  →  分母为 0，该任务 input == target，跳过并记录

四个分析组（根据任务可用数据而定）：
  human_success   — 人类成功 attempt 的轨迹
  human_failed    — 人类失败 attempt 的轨迹
  codeit_success  — CodeIt 成功程序的执行轨迹（70 个任务）
  codeit_failed   — CodeIt 失败程序的执行轨迹（127 个任务）

轨迹处理：
  - human traces：prepend input_grid_str，使第一帧 norm=0，与 CodeIt 一致
    CodeIt traces 第一帧本身已是 input_grid（来自 execute_candidate_program_with_trace）
  - 每条归一化轨迹线性插值重采样到 100 个等距时间点
  - element-wise 计算 median / p25 / p75

分层变量（写入输出）：
  baseline_group: "A"（baseline==0，抽象变换类）/ "B"（baseline>0，感知搜索类）
  numerically_coarse: True 当 n_wrong_cells < 3
    理由：可以变化的格子数 < 3 时，progress 是粒度极粗的阶梯函数（每格 = 1/total_cells 跳变）

输出：
  S04_curves/progress_curves_400.json    — 含个体轨迹 + 统计量
  S04_curves/curve_summary_400.csv       — 每 task × group 的汇总指标
  S04_curves/curve_400/{task_id}.png     — 每任务一张图（1–4 条曲线）
  S04_curves/auc_overview.png            — AUC 分布总览图
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

# ── 路径配置 ───────────────────────────────────────────────────────────────────

REPO          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR      = os.path.join(REPO, "codelt/data/evaluation")
HUMAN_TRACES  = os.path.join(REPO, "analysis_5_2/processed/S02_human_traces/human_traces_all.json")
CODEIT_TRACES = os.path.join(REPO, "analysis_5_2/processed/S03_codeit_traces/codeit_traces_3seeds.json")
OUT_DIR  = os.path.join(REPO, "analysis_5_2/processed/S04_curves")
# 四个子文件夹：baseline_group × 是否有 codeit_success 数据
PLOT_SUBDIRS = {
    ("A", True):  os.path.join(OUT_DIR, "curve_400", "A_with_codeit"),
    ("A", False): os.path.join(OUT_DIR, "curve_400", "A_no_codeit"),
    ("B", True):  os.path.join(OUT_DIR, "curve_400", "B_with_codeit"),
    ("B", False): os.path.join(OUT_DIR, "curve_400", "B_no_codeit"),
}
for d in PLOT_SUBDIRS.values():
    os.makedirs(d, exist_ok=True)

# ── 辅助函数 ───────────────────────────────────────────────────────────────────

def _parse_grid(s):
    rows = s.strip("|").split("|")
    return [[int(c) for c in row] for row in rows]

def _grid_to_str(grid):
    return "|" + "|".join("".join(str(c) for c in row) for row in grid) + "|"

def progress(g_str, t_str):
    """
    计算 g_str 与目标 t_str 的 cell-level 匹配分数（0–1）。
    尺寸不匹配时返回 0.0（等价于"完全没有匹配"）。
    """
    try:
        g = _parse_grid(g_str)
        t = _parse_grid(t_str)
        if len(g) != len(t) or any(len(gr) != len(tr) for gr, tr in zip(g, t)):
            return 0.0
        total = sum(len(row) for row in t)
        match = sum(g[r][c] == t[r][c] for r in range(len(t)) for c in range(len(t[r])))
        return match / total if total > 0 else 0.0
    except Exception:
        return 0.0

def resample(values, n=100):
    """
    线性插值将任意长度的 values 重采样到 n 个等距时间点。
    单点轨迹：复制 n 次（退化情形，实际不应出现）。
    """
    if len(values) < 2:
        return [float(values[0])] * n
    x_old = np.linspace(0, 1, len(values))
    x_new = np.linspace(0, 1, n)
    return np.interp(x_new, x_old, np.array(values, dtype=float)).tolist()

def group_stats(traces_list):
    """
    给定 n 条长度均为 100 的轨迹列表，返回 element-wise median / p25 / p75。
    n=1 时 p25==median==p75（退化情形正常处理）。
    """
    arr = np.array(traces_list, dtype=float)  # shape (n, 100)
    return (
        np.median(arr, axis=0).tolist(),
        np.percentile(arr, 25, axis=0).tolist(),
        np.percentile(arr, 75, axis=0).tolist(),
    )

# ── 加载数据 ───────────────────────────────────────────────────────────────────

print("Loading evaluation task data (input + target grids)...")
task_meta = {}   # task_id -> {input_str, target_str, total_cells}
for fname in os.listdir(EVAL_DIR):
    if not fname.endswith(".json"):
        continue
    task_id = fname.replace(".json", "")
    with open(os.path.join(EVAL_DIR, fname)) as f:
        task = json.load(f)
    try:
        inp = task["test_examples"][0]["input"]
        tgt = task["test_examples"][0]["output"]
        task_meta[task_id] = {
            "input_str":    _grid_to_str(inp),
            "target_str":   _grid_to_str(tgt),
            "total_cells":  sum(len(row) for row in tgt),
        }
    except Exception:
        pass
print(f"  Loaded: {len(task_meta)} tasks")

print("Loading human traces...")
with open(HUMAN_TRACES) as f:
    human_data = json.load(f)
print(f"  Tasks: {len(human_data)}")

print("Loading CodeIt traces...")
with open(CODEIT_TRACES) as f:
    codeit_data = json.load(f)
print(f"  Tasks: {len(codeit_data)}")

# ── 主循环 ─────────────────────────────────────────────────────────────────────

print(f"\nComputing progress curves for {len(task_meta)} tasks...")

curves_json        = {}   # task_id -> full curve data (saved to JSON)
summary_rows       = []   # for curve_summary_400.csv
skipped_baseline1  = 0

GROUP_COLORS = {
    "human_success":  "tab:green",
    "human_failed":   "tab:red",
    "codeit_success": "tab:blue",
    "codeit_failed":  "dimgray",
}
GROUP_LABELS = {
    "human_success":  "Human success",
    "human_failed":   "Human failed",
    "codeit_success": "CodeIt success",
    "codeit_failed":  "CodeIt failed",
}
GROUP_ORDER = ["human_success", "human_failed", "codeit_success", "codeit_failed"]

for task_id in tqdm(sorted(task_meta.keys()), desc="Tasks"):
    meta        = task_meta[task_id]
    inp_str     = meta["input_str"]
    tgt_str     = meta["target_str"]
    total_cells = meta["total_cells"]

    # ── baseline ──────────────────────────────────────────────────────────────
    baseline = progress(inp_str, tgt_str)

    if baseline >= 1.0:
        # input grid 已等于 target grid，分母为 0，跳过
        skipped_baseline1 += 1
        continue

    denom              = 1.0 - baseline
    n_wrong_cells      = int(total_cells * denom)
    baseline_group     = "A" if baseline == 0.0 else "B"
    numerically_coarse = n_wrong_cells < 3

    # ── 构建各组原始 grid 序列 ─────────────────────────────────────────────────
    raw_groups = {g: [] for g in GROUP_ORDER}

    # Human traces：prepend input_grid_str，使 norm[0]=0（与 CodeIt 对齐）
    if task_id in human_data:
        for tr in human_data[task_id]:
            grids = [inp_str] + tr["grids"]
            key   = "human_success" if tr["success"] else "human_failed"
            raw_groups[key].append(grids)

    # CodeIt traces：第一帧已是 input_grid
    if task_id in codeit_data:
        for tr in codeit_data[task_id]:
            grids = tr["grids"]
            key   = "codeit_success" if tr["class"] == "success" else "codeit_failed"
            raw_groups[key].append(grids)

    # ── 归一化 + 重采样 ─────────────────────────────────────────────────────────
    task_entry = {
        "baseline":           round(baseline, 6),
        "baseline_group":     baseline_group,
        "n_wrong_cells":      n_wrong_cells,
        "numerically_coarse": numerically_coarse,
    }

    has_any_group = False
    for gname in GROUP_ORDER:
        traces = raw_groups[gname]
        if not traces:
            continue

        resampled_list = []
        for grids in traces:
            raw_prog  = [progress(g, tgt_str) for g in grids]
            norm_prog = [(p - baseline) / denom for p in raw_prog]
            resampled_list.append(resample(norm_prog))

        n   = len(resampled_list)
        arr = np.array(resampled_list, dtype=float)
        med, p25, p75 = group_stats(resampled_list)
        auc_median        = float(np.trapz(med, dx=1 / 99))
        pct_traces_any_neg = float((arr.min(axis=1) < 0).mean())
        pct_steps_neg      = float((arr < 0).mean())

        task_entry[gname] = {
            "n":                             n,
            "traces":                        [[round(v, 4) for v in t] for t in resampled_list],
            "median":                        [round(v, 4) for v in med],
            "p25":                           [round(v, 4) for v in p25],
            "p75":                           [round(v, 4) for v in p75],
            "auc_median":                    round(auc_median, 4),
            "pct_traces_with_any_negative":  round(pct_traces_any_neg, 4),
            "pct_steps_negative":            round(pct_steps_neg, 4),
        }

        summary_rows.append({
            "task_id":             task_id,
            "group":               gname,
            "n_traces":            n,
            "auc_median":          round(auc_median, 4),
            "pct_negative_traces": round(pct_traces_any_neg, 4),
            "pct_negative_steps":  round(pct_steps_neg, 4),
            "baseline":            round(baseline, 6),
            "baseline_group":      baseline_group,
            "n_wrong_cells":       n_wrong_cells,
            "numerically_coarse":  numerically_coarse,
        })
        has_any_group = True

    if not has_any_group:
        continue

    curves_json[task_id] = task_entry

    # ── 每任务曲线图 ──────────────────────────────────────────────────────────
    has_codeit_success = "codeit_success" in task_entry
    plot_dir = PLOT_SUBDIRS[(baseline_group, has_codeit_success)]

    x = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(7, 4))

    # 收集所有曲线的 p25 最小值，用于自适应 y 轴下限
    all_mins = []
    for gname in GROUP_ORDER:
        if gname not in task_entry:
            continue
        ge    = task_entry[gname]
        color = GROUP_COLORS[gname]
        label = f"{GROUP_LABELS[gname]} (n={ge['n']})"
        ax.plot(x, ge["median"], color=color, linewidth=2, label=label)
        if ge["n"] >= 2:
            ax.fill_between(x, ge["p25"], ge["p75"], alpha=0.15, color=color)
            all_mins.append(min(ge["p25"]))
        else:
            all_mins.append(min(ge["median"]))

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axhline(1, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Normalized time (0=start, 1=end)")
    ax.set_ylabel("Normalized progress (0=input, 1=target)")
    coarse_tag = "  [numerically coarse]" if numerically_coarse else ""
    ax.set_title(
        f"{task_id}  |  Group {baseline_group}  |  "
        f"baseline={baseline:.3f}  n_wrong_cells={n_wrong_cells}{coarse_tag}",
        fontsize=8.5,
    )
    ax.legend(fontsize=7, loc="upper left")
    # y 轴下限：自适应（留 5% 余量），上限固定 1.2
    y_min = min(all_mins) if all_mins else -0.1
    ax.set_ylim(min(y_min - 0.05, -0.05), 1.2)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{task_id}.png"), dpi=100)
    plt.close()

# ── 保存主输出 ─────────────────────────────────────────────────────────────────

print(f"\nSkipped tasks (baseline >= 1): {skipped_baseline1}")
print(f"Tasks with curves: {len(curves_json)}")

out_json = os.path.join(OUT_DIR, "progress_curves_400.json")
with open(out_json, "w") as f:
    json.dump(curves_json, f)
print(f"Saved -> {out_json}")

summary_df = pd.DataFrame(summary_rows)
out_csv = os.path.join(OUT_DIR, "curve_summary_400.csv")
summary_df.to_csv(out_csv, index=False)
print(f"Saved -> {out_csv}  ({len(summary_rows)} rows)")

# ── 总结统计 ──────────────────────────────────────────────────────────────────

print("\n=== Coverage ===")
for gname in GROUP_ORDER:
    n_tasks = sum(1 for t in curves_json.values() if gname in t)
    print(f"  {gname}: {n_tasks} tasks")

print("\n=== AUC by group × baseline_group (median) ===")
for bg in ["A", "B"]:
    sub = summary_df[summary_df["baseline_group"] == bg]
    for gname in GROUP_ORDER:
        gsub = sub[sub["group"] == gname]
        if len(gsub) == 0:
            continue
        print(f"  Group {bg} | {gname:20s}: n_tasks={len(gsub):3d}, "
              f"median_AUC={gsub['auc_median'].median():.3f}, "
              f"pct_neg_traces={gsub['pct_negative_traces'].median():.3f}")

n_coarse = summary_df[summary_df["numerically_coarse"]]["task_id"].nunique()
print(f"\nnumerically_coarse tasks (n_wrong_cells < 3): {n_coarse}")

# ── AUC 总览图 ────────────────────────────────────────────────────────────────

print("\nGenerating AUC overview plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("S04: AUC Distribution by Group and Baseline Group", fontsize=11, fontweight="bold")

for ax_idx, bg in enumerate(["A", "B"]):
    ax  = axes[ax_idx]
    sub = summary_df[summary_df["baseline_group"] == bg]

    plot_data   = []
    plot_labels = []
    for gname in GROUP_ORDER:
        gsub = sub[sub["group"] == gname]
        if len(gsub) < 3:
            continue
        plot_data.append(gsub["auc_median"].values)
        plot_labels.append(GROUP_LABELS[gname] + f"\n(n={len(gsub)})")

    if plot_data:
        bp = ax.boxplot(plot_data, patch_artist=True, widths=0.5)
        colors = [GROUP_COLORS[g] for g in GROUP_ORDER
                  if len(sub[sub["group"] == g]) >= 3]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_xticklabels(plot_labels, fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axhline(1, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Median AUC (per task)")
        ax.set_title(f"Baseline Group {bg}  "
                     f"({'baseline=0, abstract' if bg == 'A' else 'baseline>0, perceptual'})")
        ax.grid(alpha=0.25, axis="y")

plt.tight_layout()
out_plot = os.path.join(OUT_DIR, "auc_overview.png")
plt.savefig(out_plot, dpi=130)
plt.close()
print(f"Saved -> {out_plot}")

# ── human_success vs codeit_success AUC 散点图（70 个有 CodeIt 数据的任务）──────

print("Generating human_success vs codeit_success AUC scatter...")

hs_auc = summary_df[summary_df["group"] == "human_success"][["task_id", "auc_median", "baseline_group"]].rename(columns={"auc_median": "hs_auc"})
cs_auc = summary_df[summary_df["group"] == "codeit_success"][["task_id", "auc_median"]].rename(columns={"auc_median": "cs_auc"})
scatter_df = hs_auc.merge(cs_auc, on="task_id")

if len(scatter_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Human success AUC vs CodeIt success AUC (70 tasks)", fontsize=11, fontweight="bold")

    for ax_idx, bg in enumerate(["A", "B"]):
        ax  = axes[ax_idx]
        sub = scatter_df[scatter_df["baseline_group"] == bg]
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        ax.scatter(sub["hs_auc"], sub["cs_auc"], alpha=0.6, edgecolors="white", s=50)
        lim = [min(sub["hs_auc"].min(), sub["cs_auc"].min()) - 0.05,
               max(sub["hs_auc"].max(), sub["cs_auc"].max()) + 0.05]
        ax.plot(lim, lim, color="gray", linewidth=1, linestyle="--", label="y=x")
        ax.set_xlabel("Human success AUC")
        ax.set_ylabel("CodeIt success AUC")
        ax.set_title(f"Group {bg}  (n={len(sub)})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    out_scatter = os.path.join(OUT_DIR, "auc_scatter_hs_vs_cs.png")
    plt.savefig(out_scatter, dpi=130)
    plt.close()
    print(f"Saved -> {out_scatter}")

print("\n✓ S04 complete.")
print(f"  JSON:        {out_json}")
print(f"  Summary CSV: {out_csv}")
print(f"  Per-task plots: {OUT_DIR}/curve_400/  (subdirs: A_with_codeit, A_no_codeit, B_with_codeit, B_no_codeit)")
