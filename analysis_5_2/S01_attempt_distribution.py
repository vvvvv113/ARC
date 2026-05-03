"""
S01_attempt_distribution.py
验证"最后一次 attempt 是最有信息量的"假设

研究问题：
- 参与者通常尝试同一个任务几次？
- 最后一次 attempt 的行动数量是否足够多（轨迹信息量是否充足）？
- 随着 attempt 次数增加，参与者是否真的越来越接近正确答案？

如果发现超过 20% 的参与者最后一次 attempt 只有 ≤ 3 个动作，
"最后一次 attempt"假设需要在 S02 中重新讨论，请暂停并告知用户。

分析单元：每个 (hashed_id, task_id) 组合（一个参与者在一个任务上的所有尝试）
数据范围：仅 task_type == 'evaluation'（400 个 eval 任务）
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 路径配置 ───────────────────────────────────────────────────────────────────

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV   = os.path.join(REPO, "human_data/data/data.csv")
EVAL_DIR   = os.path.join(REPO, "codelt/data/evaluation")
OUT_DIR    = os.path.join(REPO, "analysis_5_2/processed/S01_attempts")
os.makedirs(OUT_DIR, exist_ok=True)

# ── progress 计算函数 ─────────────────────────────────────────────────────────

def _parse_grid(s):
    """将管道分隔字符串解析为二维列表。"""
    rows = s.strip("|").split("|")
    return [[int(c) for c in row] for row in rows]

def _grid_to_str(grid):
    """将二维列表转为管道分隔字符串。"""
    return "|" + "|".join("".join(str(c) for c in row) for row in grid) + "|"

def progress(g_str, t_str):
    """
    计算 g_str 与目标 t_str 的匹配分数（cell-matching）。
    = 匹配格子数 / 总格子数
    返回 0.0 如果尺寸不一致或解析失败。
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

# ── 加载任务 target grid ──────────────────────────────────────────────────────

print("加载任务 target grid...")
target_grids = {}  # task_id -> target_grid_str
for fname in os.listdir(EVAL_DIR):
    if not fname.endswith(".json"):
        continue
    task_id = fname.replace(".json", "")
    with open(os.path.join(EVAL_DIR, fname)) as f:
        task = json.load(f)
    try:
        tgt = task["test_examples"][0]["output"]
        target_grids[task_id] = _grid_to_str(tgt)
    except Exception:
        pass
print(f"  加载完成：{len(target_grids)} 个任务")

# ── 加载人类数据 ──────────────────────────────────────────────────────────────

print("加载 data.csv...")
df = pd.read_csv(DATA_CSV)

# 只保留 evaluation 任务
df = df[df["task_type"] == "evaluation"].copy()

# task_id = task_name 去掉 .json 后缀
df["task_id"] = df["task_name"].str.replace(".json", "", regex=False)

print(f"  Evaluation 行数：{len(df)}")
print(f"  唯一 task_id 数：{df['task_id'].nunique()}")
print(f"  唯一 hashed_id 数：{df['hashed_id'].nunique()}")

# ── 每个 (hashed_id, task_id) 的基本统计 ────────────────────────────────────

print("\n计算每个参与者×任务的尝试次数统计...")

# max_attempt：每个参与者在每个任务上的最多尝试次数
attempt_stats = (
    df.groupby(["hashed_id", "task_id"])["attempt_number"]
    .max()
    .reset_index()
    .rename(columns={"attempt_number": "max_attempt"})
)

# last_attempt_rows：每个参与者在每个任务上最后一次 attempt 的所有行
last_attempt_idx = df.groupby(["hashed_id", "task_id"])["attempt_number"].transform("max")
last_df = df[df["attempt_number"] == last_attempt_idx].copy()

# 最后一次 attempt 的行动数（每个参与者×任务）
last_action_counts = (
    last_df.groupby(["hashed_id", "task_id"])["action_id"]
    .count()
    .reset_index()
    .rename(columns={"action_id": "last_attempt_n_actions"})
)

# 最后一次 attempt 的最终 progress（最后一个 action 对应的 test_output_grid）
def last_progress(group, task_id):
    """计算该组中最后一个非空 grid 的 progress 值。"""
    if task_id not in target_grids:
        return float("nan")
    # 按 action_id 排序，取最后一行的 test_output_grid
    sorted_g = group.sort_values("action_id")
    last_grid = sorted_g["test_output_grid"].dropna().iloc[-1] if not sorted_g["test_output_grid"].dropna().empty else None
    if last_grid is None:
        return float("nan")
    return progress(str(last_grid), target_grids[task_id])

print("  计算最终 progress（耗时约 1-2 分钟）...")
final_progress_list = []
for (hid, tid), group in last_df.groupby(["hashed_id", "task_id"]):
    p = last_progress(group, tid)
    final_progress_list.append({"hashed_id": hid, "task_id": tid, "last_attempt_final_progress": p})
final_progress_df = pd.DataFrame(final_progress_list)

# 合并
attempt_stats = attempt_stats.merge(last_action_counts, on=["hashed_id", "task_id"])
attempt_stats = attempt_stats.merge(final_progress_df, on=["hashed_id", "task_id"])

# 保存
out_csv = os.path.join(OUT_DIR, "attempt_stats.csv")
attempt_stats.to_csv(out_csv, index=False)
print(f"  保存 -> {out_csv}  ({len(attempt_stats)} 行)")

# ── 关键统计摘要 ──────────────────────────────────────────────────────────────

print("\n=== 关键统计摘要 ===")

n_total = len(attempt_stats)
print(f"总参与者×任务条目数：{n_total}")

# max attempt 分布
print("\nmax_attempt 分布（参与者在同一任务尝试了几次）：")
print(attempt_stats["max_attempt"].value_counts().sort_index().head(10).to_string())

# 最后一次 attempt 的行动数
print("\nlast_attempt_n_actions 分布：")
for p in [10, 25, 50, 75, 90]:
    print(f"  p{p}: {attempt_stats['last_attempt_n_actions'].quantile(p/100):.1f}")

# 行动数 ≤ 3 的比例（关键检查）
pct_few = (attempt_stats["last_attempt_n_actions"] <= 3).mean()
print(f"\n⚠️  last_attempt_n_actions ≤ 3 的比例：{pct_few*100:.1f}%")
if pct_few > 0.20:
    print("  ❌ 超过 20%！'最后一次 attempt'假设需要重新讨论，请告知用户。")
else:
    print("  ✓ 低于 20%，假设可以接受。")

# ── 验证"越来越接近正确答案"假设 ─────────────────────────────────────────────

print("\n=== 验证：随 attempt 增加，progress 是否提高？ ===")

# 对于有多次 attempt 的参与者，计算每次 attempt 的最终 progress
def compute_attempt_final_progress(df, target_grids):
    """计算每个参与者×任务×attempt 的最终 progress。"""
    rows = []
    for (hid, tid, att), group in df.groupby(["hashed_id", "task_id", "attempt_number"]):
        if tid not in target_grids:
            continue
        sorted_g = group.sort_values("action_id")
        last_grid = sorted_g["test_output_grid"].dropna()
        if last_grid.empty:
            continue
        p = progress(str(last_grid.iloc[-1]), target_grids[tid])
        rows.append({"hashed_id": hid, "task_id": tid, "attempt_number": att, "final_progress": p})
    return pd.DataFrame(rows)

print("  计算每次 attempt 的最终 progress（耗时约 2-3 分钟）...")
per_attempt_df = compute_attempt_final_progress(df, target_grids)

out_per_attempt = os.path.join(OUT_DIR, "per_attempt_progress.csv")
per_attempt_df.to_csv(out_per_attempt, index=False)
print(f"  保存 -> {out_per_attempt}")

# 按 attempt_number 分组，计算中位数最终 progress
trend = (
    per_attempt_df.groupby("attempt_number")["final_progress"]
    .agg(["median", "count"])
    .reset_index()
    .rename(columns={"median": "median_final_progress", "count": "n_entries"})
)
print("\nattempt_number -> 中位数最终 progress：")
print(trend.to_string(index=False))

# ── 绘图 ──────────────────────────────────────────────────────────────────────

print("\n生成图表...")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Panel 1：max attempt 分布
ax = axes[0, 0]
max_attempt_counts = attempt_stats["max_attempt"].value_counts().sort_index()
ax.bar(max_attempt_counts.index, max_attempt_counts.values, color="steelblue", edgecolor="white")
ax.set_xlabel("最多尝试次数（max attempt_number）")
ax.set_ylabel("参与者×任务 条目数")
ax.set_title("参与者在同一任务上的最多尝试次数分布")
ax.grid(alpha=0.3, axis="y")

# Panel 2：last attempt 行动数分布
ax = axes[0, 1]
n_actions = attempt_stats["last_attempt_n_actions"].clip(upper=50)
ax.hist(n_actions, bins=30, color="salmon", edgecolor="white", alpha=0.85)
ax.axvline(3, color="red", linestyle="--", linewidth=1.5, label="n_actions = 3（信息量下限）")
ax.axvline(attempt_stats["last_attempt_n_actions"].median(), color="orange",
           linestyle="--", linewidth=1.5, label=f"中位数 = {attempt_stats['last_attempt_n_actions'].median():.0f}")
ax.set_xlabel("最后一次 attempt 的行动数（上限截断为 50）")
ax.set_ylabel("条目数")
ax.set_title(f"最后一次 attempt 的行动数分布\n（≤3 的比例：{pct_few*100:.1f}%）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

# Panel 3：attempt_number vs 中位数最终 progress（验证越来越接近）
ax = axes[1, 0]
ax.plot(trend["attempt_number"], trend["median_final_progress"],
        marker="o", color="steelblue", linewidth=2)
for _, row in trend.iterrows():
    ax.annotate(f"n={int(row['n_entries'])}", (row["attempt_number"], row["median_final_progress"]),
                textcoords="offset points", xytext=(4, 4), fontsize=7)
ax.set_xlabel("attempt_number")
ax.set_ylabel("中位数最终 progress（0=输入状态, 1=目标状态）")
ax.set_title("随尝试次数增加，最终 progress 是否提高？")
ax.grid(alpha=0.3)
ax.set_ylim(0, 1.05)

# Panel 4：最后一次 attempt 的最终 progress 分布
ax = axes[1, 1]
fp = attempt_stats["last_attempt_final_progress"].dropna()
ax.hist(fp, bins=30, color="mediumpurple", edgecolor="white", alpha=0.85)
ax.axvline(fp.median(), color="orange", linestyle="--", linewidth=1.5,
           label=f"中位数 = {fp.median():.3f}")
ax.set_xlabel("最终 progress（最后一帧与目标的匹配度）")
ax.set_ylabel("条目数")
ax.set_title("最后一次 attempt 的最终 progress 分布")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
out_plot = os.path.join(OUT_DIR, "attempt_distribution.png")
plt.savefig(out_plot, dpi=130)
plt.close()
print(f"  保存 -> {out_plot}")

# Panel：attempt progress trend（单独保存）
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(trend["attempt_number"], trend["median_final_progress"],
         marker="o", color="steelblue", linewidth=2, label="中位数 final progress")
ax2.fill_between(
    per_attempt_df.groupby("attempt_number")["final_progress"].agg(lambda x: x.quantile(0.25)).index,
    per_attempt_df.groupby("attempt_number")["final_progress"].agg(lambda x: x.quantile(0.25)).values,
    per_attempt_df.groupby("attempt_number")["final_progress"].agg(lambda x: x.quantile(0.75)).values,
    alpha=0.2, color="steelblue", label="IQR（p25–p75）"
)
ax2.set_xlabel("attempt_number（第几次尝试）")
ax2.set_ylabel("最终 progress（0=输入, 1=目标）")
ax2.set_title("随尝试次数增加，参与者最终 progress 的变化趋势")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.set_ylim(0, 1.05)
plt.tight_layout()
out_trend = os.path.join(OUT_DIR, "attempt_progress_trend.png")
plt.savefig(out_trend, dpi=130)
plt.close()
print(f"  保存 -> {out_trend}")

# ── 保存摘要文本 ──────────────────────────────────────────────────────────────

summary_lines = [
    "=== S01 Attempt Distribution Summary ===",
    f"总条目数（hashed_id × task_id）：{n_total}",
    f"max_attempt 分布：",
    attempt_stats["max_attempt"].value_counts().sort_index().head(10).to_string(),
    f"\nlast_attempt_n_actions 统计：",
    attempt_stats["last_attempt_n_actions"].describe().to_string(),
    f"\n≤ 3 行动的比例：{pct_few*100:.2f}%",
    f"{'❌ 超过阈值！' if pct_few > 0.20 else '✓ 低于 20% 阈值，假设成立'}",
    "\nattempt_number vs 中位数 final progress：",
    trend.to_string(index=False),
]
with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
    f.write("\n".join(summary_lines))
print(f"  保存 -> {os.path.join(OUT_DIR, 'summary.txt')}")

print("\n✓ S01 完成。请检查 processed/S01_attempts/ 下的结果，确认假设是否成立后再运行 S02。")
