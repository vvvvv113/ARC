"""
S02_human_traces_all.py
构建全部 400 个 ARC evaluation 任务的人类轨迹

设计决策（详见 plan log-linked-quilt.md）：
- 分析单元：每个参与者在每个任务上的 last attempt（attempt_number 最大的一次）
  理由：last attempt 是参与者最充分了解任务后的最终策略；
  S01 验证了 51.8% 的多次尝试参与者在 last attempt 中进步更多。
- test_output_grid：参与者每步操作后工作区的输出 grid 状态（不是输入 grid）
- 相邻相同 grid 去重：change_color 等操作在 UI 层改变颜色映射，不改变 grid 内容，
  会产生冗余帧，去重后轨迹只保留实质状态变化。
- 不 clip 负值：v2 归一化后 norm(t) < 0 表示比初始状态更差，是有效信息，
  在 S04 中统计其频率。
- 空 grid 行（test_output_grid 为 NaN）跳过，不进入轨迹序列。
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 路径配置 ───────────────────────────────────────────────────────────────────

REPO    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(REPO, "human_data/data/data.csv")
OUT_DIR  = os.path.join(REPO, "analysis_5_2/processed/S02_human_traces")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 加载数据 ───────────────────────────────────────────────────────────────────

print("Loading data.csv...")
df = pd.read_csv(DATA_CSV)

# 只保留 evaluation 任务
df = df[df["task_type"] == "evaluation"].copy()

# task_id：去掉 .json 后缀
df["task_id"] = df["task_name"].str.replace(".json", "", regex=False)

print(f"  Evaluation rows: {len(df)}")
print(f"  Unique tasks: {df['task_id'].nunique()}")
print(f"  Unique participants: {df['hashed_id'].nunique()}")

# ── 找到每个 (hashed_id, task_id) 的 last attempt ────────────────────────────

# 用 transform("max") 找到每个组内的最大 attempt_number
# 只保留等于最大值的行（即 last attempt 的所有行）
last_attempt_mask = (
    df.groupby(["hashed_id", "task_id"])["attempt_number"]
    .transform("max")
)
last_df = df[df["attempt_number"] == last_attempt_mask].copy()

print(f"  Rows in last attempt: {len(last_df)}")

# ── 构建轨迹 ───────────────────────────────────────────────────────────────────

print("Building trajectories...")

human_traces = {}        # task_id -> list of trace dicts
skipped_empty = 0        # 因 grid 序列为空而跳过的条目数
n_dedup_removed = 0      # 被去重移除的相邻重复 grid 总数

for (task_id, hashed_id), group in last_df.groupby(["task_id", "hashed_id"]):
    # 按 action_id 升序排列，保证时间顺序
    group = group.sort_values("action_id")

    # success：该 attempt 中任何一行的 solved == True
    success = bool(group["solved"].any())

    # 收集 test_output_grid 序列
    # 防空处理：
    #   1. 跳过 NaN（dropna）
    #   2. 跳过空字符串或非字符串值（isinstance + strip 检查）
    #   某些行 test_output_grid 可能是空字符串而非 NaN，不能只用 dropna
    raw_grids = [
        g for g in group["test_output_grid"].dropna().tolist()
        if isinstance(g, str) and g.strip()
    ]

    if not raw_grids:
        # 该 attempt 没有任何有效 grid 状态，跳过
        skipped_empty += 1
        continue

    # 去重相邻相同 grid
    # 原因：change_color/change_height/change_width 等操作不修改 grid 内容，
    # 会产生连续重复帧；去重后轨迹只保留实质状态转变点。
    deduped = [raw_grids[0]]
    for g in raw_grids[1:]:
        if g != deduped[-1]:
            deduped.append(g)
        else:
            n_dedup_removed += 1

    if task_id not in human_traces:
        human_traces[task_id] = []

    human_traces[task_id].append({
        "hashed_id": hashed_id,
        "success":   success,
        "grids":     deduped,
    })

print(f"  Tasks with traces: {len(human_traces)}")
print(f"  Skipped (empty grid sequence): {skipped_empty}")
print(f"  Duplicate frames removed by dedup: {n_dedup_removed}")

# ── 保存 human_traces_all.json ────────────────────────────────────────────────

out_json = os.path.join(OUT_DIR, "human_traces_all.json")
with open(out_json, "w") as f:
    json.dump(human_traces, f)
print(f"\nSaved -> {out_json}")

# ── 生成 summary CSV ──────────────────────────────────────────────────────────

summary_rows = []
for task_id, traces in human_traces.items():
    n_success = sum(1 for t in traces if t["success"])
    n_failed  = sum(1 for t in traces if not t["success"])
    n_total   = len(traces)
    avg_len   = np.mean([len(t["grids"]) for t in traces])
    summary_rows.append({
        "task_id":   task_id,
        "n_success": n_success,
        "n_failed":  n_failed,
        "n_total":   n_total,
        "avg_grids_per_trace": round(avg_len, 1),
    })

summary_df = pd.DataFrame(summary_rows).sort_values("task_id")
out_csv = os.path.join(OUT_DIR, "human_traces_summary.csv")
summary_df.to_csv(out_csv, index=False)
print(f"Saved -> {out_csv}")

# ── 打印统计摘要 ──────────────────────────────────────────────────────────────

print("\n=== Summary Statistics ===")
print(f"Total tasks with at least one trace: {len(human_traces)}")
print(f"Tasks with at least one success trace: {(summary_df['n_success'] > 0).sum()}")
print(f"Tasks with at least one failed trace:  {(summary_df['n_failed'] > 0).sum()}")
print(f"\nn_total per task:")
for p in [10, 25, 50, 75, 90]:
    print(f"  p{p}: {summary_df['n_total'].quantile(p/100):.1f}")
print(f"\navg_grids_per_trace per task:")
for p in [25, 50, 75]:
    print(f"  p{p}: {summary_df['avg_grids_per_trace'].quantile(p/100):.1f}")

# ── 验证：success 组最终 grid 应接近 target ─────────────────────────────────
# 此处仅做快速抽查，不依赖 evaluation 文件
# 完整验证在 S04（归一化时会直接用到 target grid）

# ── 生成图表（全英文）────────────────────────────────────────────────────────

print("\nGenerating plots...")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("S02: Human Traces Coverage (400 eval tasks)", fontsize=12, fontweight="bold")

# Panel 1：每个任务的参与者总数分布
ax = axes[0]
ax.hist(summary_df["n_total"], bins=30, color="steelblue", edgecolor="white", alpha=0.85)
ax.axvline(summary_df["n_total"].median(), color="orange", linestyle="--", linewidth=1.5,
           label=f"Median = {summary_df['n_total'].median():.0f}")
ax.set_xlabel("Number of traces per task")
ax.set_ylabel("Number of tasks")
ax.set_title("Traces per task (success + failed)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

# Panel 2：success vs failed traces 分布
ax = axes[1]
ax.hist(summary_df["n_success"], bins=20, color="tab:green", edgecolor="white",
        alpha=0.7, label="success traces")
ax.hist(summary_df["n_failed"],  bins=20, color="tab:red",   edgecolor="white",
        alpha=0.7, label="failed traces")
ax.set_xlabel("Number of traces per task")
ax.set_ylabel("Number of tasks")
ax.set_title("Success vs Failed traces per task")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

# Panel 3：平均 grid 序列长度分布
ax = axes[2]
ax.hist(summary_df["avg_grids_per_trace"], bins=30, color="mediumpurple",
        edgecolor="white", alpha=0.85)
ax.axvline(summary_df["avg_grids_per_trace"].median(), color="orange",
           linestyle="--", linewidth=1.5,
           label=f"Median = {summary_df['avg_grids_per_trace'].median():.1f}")
ax.set_xlabel("Average grids per trace (after dedup)")
ax.set_ylabel("Number of tasks")
ax.set_title("Average trajectory length per task")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
out_plot = os.path.join(OUT_DIR, "human_traces_coverage.png")
plt.savefig(out_plot, dpi=130)
plt.close()
print(f"Saved -> {out_plot}")

print("\n✓ S02 complete.")
print(f"  Output: {out_json}")
print(f"  Summary: {out_csv}")
