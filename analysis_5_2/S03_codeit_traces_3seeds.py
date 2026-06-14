"""
S03_codeit_traces_3seeds.py
构建 3 个 baseline seed 合并的 CodeIt 程序轨迹

数据来源：
  seed17 → h200_full_6686251_seed17.tar.gz / solutions_97.json
  seed42 → h200_full_6686252_seed42.tar.gz / solutions_95.json
  seed123 → h200_full_6686253_seed123.tar.gz / solutions_96.json

程序分类：
  success: test_performance[0] == True
  failed:  test_performance[0] != True（包括 False 和空列表）

去重流程（success 和 failed 程序均相同，不设数量上限）：
  1. 按 program string 去重（相同字符串执行结果完全一致）
  2. 执行所有去重后的程序，获取 trace
  3. 按 trace 内容去重（SHA1 hash 作为 key）
     理由：不同程序字符串可能产生完全相同的中间 grid 序列（语义等价），
     对 progress curve 分析无额外贡献，重复计数会虚增各组数量。

  不设数量上限：保留所有语义唯一的轨迹，trace 多样性本身是信息。
  seed 来源以 HHI 连续指标量化（不用硬阈值过滤）。

非独立性量化：
  每个 task 计算 HHI = Σ(seed_i 占比²)，值域 [1/3, 1]
  HHI → 1 表示几乎全来自单一 seed（traces 非独立性高）
  HHI → 1/3 表示三个 seed 均匀分布（相对独立）

执行错误处理：
  - output 是字符串（含错误信息）→ 丢弃
  - trace 为空 → 丢弃
  - 以上情况计入 n_execution_errors
"""

import os, sys, json, tarfile, csv, hashlib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# tqdm（可选，没有时退化为普通迭代）
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

# ── 路径配置 ───────────────────────────────────────────────────────────────────

REPO     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(REPO, "codelt/data/evaluation")
OUT_DIR  = os.path.join(REPO, "analysis_5_2/processed/S03_codeit_traces")
os.makedirs(OUT_DIR, exist_ok=True)

# codelt 包路径（执行 DSL 程序需要）
CODELT_PATH = os.path.join(REPO, "codelt")
if CODELT_PATH not in sys.path:
    sys.path.insert(0, CODELT_PATH)

from codeit.policy.environment import execute_candidate_program_with_trace

# seed 配置：{seed_name: (tar_gz_path, solutions_file_in_tar)}
SEEDS = {
    "seed17":  (
        os.path.expanduser("~/Downloads/codeit_data/h200_full_6686251_seed17.tar.gz"),
        "h200_full_6686251_seed17/solutions_97.json"
    ),
    "seed42":  (
        os.path.expanduser("~/Downloads/codeit_data/h200_full_6686252_seed42.tar.gz"),
        "h200_full_6686252_seed42/solutions_95.json"
    ),
    "seed123": (
        os.path.expanduser("~/Downloads/codeit_data/h200_full_6686253_seed123.tar.gz"),
        "h200_full_6686253_seed123/solutions_96.json"
    ),
}


# ── 辅助函数 ───────────────────────────────────────────────────────────────────

def grid_to_str(grid):
    """tuple-of-tuples → 管道分隔字符串。"""
    return "|" + "|".join("".join(str(c) for c in row) for row in grid) + "|"

def to_tuple_grid(g):
    """list-of-lists → tuple-of-tuples（execute 函数要求的格式）。"""
    return tuple(tuple(row) for row in g)

def trace_key(grids_list):
    """用 grids 序列的 SHA1 hash 作为轨迹内容的唯一 key，避免存大列表做 dict key。"""
    return hashlib.sha1("|".join(grids_list).encode()).hexdigest()

def compute_step_labels(program_str, trace):
    """
    给每条箭头（相邻 grid 状态之间）计算完整的 DSL 标签。

    trace 格式：[("I", input_grid), (stripped_dsl_line, grid), ...]
    对于箭头 k→k+1，标签 = 程序里从上一个 grid 行之后到本 grid 行（含）的
    所有 DSL 行，用 " → " 连接。这样可以捕获被折叠的非 grid 中间步骤，例如：
      "x2 = fgpartition(x1) → x3 = first(x2) → x4 = subgrid(x3, x1)"

    如果有多个程序产生完全相同的 grid 序列（trace 内容去重命中），只保存
    第一个程序的 step_labels（先到先得，非 grid 步骤在语义等价程序间可能不同）。
    """
    lines_stripped = [l.strip() for l in program_str.strip().split("\n")]
    traced_line_strs = [label for label, _ in trace[1:]]  # 产生了 grid 的那些行

    # 在程序行中顺序定位每条产生 grid 的行的下标
    traced_indices = []
    search_from = 0
    for tl in traced_line_strs:
        for i in range(search_from, len(lines_stripped)):
            if lines_stripped[i] == tl:
                traced_indices.append(i)
                search_from = i + 1
                break

    # 每条箭头 = 从上一个 grid 行之后到本 grid 行之间的全部 DSL 行（含本行）
    step_labels = []
    prev_idx = -1
    for li in traced_indices:
        segment = [lines_stripped[j] for j in range(prev_idx + 1, li + 1)]
        step_labels.append(" → ".join(segment))
        prev_idx = li

    return step_labels


def compute_full_steps(program_str, trace):
    """
    给程序的每一行 DSL 生成一个条目，记录该行是否产生了合法 grid。

    返回 list of dict，每个 dict：
      {"dsl": "x1 = vmirror(I)", "has_grid": True}
      {"dsl": "x2 = fgpartition(x1)", "has_grid": False}

    用途：S08 等可视化脚本可以直接读取，对 has_grid=True 的行显示真实 grid 图像，
    对 has_grid=False 的行显示占位虚线框，无需重新执行程序。
    grid 图像本身从 trace 对应的 "grids" 字段按顺序读取即可。
    """
    lines_stripped = [l.strip() for l in program_str.strip().split("\n")]
    # trace[1:] 里的 label 就是产生了 grid 的那些行（stripped）
    grid_lines = {label for label, _ in trace[1:]}
    return [{"dsl": line, "has_grid": line in grid_lines} for line in lines_stripped]


def execute_safe(program_str, input_grid_tuple):
    """
    安全执行 DSL 程序，返回 (grids_list, step_labels, full_steps, error_flag)。
    grids_list:  轨迹中每个 grid 状态的管道字符串列表（含 input grid）
    step_labels: 每条箭头的折叠 DSL 标签（len = len(grids_list) - 1）
    full_steps:  每条 DSL 行的条目列表，含 has_grid 标记（len = 程序行数）
    error_flag:  True 表示执行出错或结果无效
    """
    try:
        output, trace = execute_candidate_program_with_trace(program_str, input_grid_tuple)
    except Exception:
        return [], [], [], True

    # output 是字符串 → 执行错误
    if isinstance(output, str):
        return [], [], [], True

    # trace 为空 → 无中间状态
    if not trace:
        return [], [], [], True

    grids       = [grid_to_str(g) for (_, g) in trace]
    step_labels = compute_step_labels(program_str, trace)
    full_steps  = compute_full_steps(program_str, trace)
    return grids, step_labels, full_steps, False

# ── 加载所有 solutions 文件 ───────────────────────────────────────────────────

# 数据结构：
# per_task_programs[task_id]["success"] = {prog_str: seed_name}
# per_task_programs[task_id]["failed"]  = {prog_str: seed_name}
# （如果同一 prog_str 出现在多个 seed，保留最早的 seed）
SEED_ORDER = ["seed17", "seed42", "seed123"]

per_task_programs = defaultdict(lambda: {"success": {}, "failed": {}})

print("Loading solutions from 3 seeds...")
for seed_name in SEED_ORDER:
    tar_path, sol_file = SEEDS[seed_name]
    print(f"  {seed_name}: {sol_file}")
    with tarfile.open(tar_path) as tf:
        f = tf.extractfile(sol_file)
        data = json.load(f)

    policy = data["policy"]
    n_progs = 0
    for split_name, tasks in policy.items():
        if not isinstance(tasks, dict):
            continue
        for task_id, progs in tasks.items():
            if not isinstance(progs, dict):
                continue
            for prog_str, meta in progs.items():
                if not isinstance(meta, dict):
                    continue
                tp = meta.get("test_performance", [])
                cls = "success" if (tp and tp[0] is True) else "failed"
                # 按 seed 优先级：只有在该 seed 还没有这个程序时才加入
                if prog_str not in per_task_programs[task_id][cls]:
                    per_task_programs[task_id][cls][prog_str] = seed_name
                    n_progs += 1
    print(f"    New programs added: {n_progs}")

print(f"\nTotal tasks with any program: {len(per_task_programs)}")
print(f"Tasks with success programs: {sum(1 for t in per_task_programs.values() if t['success'])}")
print(f"Tasks with failed programs:  {sum(1 for t in per_task_programs.values() if t['failed'])}")

# ── 加载 input grids ──────────────────────────────────────────────────────────

print("\nLoading input grids...")
input_grids = {}  # task_id -> tuple-of-tuples
for fname in os.listdir(EVAL_DIR):
    if not fname.endswith(".json"):
        continue
    task_id = fname.replace(".json", "")
    with open(os.path.join(EVAL_DIR, fname)) as f:
        task = json.load(f)
    try:
        input_grids[task_id] = to_tuple_grid(task["test_examples"][0]["input"])
    except Exception:
        pass
print(f"  Loaded: {len(input_grids)} tasks")

# ── 执行程序，构建轨迹 ─────────────────────────────────────────────────────────

codeit_traces  = {}   # task_id -> list of trace dicts
seed_breakdown = []   # 用于生成 seed_breakdown.csv

tasks_to_process = sorted(per_task_programs.keys())

print(f"\nExecuting programs for {len(tasks_to_process)} tasks...")

for task_id in tqdm(tasks_to_process, desc="Tasks"):
    if task_id not in input_grids:
        # 没有 input grid 文件，跳过
        continue

    input_grid = input_grids[task_id]
    prog_data  = per_task_programs[task_id]
    task_traces = []

    # ── success 程序 ──────────────────────────────────────────────────────────
    # Step 1: program string 去重已在加载时完成
    # Step 2: 执行全部，按 trace 内容去重（语义等价程序产生相同轨迹，去重后不重复计数）
    unique_success = {}   # trace_key -> {grids, seed, step_labels, full_steps}
    n_success_errors = 0
    for prog_str, seed_name in prog_data["success"].items():
        grids, step_labels, full_steps, err = execute_safe(prog_str, input_grid)
        if err:
            n_success_errors += 1
            continue
        tk = trace_key(grids)
        if tk not in unique_success:
            unique_success[tk] = {"grids": grids, "seed": seed_name,
                                  "step_labels": step_labels,
                                  "full_steps":  full_steps}

    for item in unique_success.values():
        task_traces.append({
            "program":     "",          # trace 内容去重后不再对应唯一程序，留空
            "class":       "success",
            "seed":        item["seed"],
            "grids":       item["grids"],
            "step_labels": item["step_labels"],   # 每条箭头的折叠 DSL 标签
            "full_steps":  item["full_steps"],    # 每行 DSL 的 {dsl, has_grid} 条目
        })

    # ── failed 程序 ───────────────────────────────────────────────────────────
    # Step 1: program string 去重已在加载时完成
    # Step 2: 执行全部，按 trace 内容去重
    unique_traces = {}   # trace_key -> {grids, seed, step_labels, full_steps}
    n_failed_errors = 0
    for prog_str, seed_name in prog_data["failed"].items():
        grids, step_labels, full_steps, err = execute_safe(prog_str, input_grid)
        if err:
            n_failed_errors += 1
            continue
        tk = trace_key(grids)
        if tk not in unique_traces:
            # 首次出现的 trace 内容，记录并标注 seed 来源
            unique_traces[tk] = {"grids": grids, "seed": seed_name,
                                 "step_labels": step_labels,
                                 "full_steps":  full_steps}

    # 保留所有唯一 trace 内容，不设数量上限
    unique_list = list(unique_traces.values())
    K = len(unique_list)

    for item in unique_list:
        task_traces.append({
            "program":     "",          # trace 内容去重后不再对应唯一程序，留空
            "class":       "failed",
            "seed":        item["seed"],
            "grids":       item["grids"],
            "step_labels": item["step_labels"],   # 每条箭头的折叠 DSL 标签
            "full_steps":  item["full_steps"],    # 每行 DSL 的 {dsl, has_grid} 条目
        })

    # ── seed breakdown 统计 ───────────────────────────────────────────────────
    success_seeds = [t["seed"] for t in task_traces if t["class"] == "success"]
    failed_seeds  = [t["seed"] for t in task_traces if t["class"] == "failed"]

    def seed_counts(seed_list):
        c = {"seed17": 0, "seed42": 0, "seed123": 0}
        for s in seed_list:
            if s in c:
                c[s] += 1
        return c

    sc = seed_counts(success_seeds)
    fc = seed_counts(failed_seeds)
    n_fail = len(failed_seeds)

    # HHI for failed traces（failed 组才有非独立性问题）
    if n_fail > 0:
        p17  = fc["seed17"]  / n_fail
        p42  = fc["seed42"]  / n_fail
        p123 = fc["seed123"] / n_fail
        hhi  = round(p17**2 + p42**2 + p123**2, 4)
    else:
        hhi = float("nan")

    n_success_unique_before_dedup = len(prog_data["success"])  # program string 去重后数量
    seed_breakdown.append({
        "task_id":                          task_id,
        "n_success_traces":                 len(success_seeds),
        "n_success_prog_str_unique":        n_success_unique_before_dedup,
        "n_failed_traces":                  n_fail,
        "n_failed_unique_before_cap":       K,
        "success_seed17":             sc["seed17"],
        "success_seed42":             sc["seed42"],
        "success_seed123":            sc["seed123"],
        "failed_seed17":              fc["seed17"],
        "failed_seed42":              fc["seed42"],
        "failed_seed123":             fc["seed123"],
        "failed_hhi":                 hhi,
        "n_success_exec_errors":      n_success_errors,
        "n_failed_exec_errors":       n_failed_errors,
    })

    if task_traces:
        codeit_traces[task_id] = task_traces

# ── 保存 codeit_traces_3seeds.json ───────────────────────────────────────────

out_json = os.path.join(OUT_DIR, "codeit_traces_3seeds.json")
with open(out_json, "w") as f:
    json.dump(codeit_traces, f)
print(f"\nSaved -> {out_json}")

# ── 保存 seed_breakdown.csv ───────────────────────────────────────────────────

breakdown_df = pd.DataFrame(seed_breakdown)
out_breakdown = os.path.join(OUT_DIR, "seed_breakdown.csv")
breakdown_df.to_csv(out_breakdown, index=False)
print(f"Saved -> {out_breakdown}")

# ── 打印摘要统计 ──────────────────────────────────────────────────────────────

print("\n=== Summary Statistics ===")
n_success_tasks = sum(1 for t in codeit_traces.values()
                      if any(tr["class"] == "success" for tr in t))
n_failed_tasks  = sum(1 for t in codeit_traces.values()
                      if any(tr["class"] == "failed"  for tr in t))
print(f"Tasks with at least one success trace: {n_success_tasks}")
print(f"Tasks with at least one failed trace:  {n_failed_tasks}")
print(f"Total tasks with any trace:            {len(codeit_traces)}")

total_exec_errors = breakdown_df["n_success_exec_errors"].sum() + breakdown_df["n_failed_exec_errors"].sum()
print(f"\nTotal execution errors (discarded):    {int(total_exec_errors)}")

hhi_valid = breakdown_df["failed_hhi"].dropna()
print(f"\nFailed traces HHI distribution (n={len(hhi_valid)} tasks with failed traces):")
for p in [25, 50, 75, 90]:
    print(f"  p{p}: {hhi_valid.quantile(p/100):.3f}")
print(f"  HHI == 1.0 (all from one seed): {(hhi_valid == 1.0).sum()} tasks")
print(f"  HHI < 0.40 (relatively balanced): {(hhi_valid < 0.40).sum()} tasks")

# ── 图表 ──────────────────────────────────────────────────────────────────────

print("\nGenerating plots...")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("S03: CodeIt Traces — 3 Seeds Combined", fontsize=12, fontweight="bold")

valid = breakdown_df[breakdown_df["n_success_traces"] + breakdown_df["n_failed_traces"] > 0]

# Panel 1: success vs failed traces per task
ax = axes[0]
ax.hist(valid["n_success_traces"], bins=20, color="tab:blue",  alpha=0.7,
        edgecolor="white", label="success traces")
ax.hist(valid["n_failed_traces"],  bins=20, color="tab:orange", alpha=0.7,
        edgecolor="white", label="failed traces")
ax.set_xlabel("Number of traces per task")
ax.set_ylabel("Number of tasks")
ax.set_title("Traces per task (success vs failed)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

# Panel 2: HHI distribution for failed traces
ax = axes[1]
ax.hist(hhi_valid, bins=20, color="tomato", edgecolor="white", alpha=0.85)
ax.axvline(1/3, color="green", linestyle="--", linewidth=1.5,
           label=f"HHI=1/3 (uniform across seeds)")
ax.axvline(hhi_valid.median(), color="orange", linestyle="--", linewidth=1.5,
           label=f"Median={hhi_valid.median():.3f}")
ax.set_xlabel("HHI (seed concentration of failed traces)")
ax.set_ylabel("Number of tasks")
ax.set_title("Seed concentration (HHI) of failed traces\nHHI=1 → all from one seed")
ax.legend(fontsize=7)
ax.grid(alpha=0.3, axis="y")

# Panel 3: execution errors per task
ax = axes[2]
total_errors = valid["n_success_exec_errors"] + valid["n_failed_exec_errors"]
ax.hist(total_errors, bins=20, color="gray", edgecolor="white", alpha=0.85)
ax.axvline(total_errors.median(), color="orange", linestyle="--", linewidth=1.5,
           label=f"Median={total_errors.median():.0f}")
ax.set_xlabel("Execution errors per task (discarded programs)")
ax.set_ylabel("Number of tasks")
ax.set_title("Execution errors per task")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
out_plot = os.path.join(OUT_DIR, "codeit_traces_coverage.png")
plt.savefig(out_plot, dpi=130)
plt.close()
print(f"Saved -> {out_plot}")

print("\n✓ S03 complete.")
print(f"  Traces JSON:   {out_json}")
print(f"  Seed breakdown: {out_breakdown}")
