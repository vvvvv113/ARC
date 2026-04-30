"""
Evaluate Method 2 (DTW trajectory bias) by comparing baseline vs biased solutions.

整体逻辑：
  给定两个 CodeIt 训练结果（baseline = 不加 human bias，biased = 加了 human bias），
  对每个 task 的每个 CodeIt program，计算它的 progress curve 和 human trajectories 的
  DTW similarity。比较两个版本在以下维度上的差异：
    1. Solve rate：是否加 bias 会损害任务完成率
    2. DTW similarity：加 bias 后的程序是否更像人类的解题路径
    3. Wasserstein distance：CodeIt 整体的 progress 分布是否向人类分布靠近
    4. Per-task delta：哪些难度类别的 task 从 bias 中受益最多

Usage:
    python analysis/07_method2_eval.py \
        --baseline data/solutions_baseline.json \
        --biased   data/solutions_biased.json

If only --baseline is provided, prints stats for that run alone.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # 不弹出窗口，直接保存图片（适合服务器环境）
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wasserstein_distance

# ── 路径设置 ────────────────────────────────────────────────────────────────────

# __file__ 是当前脚本路径（analysis/07_method2_eval.py）
# dirname 两次得到 repo 根目录
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 把 repo 根目录和 codelt 子模块加入 Python 路径，
# 这样才能 import codeit.human_trajectories（在 codelt/ 子模块里）
sys.path.insert(0, REPO)
sys.path.insert(1, os.path.join(REPO, "codelt"))

# 从 codelt 子模块中导入三个核心函数：
#   build_human_curves     : 读取 human_traces.json，返回每个 task 的 progress curves 列表
#   compute_dtw_similarity : 计算单条 program curve 与一组 human curves 的 DTW 相似度
#   get_program_curve      : 执行一个 DSL program，提取逐步的 progress curve
from codeit.human_trajectories import (
    build_human_curves,
    compute_dtw_similarity,
    get_program_curve,
)

# ── 文件路径常量 ────────────────────────────────────────────────────────────────

# 注意：这里的路径是旧路径（没有 04_human_traces/ 子文件夹），
# 如果在本地运行报错，需要改为：
# analysis/processed/04_human_traces/human_traces.json
HUMAN_TRACES = os.path.join(REPO, "analysis/processed/human_traces.json")

# 每个 task 的 input/target grid 存在这里（JSON 格式）
EVAL_DIR = os.path.join(REPO, "codelt/data/evaluation")

# task 难度分类表（Easy for both / Hard for both / Only hard for AI / Only hard for humans）
DIFFICULTY_CSV = os.path.join(REPO, "analysis/processed/01_difficulty/task_difficulty.csv")

# 输出图片保存目录
PLOTS_DIR = os.path.join(REPO, "analysis/processed/method2_eval")


# ── 辅助函数：加载 task 的 input 和 target grid ────────────────────────────────

def load_input_target(task_id):
    """
    从 codelt/data/evaluation/{task_id}.json 读取测试样例的 input 和 output grid。

    返回值：
      input_grid  : tuple of tuples，表示 ARC task 的输入网格
      target_grid : tuple of tuples，表示 ARC task 的目标输出网格

    用 tuple 而不是 list 是因为 tuple 可哈希，方便后续缓存或用作 dict key。
    """
    path = os.path.join(EVAL_DIR, f"{task_id}.json")
    with open(path) as f:
        task_data = json.load(f)
    # test_examples[0] 是第一个（通常也是唯一一个）测试样例
    input_grid  = tuple(tuple(r) for r in task_data["test_examples"][0]["input"])
    target_grid = tuple(tuple(r) for r in task_data["test_examples"][0]["output"])
    return input_grid, target_grid


# ── 核心评估函数 ───────────────────────────────────────────────────────────────

def evaluate_solutions(solutions_path, human_curves):
    """
    读取一个 CodeIt 训练结果文件（baseline 或 biased），
    对每个 task 的每个 program 计算 DTW similarity，并汇总统计。

    参数：
      solutions_path : str，solutions JSON 文件路径（baseline 或 biased）
      human_curves   : dict，{task_id: [curve_1, curve_2, ...]},
                       每条 curve 是长度为 100 的 progress 列表（由 build_human_curves 生成）

    返回：
      per_task         : dict，每个 task 的结果 {task_id: {solved, mean_dtw_sim, n_programs}}
      solve_rate       : float，解决的 task 占总 task 数的比例
      mean_dtw         : float，所有 program 的平均 DTW similarity
      median_dtw       : float，所有 program 的中位 DTW similarity
      all_dtw_sims     : list，每个 program 的 DTW similarity 值（用于假设检验）
      all_program_curves: list，每个 program 的 progress curve（用于 Wasserstein 计算）
    """
    with open(solutions_path) as f:
        data = json.load(f)

    # solutions JSON 的结构：{"policy": {"task_demonstration": {task_id: {program_str: meta}}}}
    # 也可能直接是 {"task_demonstration": {...}}，所以用 .get("policy", data) 做兼容
    tasks = data.get("policy", data).get("task_demonstration", {})

    per_task     = {}   # 存储每个 task 的结果
    all_dtw_sims = []   # 存储所有 program 的 DTW similarity（跨 task）
    n_tasks      = 0    # 有 human curves 的 task 总数
    n_solved     = 0    # 被解决的 task 数

    for task_id, programs in tasks.items():
        # 跳过没有 human curves 的 task（这些 task 无法计算 DTW similarity）
        if task_id not in human_curves:
            continue

        curves = human_curves[task_id]  # 该 task 的所有 human progress curves

        # 加载 input/target grid，失败则跳过（可能是评估文件缺失）
        try:
            input_grid, target_grid = load_input_target(task_id)
        except Exception:
            continue

        n_tasks     += 1
        task_solved  = False
        task_sims    = []  # 该 task 下所有 program 的 DTW similarity

        for program_str, meta in programs.items():
            # ── 判断该 program 是否解决了 task ──────────────────────────────
            # task_demonstration_performance 是一个 bool 列表，表示程序在每个 demo 样例上是否正确
            # any(perf) = True 表示至少有一个样例被正确解决
            perf = meta.get("task_demonstration_performance", [])
            if isinstance(perf, list) and any(perf):
                task_solved = True
            elif perf:  # 如果不是列表但是 truthy（比如直接是 True）
                task_solved = True

            # ── 计算该 program 的 progress curve ────────────────────────────
            # get_program_curve 执行 DSL program，记录每一步的 progress（cells matching target / total cells）
            # 返回长度为 100 的列表（已线性插值到归一化时间轴）
            curve = get_program_curve(program_str, input_grid, target_grid)

            # ── 计算 DTW similarity ──────────────────────────────────────────
            # compute_dtw_similarity 计算该 program curve 与所有 human curves 的 DTW distance，
            # 取最小值（最相似的那条 human curve），然后转换为 similarity：
            #   similarity = 1 / (1 + min_DTW_distance)
            # similarity 范围 (0, 1]，越接近 1 越像人类
            sim = compute_dtw_similarity(curve, curves)
            task_sims.append(sim)
            all_dtw_sims.append(sim)

        if task_solved:
            n_solved += 1

        # 记录该 task 的汇总结果
        per_task[task_id] = {
            "solved":       task_solved,
            "mean_dtw_sim": float(np.mean(task_sims)) if task_sims else 0.0,
            "n_programs":   len(task_sims),
        }

    # ── 全局汇总统计 ────────────────────────────────────────────────────────────
    solve_rate  = n_solved / n_tasks if n_tasks > 0 else 0.0
    mean_dtw    = float(np.mean(all_dtw_sims))   if all_dtw_sims else 0.0
    median_dtw  = float(np.median(all_dtw_sims)) if all_dtw_sims else 0.0

    # ── 重新收集所有 program 的 curve，用于 Wasserstein 计算 ─────────────────────
    # 注意：这里重新遍历一次是因为上面的循环里没有保存 curve（只保存了 sim 值）
    # 这是一个可以优化的地方（避免重复执行程序），但对正确性没有影响
    all_program_curves = []
    for task_id, programs in tasks.items():
        if task_id not in human_curves:
            continue
        try:
            input_grid, target_grid = load_input_target(task_id)
        except Exception:
            continue
        for program_str in programs:
            curve = get_program_curve(program_str, input_grid, target_grid)
            if curve is not None:
                all_program_curves.append(curve)

    return per_task, solve_rate, mean_dtw, median_dtw, all_dtw_sims, all_program_curves


# ── Wasserstein distance 计算 ──────────────────────────────────────────────────

def compute_wasserstein(program_curves, human_curves_dict):
    """
    用 Wasserstein distance（earth mover's distance）衡量 CodeIt 的 progress 分布
    与 human 的 progress 分布之间的差距。

    做法：
      将每条 curve（长度 100 的序列）压缩成一个标量——curve 的均值（即 AUC 的代理指标）。
      AUC 高 = 早期就有很多进展（快速收敛）；AUC 低 = 进展很慢或没有进展。
      然后计算两个 AUC 分布之间的 Wasserstein distance（1D Earth Mover's Distance）。

    Wasserstein distance 越小，说明 CodeIt 的解题节奏分布越接近人类。
    如果 biased 版本的 Wasserstein < baseline 版本，说明 bias 成功地让 CodeIt 的
    整体行为模式向人类靠近。
    """
    # 每个 program curve 的均值（代表该程序的整体进展速度）
    ai_aucs    = [float(np.mean(c)) for c in program_curves]
    # 把所有 task 的所有 human curves 展平，计算每条 human curve 的均值
    human_aucs = [float(np.mean(c)) for curves in human_curves_dict.values() for c in curves]

    if not ai_aucs or not human_aucs:
        return float("nan")

    # scipy 的 wasserstein_distance 计算两个 1D 分布之间的 Earth Mover's Distance
    return float(wasserstein_distance(ai_aucs, human_aucs))


# ── 打印报告 ───────────────────────────────────────────────────────────────────

def print_report(label, per_task, solve_rate, mean_dtw, median_dtw, wass):
    """打印单个版本（baseline 或 biased）的汇总统计到终端。"""
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Tasks evaluated:        {len(per_task)}")
    print(f"  Solve rate:             {solve_rate:.3f}")
    print(f"  Mean DTW similarity:    {mean_dtw:.4f}")
    print(f"  Median DTW similarity:  {median_dtw:.4f}")
    print(f"  Wasserstein distance:   {wass:.4f}  (lower = more human-like distribution)")


# ── 主函数 ─────────────────────────────────────────────────────────────────────

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline",     required=True,        help="baseline solutions JSON 路径")
    parser.add_argument("--biased",       default=None,         help="biased solutions JSON 路径（可选）")
    parser.add_argument("--human-traces", default=HUMAN_TRACES, help="human_traces.json 路径")
    args = parser.parse_args()

    # ── Step 1：加载 human trajectory curves ───────────────────────────────────
    # build_human_curves 读取 human_traces.json，
    # 对每条 human trajectory 计算 progress curve 并插值到 100 个点
    # 返回 {task_id: [curve_1, curve_2, ...]}
    print("Loading human trajectory curves...")
    human_curves = build_human_curves(args.human_traces)
    print(f"Loaded curves for {len(human_curves)} tasks")

    # ── Step 2：评估 baseline（λ=0，原始 CodeIt，不加 human bias）─────────────
    print("\nEvaluating baseline...")
    b_per_task, b_solve, b_mean_dtw, b_med_dtw, b_sims, b_curves = evaluate_solutions(
        args.baseline, human_curves
    )
    b_wass = compute_wasserstein(b_curves, human_curves)
    print_report("Baseline (λ=0)", b_per_task, b_solve, b_mean_dtw, b_med_dtw, b_wass)

    # ── Step 3：如果提供了 biased 文件，对比两个版本 ──────────────────────────
    if args.biased:
        print("\nEvaluating biased...")
        h_per_task, h_solve, h_mean_dtw, h_med_dtw, h_sims, h_curves = evaluate_solutions(
            args.biased, human_curves
        )
        h_wass = compute_wasserstein(h_curves, human_curves)
        print_report("Human-biased (λ>0)", h_per_task, h_solve, h_mean_dtw, h_med_dtw, h_wass)

        # ── Delta 汇总：biased - baseline ────────────────────────────────────
        # solve rate delta 负值 = bias 损害了任务完成率（代价）
        # DTW sim delta 正值 = bias 让程序更像人类（收益）
        # Wasserstein delta 负值 = bias 让整体分布更接近人类（收益）
        print(f"\n{'='*50}")
        print("  Delta (biased - baseline)")
        print(f"{'='*50}")
        print(f"  Solve rate delta:       {h_solve - b_solve:+.3f}")
        print(f"  Mean DTW sim delta:     {h_mean_dtw - b_mean_dtw:+.4f}  ({'better' if h_mean_dtw > b_mean_dtw else 'worse'})")
        print(f"  Median DTW sim delta:   {h_med_dtw - b_med_dtw:+.4f}  ({'better' if h_med_dtw > b_med_dtw else 'worse'})")
        print(f"  Wasserstein delta:      {h_wass - b_wass:+.4f}  ({'better' if h_wass < b_wass else 'worse'})")

        # ── 假设检验：Mann-Whitney U test ─────────────────────────────────────
        # 检验 biased 版本的 DTW similarity 分布是否显著高于 baseline
        # 使用 Mann-Whitney U（非参数检验，不假设正态分布），比 t-test 更适合这里
        # alternative="greater" 表示单侧检验：H1 = biased > baseline
        # p < 0.05 → 拒绝 H0，bias 显著提升了 DTW similarity
        print(f"\n{'='*50}")
        print("  Hypothesis Test (Mann-Whitney U)")
        print(f"{'='*50}")
        stat, p_value = mannwhitneyu(h_sims, b_sims, alternative="greater")
        print(f"  H0: biased DTW sim <= baseline DTW sim")
        print(f"  U statistic: {stat:.1f}")
        print(f"  p-value:     {p_value:.4f}")
        if p_value < 0.05:
            print(f"  Result:      SIGNIFICANT (p < 0.05) — bias improved human-likeness")
        else:
            print(f"  Result:      NOT significant (p >= 0.05)")

        # ── Per-task 分解：每个 task 的 DTW sim 变化量 ────────────────────────
        # 用于判断哪些难度类别的 task 从 human bias 中受益最多
        difficulty_map = load_difficulty()
        print(f"\n{'='*50}")
        print("  Per-task DTW sim (biased - baseline)")
        print(f"{'='*50}")
        common_tasks = set(b_per_task) & set(h_per_task)  # 两个版本都评估过的 task
        deltas = []
        for task_id in sorted(common_tasks):
            delta = h_per_task[task_id]["mean_dtw_sim"] - b_per_task[task_id]["mean_dtw_sim"]
            deltas.append(delta)
            cat = difficulty_map.get(task_id, "Unknown")
            print(f"  {task_id}  {delta:+.4f}  [{cat}]")
        print(f"\n  Mean per-task delta: {np.mean(deltas):+.4f}")
        print(f"  Tasks improved: {sum(d > 0 for d in deltas)}/{len(deltas)}")

        # ── 生成可视化图表 ────────────────────────────────────────────────────
        os.makedirs(PLOTS_DIR, exist_ok=True)
        print(f"\nGenerating plots -> {PLOTS_DIR}/")
        plot_dtw_distributions(b_sims, h_sims, PLOTS_DIR)
        plot_per_task_delta(b_per_task, h_per_task, difficulty_map, PLOTS_DIR)
        plot_mean_curves(human_curves, b_curves, h_curves, PLOTS_DIR)


# ── 辅助：加载难度分类表 ───────────────────────────────────────────────────────

def load_difficulty():
    """读取 task_difficulty.csv，返回 {task_id: difficulty_category} 字典。"""
    if os.path.exists(DIFFICULTY_CSV):
        df = pd.read_csv(DIFFICULTY_CSV)[["task_id", "difficulty_category"]]
        return dict(zip(df["task_id"], df["difficulty_category"]))
    return {}


# ── 可视化函数 ─────────────────────────────────────────────────────────────────

def plot_dtw_distributions(b_sims, h_sims, out_dir):
    """
    图 1：DTW similarity 分布对比直方图（baseline vs biased）。

    如果 biased 的分布整体向右移动（更高的 similarity），说明 bias 有效。
    竖虚线标注各自的均值，方便直观比较。
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 30)  # 0 到 1 之间 30 个 bin
    ax.hist(b_sims, bins=bins, alpha=0.6, label="Baseline (λ=0)",      color="tab:blue")
    ax.hist(h_sims, bins=bins, alpha=0.6, label="Human-biased (λ>0)",  color="tab:orange")
    # 均值线
    ax.axvline(np.mean(b_sims), color="tab:blue",   linestyle="--", linewidth=1.5)
    ax.axvline(np.mean(h_sims), color="tab:orange", linestyle="--", linewidth=1.5)
    ax.set_xlabel("DTW Similarity")
    ax.set_ylabel("Count")
    ax.set_title("DTW Similarity Distribution: Baseline vs Human-Biased")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "dtw_distribution.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved -> {path}")


def plot_per_task_delta(b_per_task, h_per_task, difficulty_map, out_dir):
    """
    图 2：每个 task 的 DTW similarity 变化量（biased - baseline）条形图，按难度类别着色。

    正值（条形向上）= biased 版本在该 task 上更像人类（bias 有效）。
    负值（条形向下）= biased 版本在该 task 上反而更不像人类（bias 有害）。
    颜色区分难度类别，便于观察哪类 task 最受益于 human bias。
    """
    common_tasks = sorted(set(b_per_task) & set(h_per_task))
    deltas     = [h_per_task[t]["mean_dtw_sim"] - b_per_task[t]["mean_dtw_sim"] for t in common_tasks]
    categories = [difficulty_map.get(t, "Unknown") for t in common_tasks]

    cat_colors = {
        "Easy for both":        "tab:green",
        "Only hard for AI":     "tab:orange",
        "Only hard for humans": "tab:blue",
        "Hard for both":        "tab:red",
        "Unknown":              "tab:gray",
    }
    colors = [cat_colors[c] for c in categories]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(common_tasks)), deltas, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)  # 零线：delta = 0
    ax.set_xticks(range(len(common_tasks)))
    ax.set_xticklabels(common_tasks, rotation=90, fontsize=7)
    ax.set_ylabel("DTW Similarity Delta (biased - baseline)")
    ax.set_title("Per-task DTW Similarity Improvement by Difficulty Category")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cat_colors.values()]
    ax.legend(handles, cat_colors.keys(), fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "per_task_delta.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved -> {path}")


def plot_mean_curves(human_curves, b_curves, h_curves, out_dir):
    """
    图 3：平均 progress curve 对比（human vs baseline vs biased）。

    将所有 human / baseline / biased 的 progress curves 分别取均值，
    画在同一张图上，直观展示 biased 版本的解题节奏是否更接近人类。

    理想情况：biased（橙线）比 baseline（灰虚线）更接近 human（蓝线）。
    """
    x          = np.linspace(0, 1, 100)
    # 把所有 task 的所有 human curves 展平，取逐点均值
    human_mean = np.mean([c for curves in human_curves.values() for c in curves], axis=0)
    b_mean     = np.mean(b_curves, axis=0) if b_curves else None
    h_mean     = np.mean(h_curves, axis=0) if h_curves else None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, human_mean, color="tab:blue",  label="Human",                  linewidth=2)
    if b_mean is not None:
        ax.plot(x, b_mean, color="tab:gray",   linestyle="--",
                label="CodeIt baseline (λ=0)", linewidth=2)
    if h_mean is not None:
        ax.plot(x, h_mean, color="tab:orange", label="CodeIt biased (λ>0)",   linewidth=2)
    ax.set_xlabel("Normalised step (0=start, 1=end)")
    ax.set_ylabel("Progress (fraction of cells correct)")
    ax.set_title("Mean Progress Curves: Human vs CodeIt")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "mean_progress_curves.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved -> {path}")


if __name__ == "__main__":
    main()
