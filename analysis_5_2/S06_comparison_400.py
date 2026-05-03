"""
S06_comparison_400.py
L2、Pearson r、Median DTW、All-pairs DTW、Permutation Test、Seed Sensitivity

四个分析对（pair）：
  Pair 1: human_success  vs codeit_success
  Pair 2: human_failed   vs codeit_failed
  Pair 3: human_success  vs human_failed
  Pair 4: codeit_success vs codeit_failed

FDR 校正 family 定义：
  F1（Pearson r 曲线相似性）：4 pair × 2 baseline_group = 8 个 Wilcoxon 检验
      每个任务计算两条 median curve 的 Pearson r（100 个时间点）；
      跨任务的 r 分布用 Wilcoxon signed-rank test 检验是否显著 > 0。
  F2（Median DTW 差异性）：4 pair × 2 baseline_group = 8 个 Wilcoxon 检验
      每个任务计算两条 median curve 的 DTW 距离；
      r 分布用 Wilcoxon test 检验是否显著 > 0。
  F3（Permutation test）：单个 global 检验，不需要 FDR（permutation test 本身已控制 Type I error）。

DTW 约束：Sakoe-Chiba window = 10
  DTW > L2 的任务比例作为诊断报告（不是 assertion）。

Bootstrap CI（Median DTW）：
  - Pairs 1、2：只 resample human traces，固定 CodeIt median curve
    语义：如果重新招募参与者，human median 与 CodeIt 典型轨迹的距离会在 [lo, hi] 之间
  - Pair 3：resample human_success 和 human_failed 各自独立（两组均来自人类）
  - Pair 4：无需 bootstrap（CodeIt 不是从参与者总体中抽样）
  - B = 1000；n < 5 时 CI = NaN

All-pairs DTW（Pairs 1、2）：
  对每条 human trace h_i 和每条 codeit trace c_j 计算 DTW(h_i, c_j)
  两种聚合：
    min  聚合：score(h_i) = min_j  DTW(h_i, c_j)  —— CodeIt 最佳匹配（乐观估计）
    median 聚合：score(h_i) = median_j DTW(h_i, c_j)  —— 典型匹配（保守估计）

Permutation Test（Pair 3：human_success vs human_failed）：
  test statistic：所有任务的 per-task DTW(median_success, median_failed) 的 median（全局统计量）
  N_perm = 5000：每次置换在各任务内部独立打乱 success/failed 标签，重算全局 median DTW
  p = (permuted global median DTW >= observed global median DTW 的次数) / N_perm
  单个 p 值 + 置换分布直方图

Seed Sensitivity Analysis（Pairs 1、2）：
  对 seed17、seed42、seed123 分别过滤 codeit traces，重算 all-pairs DTW（median 聚合）
  与 3-seed 合并结果对比（箱线图）
  若三个 seed 结论（方向 + 量级）一致，非独立性对结论影响有限。
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

# ── 路径配置 ───────────────────────────────────────────────────────────────────

REPO            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURVES_JSON     = os.path.join(REPO, "analysis_5_2/processed/S04_curves/progress_curves_400.json")
CODEIT_TRACES_J = os.path.join(REPO, "analysis_5_2/processed/S03_codeit_traces/codeit_traces_3seeds.json")
EVAL_DIR        = os.path.join(REPO, "codelt/data/evaluation")
OUT_DIR         = os.path.join(REPO, "analysis_5_2/processed/S06_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

# ── DTW 实现（Sakoe-Chiba window=10，纯 Python + numpy）───────────────────────

DTW_WINDOW = 10

def dtw_sc(a, b):
    """
    DTW distance with Sakoe-Chiba window=10.
    a, b: numpy arrays of length 100 (normalized progress curves).
    """
    n = len(a)
    prev = np.full(n, np.inf)
    # 第一行
    for j in range(min(DTW_WINDOW + 1, n)):
        prev[j] = abs(a[0] - b[j]) + (prev[j-1] if j > 0 else 0.0)

    for i in range(1, n):
        curr = np.full(n, np.inf)
        j_lo = max(0, i - DTW_WINDOW)
        j_hi = min(n, i + DTW_WINDOW + 1)
        for j in range(j_lo, j_hi):
            cost = abs(a[i] - b[j])
            best = prev[j]
            if j > 0:
                v = curr[j-1]
                if v < best: best = v
                v = prev[j-1]
                if v < best: best = v
            curr[j] = cost + best
        prev = curr
    return prev[n - 1]

# ── BH 校正 ────────────────────────────────────────────────────────────────────

def bh_correction(pvals):
    n  = len(pvals)
    idx = np.argsort(pvals)
    sp  = np.array(pvals)[idx]
    adj = np.minimum.accumulate((sp * n / (np.arange(n) + 1))[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    result = np.empty(n)
    result[idx] = adj
    return result.tolist()

# ── 辅助：从 progress_curves_400.json 提取 median / individual traces ──────────

def get_median(task_data, group):
    if group not in task_data:
        return None
    return np.array(task_data[group]["median"], dtype=float)

def get_traces(task_data, group):
    if group not in task_data:
        return None
    return np.array(task_data[group]["traces"], dtype=float)  # shape (n, 100)

# ── 加载数据 ───────────────────────────────────────────────────────────────────

print("Loading progress curves JSON...")
with open(CURVES_JSON) as f:
    curves = json.load(f)
print(f"  Tasks: {len(curves)}")

print("Loading CodeIt traces JSON (for seed sensitivity)...")
with open(CODEIT_TRACES_J) as f:
    codeit_raw = json.load(f)

# 加载 target/input grids（seed sensitivity 需要重算 progress）
print("Loading eval task grids...")
task_grids = {}   # task_id -> {input_str, target_str}
def _grid_to_str(grid):
    return "|" + "|".join("".join(str(c) for c in row) for row in grid) + "|"
for fname in os.listdir(EVAL_DIR):
    if not fname.endswith(".json"): continue
    tid = fname.replace(".json", "")
    with open(os.path.join(EVAL_DIR, fname)) as f:
        task = json.load(f)
    try:
        task_grids[tid] = {
            "input_str":  _grid_to_str(task["test_examples"][0]["input"]),
            "target_str": _grid_to_str(task["test_examples"][0]["output"]),
        }
    except Exception:
        pass

x100 = np.linspace(0, 1, 100)
rng  = np.random.default_rng(42)

PAIRS = [
    ("pair1", "human_success",  "codeit_success"),
    ("pair2", "human_failed",   "codeit_failed"),
    ("pair3", "human_success",  "human_failed"),
    ("pair4", "codeit_success", "codeit_failed"),
]
BASELINE_GROUPS = ["A", "B"]

# ══════════════════════════════════════════════════════════════════════════════
# 6.1  L2 距离 和 Pearson r（per task per pair）
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 6.1 L2 and Pearson r ──")

l2_pearson_rows = []

for task_id, task_data in curves.items():
    bg = task_data["baseline_group"]
    for pname, g1, g2 in PAIRS:
        m1 = get_median(task_data, g1)
        m2 = get_median(task_data, g2)
        if m1 is None or m2 is None:
            continue
        l2  = float(np.sqrt(np.sum((m1 - m2) ** 2)))
        r, p = stats.pearsonr(m1, m2)
        l2_pearson_rows.append({
            "task_id": task_id, "pair": pname, "group1": g1, "group2": g2,
            "baseline_group": bg, "L2": round(l2, 4), "pearson_r": round(float(r), 4),
            "pearson_p": round(float(p), 6),
        })

lp_df = pd.DataFrame(l2_pearson_rows)

# Wilcoxon test per (pair × baseline_group)：pearson_r 是否显著 > 0
f1_tests, f1_labels = [], []
for pname, g1, g2 in PAIRS:
    for bg in BASELINE_GROUPS:
        sub = lp_df[(lp_df["pair"] == pname) & (lp_df["baseline_group"] == bg)]["pearson_r"].dropna()
        if len(sub) < 5:
            continue
        stat, p = stats.wilcoxon(sub, alternative="greater")
        f1_tests.append({"pair": pname, "baseline_group": bg, "n": len(sub),
                          "median_r": round(sub.median(), 4), "p_raw": round(p, 6)})
        f1_labels.append(f"{pname}_{bg}")

f1_pvals = [r["p_raw"] for r in f1_tests]
f1_adj   = bh_correction(f1_pvals)
for i, r in enumerate(f1_tests):
    r["p_adj_BH"] = round(f1_adj[i], 6)

f1_df = pd.DataFrame(f1_tests)
print(f"  F1 Wilcoxon tests: {len(f1_tests)}")
print(f1_df[["pair","baseline_group","n","median_r","p_raw","p_adj_BH"]].to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 6.2  Median DTW + Bootstrap CI（B=1000）
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 6.2 Median DTW + Bootstrap CI (B=1000) ──")
B_DTW = 1000

dtw_rows    = []
diag_dtw_gt_l2 = 0  # 诊断：DTW > L2 的任务数

task_ids_sorted = sorted(curves.keys())

for task_id in tqdm(task_ids_sorted, desc="Median DTW"):
    task_data = curves[task_id]
    bg = task_data["baseline_group"]

    for pname, g1, g2 in PAIRS:
        m1 = get_median(task_data, g1)
        m2 = get_median(task_data, g2)
        if m1 is None or m2 is None:
            continue

        dtw_obs = dtw_sc(m1, m2)
        l2_obs  = float(np.sqrt(np.sum((m1 - m2) ** 2)))
        if dtw_obs > l2_obs:
            diag_dtw_gt_l2 += 1

        # Bootstrap CI（B=1000）
        ci_lo, ci_hi = float("nan"), float("nan")

        if pname in ("pair1", "pair2"):
            # 只 resample human traces，固定 CodeIt median
            tr_human = get_traces(task_data, g1)  # g1 is always human group
            if tr_human is not None and len(tr_human) >= 5:
                boot = []
                for _ in range(B_DTW):
                    idx = rng.integers(0, len(tr_human), size=len(tr_human))
                    med_h = np.median(tr_human[idx], axis=0)
                    boot.append(dtw_sc(med_h, m2))
                ci_lo = float(np.percentile(boot, 2.5))
                ci_hi = float(np.percentile(boot, 97.5))

        elif pname == "pair3":
            # human_success vs human_failed：resample 两组各自独立
            tr1 = get_traces(task_data, g1)
            tr2 = get_traces(task_data, g2)
            if tr1 is not None and tr2 is not None and len(tr1) >= 5 and len(tr2) >= 5:
                boot = []
                for _ in range(B_DTW):
                    i1 = rng.integers(0, len(tr1), size=len(tr1))
                    i2 = rng.integers(0, len(tr2), size=len(tr2))
                    boot.append(dtw_sc(np.median(tr1[i1], axis=0),
                                       np.median(tr2[i2], axis=0)))
                ci_lo = float(np.percentile(boot, 2.5))
                ci_hi = float(np.percentile(boot, 97.5))
        # pair4：无 bootstrap

        dtw_rows.append({
            "task_id": task_id, "pair": pname, "group1": g1, "group2": g2,
            "baseline_group": bg, "dtw_median_curves": round(dtw_obs, 4),
            "dtw_ci_lo": round(ci_lo, 4) if not np.isnan(ci_lo) else float("nan"),
            "dtw_ci_hi": round(ci_hi, 4) if not np.isnan(ci_hi) else float("nan"),
            "l2_median_curves": round(l2_obs, 4),
        })

dtw_df = pd.DataFrame(dtw_rows)

# 诊断报告：DTW > L2
total_pairs = len(dtw_df)
pct_dtw_gt_l2 = diag_dtw_gt_l2 / total_pairs * 100
print(f"\n  [Diagnostic] DTW > L2: {diag_dtw_gt_l2}/{total_pairs} ({pct_dtw_gt_l2:.1f}%)")
if pct_dtw_gt_l2 > 10:
    print("  WARNING: >10% of pairs have DTW > L2, consider adjusting window parameter.")
else:
    print("  OK: within expected range (window=10 constraint is not too tight).")

# Wilcoxon test per (pair × baseline_group)：DTW 是否显著 > 0（F2 family）
f2_tests = []
for pname, g1, g2 in PAIRS:
    for bg in BASELINE_GROUPS:
        sub = dtw_df[(dtw_df["pair"] == pname) & (dtw_df["baseline_group"] == bg)]["dtw_median_curves"].dropna()
        if len(sub) < 5:
            continue
        stat, p = stats.wilcoxon(sub, alternative="greater")
        f2_tests.append({"pair": pname, "baseline_group": bg, "n": len(sub),
                          "median_DTW": round(sub.median(), 4), "p_raw": round(p, 6)})

f2_pvals = [r["p_raw"] for r in f2_tests]
f2_adj   = bh_correction(f2_pvals)
for i, r in enumerate(f2_tests):
    r["p_adj_BH"] = round(f2_adj[i], 6)

f2_df = pd.DataFrame(f2_tests)
print(f"\n  F2 Wilcoxon tests: {len(f2_tests)}")
print(f2_df[["pair","baseline_group","n","median_DTW","p_raw","p_adj_BH"]].to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 6.3  All-pairs DTW（Pairs 1、2；min & median aggregation）
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 6.3 All-pairs DTW (Pairs 1, 2) ──")

allpairs_rows = []

for pname, g_human, g_codeit in [("pair1","human_success","codeit_success"),
                                   ("pair2","human_failed","codeit_failed")]:
    for task_id in tqdm(task_ids_sorted, desc=f"All-pairs {pname}"):
        task_data = curves[task_id]
        bg = task_data["baseline_group"]
        tr_h = get_traces(task_data, g_human)
        tr_c = get_traces(task_data, g_codeit)
        if tr_h is None or tr_c is None:
            continue

        n_h, n_c = len(tr_h), len(tr_c)
        for i in range(n_h):
            dists = [dtw_sc(tr_h[i], tr_c[j]) for j in range(n_c)]
            allpairs_rows.append({
                "task_id": task_id, "pair": pname, "trace_idx": i,
                "baseline_group": bg,
                "score_min":    round(float(min(dists)), 4),
                "score_median": round(float(np.median(dists)), 4),
                "n_codeit":     n_c,
            })

ap_df = pd.DataFrame(allpairs_rows)
print(f"  Total human traces processed: {len(ap_df)}")

# ══════════════════════════════════════════════════════════════════════════════
# 6.4  Permutation Test（Pair 3：human_success vs human_failed）
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 6.4 Permutation Test (Pair 3, N_perm=5000) ──")
N_PERM = 5000

# 收集所有同时有 human_success 和 human_failed 的任务
perm_tasks = []
for task_id in task_ids_sorted:
    td = curves[task_id]
    tr_s = get_traces(td, "human_success")
    tr_f = get_traces(td, "human_failed")
    if tr_s is None or tr_f is None:
        continue
    if len(tr_s) < 2 or len(tr_f) < 2:
        continue
    perm_tasks.append((task_id, tr_s, tr_f, td["baseline_group"]))

print(f"  Tasks eligible for permutation test: {len(perm_tasks)}")

# Observed global statistic：所有任务的 per-task DTW(median_success, median_failed) 的中位数
obs_per_task = []
for task_id, tr_s, tr_f, bg in perm_tasks:
    dtw_t = dtw_sc(np.median(tr_s, axis=0), np.median(tr_f, axis=0))
    obs_per_task.append(dtw_t)
obs_stat = float(np.median(obs_per_task))
print(f"  Observed global median DTW: {obs_stat:.4f}")

# 5000 次置换：在每个任务内部独立打乱标签，重算全局中位数 DTW
perm_stats = []
for _ in tqdm(range(N_PERM), desc="Permuting"):
    per_task_dtw = []
    for task_id, tr_s, tr_f, bg in perm_tasks:
        all_traces = np.vstack([tr_s, tr_f])
        n_s = len(tr_s)
        shuffled = rng.permutation(len(all_traces))
        new_s = all_traces[shuffled[:n_s]]
        new_f = all_traces[shuffled[n_s:]]
        d = dtw_sc(np.median(new_s, axis=0), np.median(new_f, axis=0))
        per_task_dtw.append(d)
    perm_stats.append(float(np.median(per_task_dtw)))

p_perm = float(np.mean(np.array(perm_stats) >= obs_stat))
print(f"  p_perm = {p_perm:.4f}  (fraction of {N_PERM} permutations >= observed)")

# ══════════════════════════════════════════════════════════════════════════════
# 6.5  Seed Sensitivity Analysis（Pairs 1、2）
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 6.5 Seed Sensitivity Analysis ──")

def _parse_grid(s):
    rows = s.strip("|").split("|")
    return [[int(c) for c in row] for row in rows]

def progress_raw(g_str, t_str):
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

def resample_to_100(values):
    if len(values) < 2:
        return [float(values[0])] * 100
    xo = np.linspace(0, 1, len(values))
    xn = np.linspace(0, 1, 100)
    return np.interp(xn, xo, np.array(values, dtype=float))

def build_codeit_curves_for_seed(seed_name, cls):
    """
    从 S03 JSON 中过滤 seed_name 的 cls（"success"/"failed"）traces，
    计算 normalized + resampled 曲线，返回 {task_id: np.array shape(n,100)}。
    """
    result = {}
    for task_id, traces in codeit_raw.items():
        if task_id not in curves or task_id not in task_grids:
            continue
        td = curves[task_id]
        baseline = td["baseline"]
        if baseline >= 1.0:
            continue
        denom = 1.0 - baseline
        tgt   = task_grids[task_id]["target_str"]
        inp   = task_grids[task_id]["input_str"]

        filtered = [tr for tr in traces if tr["class"] == cls and tr["seed"] == seed_name]
        if not filtered:
            continue
        resampled = []
        for tr in filtered:
            grids = tr["grids"]
            raw_prog  = [progress_raw(g, tgt) for g in grids]
            norm_prog = [(p - baseline) / denom for p in raw_prog]
            resampled.append(resample_to_100(norm_prog))
        result[task_id] = np.array(resampled, dtype=float)
    return result

SEEDS = ["seed17", "seed42", "seed123", "combined"]

sensitivity_rows = []

for pname, g_human, g_codeit, codeit_cls in [
    ("pair1", "human_success", "codeit_success", "success"),
    ("pair2", "human_failed",  "codeit_failed",  "failed"),
]:
    print(f"  Seed sensitivity for {pname}...")

    # 组合（combined）的 codeit median 来自 progress_curves_400.json
    for seed_name in tqdm(SEEDS, desc=pname):
        if seed_name == "combined":
            # 使用 S04 已计算的结果
            for task_id in task_ids_sorted:
                td = curves[task_id]
                bg = td["baseline_group"]
                tr_h = get_traces(td, g_human)
                m_c  = get_median(td, g_codeit)
                if tr_h is None or m_c is None:
                    continue
                dtw_val = dtw_sc(np.median(tr_h, axis=0), m_c)
                # all-pairs median聚合
                ap_scores = [dtw_sc(tr_h[i], m_c) for i in range(len(tr_h))]
                sensitivity_rows.append({
                    "pair": pname, "seed": seed_name, "task_id": task_id,
                    "baseline_group": bg,
                    "median_dtw_medians": round(dtw_val, 4),
                    "allpairs_median": round(float(np.median(ap_scores)), 4),
                })
        else:
            seed_curves = build_codeit_curves_for_seed(seed_name, codeit_cls)
            for task_id in task_ids_sorted:
                if task_id not in seed_curves:
                    continue
                td = curves[task_id]
                bg = td["baseline_group"]
                tr_h = get_traces(td, g_human)
                if tr_h is None:
                    continue
                tr_c_seed = seed_curves[task_id]
                if len(tr_c_seed) == 0:
                    continue
                m_c_seed = np.median(tr_c_seed, axis=0)
                dtw_val  = dtw_sc(np.median(tr_h, axis=0), m_c_seed)
                ap_scores = []
                for i in range(len(tr_h)):
                    ap_scores.append(dtw_sc(tr_h[i], m_c_seed))
                sensitivity_rows.append({
                    "pair": pname, "seed": seed_name, "task_id": task_id,
                    "baseline_group": bg,
                    "median_dtw_medians": round(dtw_val, 4),
                    "allpairs_median": round(float(np.median(ap_scores)), 4),
                })

sens_df = pd.DataFrame(sensitivity_rows)

# ══════════════════════════════════════════════════════════════════════════════
# 保存所有 CSV
# ══════════════════════════════════════════════════════════════════════════════

lp_df.to_csv(os.path.join(OUT_DIR, "l2_pearson_per_task.csv"), index=False)
dtw_df.to_csv(os.path.join(OUT_DIR, "dtw_median_per_task.csv"), index=False)
ap_df.to_csv(os.path.join(OUT_DIR, "allpairs_dtw.csv"), index=False)
f1_df.to_csv(os.path.join(OUT_DIR, "f1_pearson_wilcoxon.csv"), index=False)
f2_df.to_csv(os.path.join(OUT_DIR, "f2_dtw_wilcoxon.csv"), index=False)
sens_df.to_csv(os.path.join(OUT_DIR, "sensitivity_seed.csv"), index=False)

perm_result = pd.DataFrame([{
    "n_perm": N_PERM, "obs_stat": round(obs_stat, 4),
    "perm_mean": round(float(np.mean(perm_stats)), 4),
    "p_perm": round(p_perm, 4),
    "n_tasks": len(perm_tasks),
}])
perm_result.to_csv(os.path.join(OUT_DIR, "permutation_result.csv"), index=False)

print("\nAll CSVs saved.")

# ══════════════════════════════════════════════════════════════════════════════
# 绘图
# ══════════════════════════════════════════════════════════════════════════════

print("\nGenerating plots...")

PAIR_COLORS = {
    "pair1": "steelblue", "pair2": "tomato",
    "pair3": "mediumseagreen", "pair4": "mediumpurple",
}
PAIR_NAMES = {
    "pair1": "Human success\nvs CodeIt success",
    "pair2": "Human failed\nvs CodeIt failed",
    "pair3": "Human success\nvs Human failed",
    "pair4": "CodeIt success\nvs CodeIt failed",
}

# ── 图1：L2 分布箱线图（分 Group A / B）──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("S06: L2 Distance between Median Curves (per task)", fontsize=11, fontweight="bold")
for ax_idx, bg in enumerate(BASELINE_GROUPS):
    ax  = axes[ax_idx]
    sub = lp_df[lp_df["baseline_group"] == bg]
    data, labels, colors = [], [], []
    for pname, _, _ in PAIRS:
        vals = sub[sub["pair"] == pname]["L2"].dropna()
        if len(vals) < 2: continue
        data.append(vals.values)
        labels.append(PAIR_NAMES[pname] + f"\n(n={len(vals)})")
        colors.append(PAIR_COLORS[pname])
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("L2 distance"); ax.set_title(f"Group {bg}")
    ax.grid(alpha=0.25, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "l2_distribution.png"), dpi=130)
plt.close()

# ── 图2：Median DTW 分布箱线图 + CI（分 Group A / B）────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("S06: Median DTW between Median Curves (per task, Sakoe-Chiba w=10)", fontsize=11, fontweight="bold")
for ax_idx, bg in enumerate(BASELINE_GROUPS):
    ax  = axes[ax_idx]
    sub = dtw_df[dtw_df["baseline_group"] == bg]
    data, labels, colors = [], [], []
    for pname, _, _ in PAIRS:
        vals = sub[sub["pair"] == pname]["dtw_median_curves"].dropna()
        if len(vals) < 2: continue
        data.append(vals.values)
        labels.append(PAIR_NAMES[pname] + f"\n(n={len(vals)})")
        colors.append(PAIR_COLORS[pname])
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("DTW distance"); ax.set_title(f"Group {bg}")
    ax.grid(alpha=0.25, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "dtw_distribution.png"), dpi=130)
plt.close()

# ── 图3：DTW vs L2 诊断图 ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
for pname, _, _ in PAIRS:
    sub = dtw_df[dtw_df["pair"] == pname]
    ax.scatter(sub["l2_median_curves"], sub["dtw_median_curves"],
               alpha=0.25, s=15, label=PAIR_NAMES[pname].replace("\n", " "),
               color=PAIR_COLORS[pname])
lim = [0, max(dtw_df["l2_median_curves"].max(), dtw_df["dtw_median_curves"].max()) * 1.05]
ax.plot(lim, lim, "k--", linewidth=1, label="DTW = L2")
ax.set_xlabel("L2 distance"); ax.set_ylabel("DTW distance")
ax.set_title(f"DTW vs L2 Diagnostic\n(DTW > L2: {pct_dtw_gt_l2:.1f}% of pairs)")
ax.legend(fontsize=7); ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "dtw_vs_l2_diagnostic.png"), dpi=130)
plt.close()

# ── 图4：All-pairs DTW 分布（stripplot style，min & median aggregation）──────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("S06: All-pairs DTW Distribution (human trace vs CodeIt traces)",
             fontsize=11, fontweight="bold")
for col_idx, agg in enumerate(["score_min", "score_median"]):
    agg_label = "Min aggregation (best-case)" if agg == "score_min" else "Median aggregation (typical)"
    for row_idx, (pname, g_h, g_c) in enumerate([("pair1","human_success","codeit_success"),
                                                   ("pair2","human_failed","codeit_failed")]):
        ax  = axes[row_idx][col_idx]
        sub = ap_df[ap_df["pair"] == pname]
        data_A = sub[sub["baseline_group"] == "A"][agg].dropna().values
        data_B = sub[sub["baseline_group"] == "B"][agg].dropna().values
        bp = ax.boxplot([data_A, data_B], patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=2))
        bp["boxes"][0].set_facecolor("steelblue"); bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor("salmon");    bp["boxes"][1].set_alpha(0.6)
        ax.set_xticklabels([f"Group A\n(n={len(data_A)})", f"Group B\n(n={len(data_B)})"])
        ax.set_ylabel("DTW score per human trace")
        ax.set_title(f"{PAIR_NAMES[pname].replace(chr(10),' ')}\n{agg_label}")
        ax.grid(alpha=0.25, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "allpairs_dtw_distribution.png"), dpi=130)
plt.close()

# ── 图5：Permutation Test 直方图 ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(perm_stats, bins=50, color="steelblue", alpha=0.8, edgecolor="white",
        label="Permuted global median DTW")
ax.axvline(obs_stat, color="firebrick", linewidth=2.5,
           label=f"Observed = {obs_stat:.4f}\np = {p_perm:.4f}")
ax.set_xlabel("Global median DTW(human_success, human_failed)")
ax.set_ylabel("Count")
ax.set_title(f"S06: Permutation Test — Pair 3 (N_perm={N_PERM})\n"
             f"n_tasks={len(perm_tasks)}, p_perm={p_perm:.4f}")
ax.legend(fontsize=9); ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "permutation_test.png"), dpi=130)
plt.close()

# ── 图6：Seed Sensitivity 箱线图 ──────────────────────────────────────────────
for pname in ["pair1", "pair2"]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"S06: Seed Sensitivity — {PAIR_NAMES[pname].replace(chr(10),' ')} (median aggregation)",
                 fontsize=10, fontweight="bold")
    for ax_idx, bg in enumerate(BASELINE_GROUPS):
        ax = axes[ax_idx]
        sub = sens_df[(sens_df["pair"] == pname) & (sens_df["baseline_group"] == bg)]
        data, labels = [], []
        for seed in SEEDS:
            vals = sub[sub["seed"] == seed]["allpairs_median"].dropna().values
            if len(vals) < 2: continue
            data.append(vals)
            labels.append(f"{seed}\n(n={len(vals)})")
        if data:
            bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                            medianprops=dict(color="black", linewidth=2))
            seed_colors = ["steelblue","tomato","mediumseagreen","mediumpurple"]
            for patch, c in zip(bp["boxes"], seed_colors[:len(bp["boxes"])]):
                patch.set_facecolor(c); patch.set_alpha(0.6)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel("All-pairs median DTW per human trace")
            ax.set_title(f"Group {bg}")
            ax.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"sensitivity_seed_{pname}.png"), dpi=130)
    plt.close()

# ── 汇总打印 ──────────────────────────────────────────────────────────────────
print("\n=== F1 (Pearson r Wilcoxon) ===")
print(f1_df.to_string(index=False))
print("\n=== F2 (Median DTW Wilcoxon) ===")
print(f2_df.to_string(index=False))
print(f"\n=== Permutation Test (Pair 3) ===")
print(f"  obs_stat={obs_stat:.4f}, p_perm={p_perm:.4f}, N_perm={N_PERM}, n_tasks={len(perm_tasks)}")
print(f"\n[Diagnostic] DTW > L2: {pct_dtw_gt_l2:.1f}% of task-pair combinations")

print("\n✓ S06 complete.")
print(f"  Output dir: {OUT_DIR}/")
