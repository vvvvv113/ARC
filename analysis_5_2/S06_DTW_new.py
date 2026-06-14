"""
S06_DTW_new.py
DTW on raw variable-length normalized progress sequences (no resampling).

Key differences from S06_comparison_400.py:
  1. No resampling to 100 points — DTW operates on original variable-length sequences.
     Rationale: resampling introduces artificial data points between sparse observations;
     a 3-frame trace resampled to 100 points has 97 interpolated values with no empirical
     basis. DTW on raw sequences avoids this distortion.
  2. Sakoe-Chiba 10% window DTW.
     r=0.10 means the band half-width = max(round(0.1 * max(n,m)), |n-m|).
     The |n-m| floor guarantees the endpoint is always reachable for unequal-length
     sequences. Without it, a window smaller than |n-m| would make alignment infeasible.
     10% limits extreme warping (e.g. a 1-step CodeIt trace matching a 50-step human
     trace by stretching) while remaining generous enough not to penalise legitimate
     speed differences between human UI actions and DSL steps.
  3. Short trace filter: traces with < MIN_FRAMES unique grid frames are excluded before
     computing any group statistics. Short traces resampled to 100 points become flat lines
     that wash out the success/failure distinction in the group median.
  4. Medoid instead of element-wise median for group representative curves.
     Element-wise median requires equal-length sequences; the medoid is the trace with
     minimum sum of DTW distances to all others in the group, preserving temporal structure.

Four analysis pairs (same as S06):
  Pair 1: human_success  vs codeit_success
  Pair 2: human_failed   vs codeit_failed
  Pair 3: human_success  vs human_failed
  Pair 4: codeit_success vs codeit_failed

Outputs saved to analysis_5_2/processed/S06_DTW_new/
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

REPO             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURVES_JSON      = os.path.join(REPO, "analysis_5_2/processed/S04_curves/progress_curves_400.json")
HUMAN_TRACES_J   = os.path.join(REPO, "analysis_5_2/processed/S02_human_traces/human_traces_all.json")
CODEIT_TRACES_J  = os.path.join(REPO, "analysis_5_2/processed/S03_codeit_traces/codeit_traces_3seeds.json")
EVAL_DIR         = os.path.join(REPO, "codelt/data/evaluation")
OUT_DIR          = os.path.join(REPO, "analysis_5_2/processed/S06_DTW_new")
os.makedirs(OUT_DIR, exist_ok=True)

MIN_FRAMES = 5   # traces with fewer unique grid frames are excluded

# ── Sakoe-Chiba 10% window DTW ────────────────────────────────────────────────

DTW_RADIUS = 0.10   # band half-width as fraction of max(n, m)

def dtw_sakoe_chiba(a, b, r=DTW_RADIUS):
    """
    DTW with Sakoe-Chiba band of radius r (fraction of the longer sequence).

    window = max(round(r * max(n, m)), |n - m|)

    The |n-m| floor is mandatory: without it a window < |n-m| makes the
    (n-1, m-1) cell unreachable for unequal-length sequences, returning inf.
    Time O(n * window), space O(m).
    """
    n, m   = len(a), len(b)
    window = max(int(round(r * max(n, m))), abs(n - m))
    INF    = float("inf")

    prev = np.full(m, INF)
    prev[0] = abs(a[0] - b[0])
    for j in range(1, min(window + 1, m)):
        prev[j] = abs(a[0] - b[j]) + prev[j - 1]

    for i in range(1, n):
        curr  = np.full(m, INF)
        j_lo  = max(0, i - window)
        j_hi  = min(m - 1, i + window)
        for j in range(j_lo, j_hi + 1):
            cost   = abs(a[i] - b[j])
            left   = curr[j - 1] if j > j_lo else INF
            up     = prev[j]
            diag   = prev[j - 1] if j > 0 else INF
            curr[j] = cost + min(left, up, diag)
        prev = curr

    return float(prev[m - 1])

# ── BH 校正 ────────────────────────────────────────────────────────────────────

def bh_correction(pvals):
    n   = len(pvals)
    idx = np.argsort(pvals)
    sp  = np.array(pvals)[idx]
    adj = np.minimum.accumulate((sp * n / (np.arange(n) + 1))[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    result = np.empty(n)
    result[idx] = adj
    return result.tolist()

# ── 辅助：progress 计算 ────────────────────────────────────────────────────────

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

# ── medoid：变长序列的组代表 ───────────────────────────────────────────────────

def medoid(traces):
    """
    Return the trace with minimum sum of DTW distances to all others.
    For n=1 returns the single trace. For n=0 returns None.
    Time O(n^2 * L^2) where L = typical trace length.
    """
    if len(traces) == 0:
        return None
    if len(traces) == 1:
        return traces[0]
    n = len(traces)
    sums = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_sakoe_chiba(traces[i], traces[j])
            sums[i] += d
            sums[j] += d
    return traces[int(np.argmin(sums))]

# ── 加载数据 ───────────────────────────────────────────────────────────────────

print("Loading S04 baselines and group metadata...")
with open(CURVES_JSON) as f:
    curves_meta = json.load(f)
baselines       = {tid: td["baseline"]       for tid, td in curves_meta.items()}
baseline_groups = {tid: td["baseline_group"] for tid, td in curves_meta.items()}

print("Loading S02 human raw traces...")
with open(HUMAN_TRACES_J) as f:
    human_raw = json.load(f)

print("Loading S03 CodeIt raw traces...")
with open(CODEIT_TRACES_J) as f:
    codeit_raw = json.load(f)

print("Loading eval task target grids...")
task_target = {}  # task_id -> target_str
for fname in os.listdir(EVAL_DIR):
    if not fname.endswith(".json"):
        continue
    tid = fname.replace(".json", "")
    with open(os.path.join(EVAL_DIR, fname)) as f:
        task = json.load(f)
    try:
        rows = task["test_examples"][0]["output"]
        task_target[tid] = "|" + "|".join("".join(str(c) for c in row) for row in rows) + "|"
    except Exception:
        pass

rng = np.random.default_rng(42)

# ── 构建归一化原始轨迹（变长，已过滤短轨迹）──────────────────────────────────

def build_human_group(task_id, is_success_group):
    """
    Returns list of np.arrays (variable length), one per trace.
    Filters: trace must have >= MIN_FRAMES unique grid frames.
    """
    if task_id not in human_raw or task_id not in task_target:
        return []
    baseline = baselines.get(task_id, 0.0)
    if baseline >= 1.0:
        return []
    denom      = 1.0 - baseline
    target_str = task_target[task_id]
    out = []
    for t in human_raw[task_id]:
        if bool(t.get("success", False)) != is_success_group:
            continue
        grids = t["grids"]
        if len(grids) < MIN_FRAMES:
            continue
        prog = np.array([(progress_raw(g, target_str) - baseline) / denom for g in grids])
        out.append(prog)
    return out

def build_codeit_group(task_id, cls):
    """
    Returns list of np.arrays (variable length), one per trace.
    cls: "success" or "failed". Filters < MIN_FRAMES.
    Uses 3-seed combined traces from S03.
    """
    if task_id not in codeit_raw or task_id not in task_target:
        return []
    baseline = baselines.get(task_id, 0.0)
    if baseline >= 1.0:
        return []
    denom      = 1.0 - baseline
    target_str = task_target[task_id]
    out = []
    for t in codeit_raw[task_id]:
        if t.get("class") != cls:
            continue
        grids = t["grids"]
        if len(grids) < MIN_FRAMES:
            continue
        prog = np.array([(progress_raw(g, target_str) - baseline) / denom for g in grids])
        out.append(prog)
    return out

# group name -> (builder function args)
GROUP_BUILDERS = {
    "human_success":  lambda tid: build_human_group(tid, True),
    "human_failed":   lambda tid: build_human_group(tid, False),
    "codeit_success": lambda tid: build_codeit_group(tid, "success"),
    "codeit_failed":  lambda tid: build_codeit_group(tid, "failed"),
}

PAIRS = [
    ("pair1", "human_success",  "codeit_success"),
    ("pair2", "human_failed",   "codeit_failed"),
    ("pair3", "human_success",  "human_failed"),
    ("pair4", "codeit_success", "codeit_failed"),
]
BASELINE_GROUPS = ["A", "B"]

task_ids_sorted = sorted(curves_meta.keys())

# 预计算所有 (task, group) 的原始轨迹（避免重复构建）
print("\nBuilding raw normalized traces for all tasks and groups...")
all_traces = {}  # (task_id, group_name) -> list[np.array]
for task_id in tqdm(task_ids_sorted, desc="Building traces"):
    for gname, builder in GROUP_BUILDERS.items():
        all_traces[(task_id, gname)] = builder(task_id)

def get_raw(task_id, group):
    return all_traces.get((task_id, group), [])

# ══════════════════════════════════════════════════════════════════════════════
# 6.1  Medoid DTW（pairs 1–4, per task）
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 6.1 Medoid DTW (per task, per pair) ──")

medoid_rows = []
diag_empty  = 0  # tasks skipped due to empty group

for task_id in tqdm(task_ids_sorted, desc="Medoid DTW"):
    bg = baseline_groups.get(task_id, "?")
    for pname, g1, g2 in PAIRS:
        tr1 = get_raw(task_id, g1)
        tr2 = get_raw(task_id, g2)
        if len(tr1) == 0 or len(tr2) == 0:
            diag_empty += 1
            continue
        med1 = medoid(tr1)
        med2 = medoid(tr2)
        dtw_val = dtw_sakoe_chiba(med1, med2)
        medoid_rows.append({
            "task_id":        task_id,
            "pair":           pname,
            "group1":         g1,
            "group2":         g2,
            "baseline_group": bg,
            "dtw_medoid":     round(dtw_val, 4),
            "n1":             len(tr1),
            "n2":             len(tr2),
            "len_medoid1":    len(med1),
            "len_medoid2":    len(med2),
        })

med_df = pd.DataFrame(medoid_rows)
print(f"  Task-pair rows computed: {len(med_df)}  (skipped {diag_empty} empty-group pairs)")

# Wilcoxon test per (pair × baseline_group)
f2_tests = []
for pname, g1, g2 in PAIRS:
    for bg in BASELINE_GROUPS:
        sub = med_df[(med_df["pair"] == pname) & (med_df["baseline_group"] == bg)]["dtw_medoid"].dropna()
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
print(f"  Wilcoxon tests: {len(f2_tests)}")
print(f2_df[["pair", "baseline_group", "n", "median_DTW", "p_raw", "p_adj_BH"]].to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 6.2  Bootstrap CI on Medoid DTW（Pairs 1, 2: resample human, fix codeit medoid）
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 6.2 Bootstrap CI on Medoid DTW (B=500) ──")
B_BOOT = 500  # 500 is sufficient; increase to 1000 if needed

boot_rows = []

for pname, g_human, g_codeit in [
    ("pair1", "human_success", "codeit_success"),
    ("pair2", "human_failed",  "codeit_failed"),
]:
    for task_id in tqdm(task_ids_sorted, desc=f"Bootstrap {pname}"):
        bg       = baseline_groups.get(task_id, "?")
        tr_human = get_raw(task_id, g_human)
        tr_codeit = get_raw(task_id, g_codeit)
        if len(tr_human) < 5 or len(tr_codeit) == 0:
            continue
        fixed_medoid = medoid(tr_codeit)
        boot = []
        for _ in range(B_BOOT):
            idx       = rng.integers(0, len(tr_human), size=len(tr_human))
            resampled = [tr_human[i] for i in idx]
            med_h     = medoid(resampled)
            boot.append(dtw_sakoe_chiba(med_h, fixed_medoid))
        obs_dtw = dtw_sakoe_chiba(medoid(tr_human), fixed_medoid)
        boot_rows.append({
            "task_id":        task_id,
            "pair":           pname,
            "baseline_group": bg,
            "dtw_obs":        round(obs_dtw, 4),
            "ci_lo":          round(float(np.percentile(boot, 2.5)),  4),
            "ci_hi":          round(float(np.percentile(boot, 97.5)), 4),
            "n_human":        len(tr_human),
        })

boot_df = pd.DataFrame(boot_rows)
print(f"  Bootstrap CI rows: {len(boot_df)}")

# ══════════════════════════════════════════════════════════════════════════════
# 6.3  All-pairs DTW（Pairs 1, 2: each human trace vs each codeit trace）
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 6.3 All-pairs DTW (Pairs 1, 2) ──")

allpairs_rows = []

for pname, g_human, g_codeit in [
    ("pair1", "human_success", "codeit_success"),
    ("pair2", "human_failed",  "codeit_failed"),
]:
    for task_id in tqdm(task_ids_sorted, desc=f"All-pairs {pname}"):
        bg        = baseline_groups.get(task_id, "?")
        tr_human  = get_raw(task_id, g_human)
        tr_codeit = get_raw(task_id, g_codeit)
        if len(tr_human) == 0 or len(tr_codeit) == 0:
            continue
        for i, h in enumerate(tr_human):
            dists = [dtw_sakoe_chiba(h, c) for c in tr_codeit]
            allpairs_rows.append({
                "task_id":        task_id,
                "pair":           pname,
                "trace_idx":      i,
                "baseline_group": bg,
                "score_min":      round(float(min(dists)),           4),
                "score_median":   round(float(np.median(dists)),     4),
                "n_codeit":       len(tr_codeit),
                "len_human":      len(h),
            })

ap_df = pd.DataFrame(allpairs_rows)
print(f"  Total human traces processed: {len(ap_df)}")

# ══════════════════════════════════════════════════════════════════════════════
# 6.4  Permutation Test（Pair 3: human_success vs human_failed）
#
# Speed optimisation: pre-compute the full pairwise DTW distance matrix for
# each task's combined traces (success + failed) ONCE. Each permutation then
# only reindexes into this cached matrix — no DTW recomputation per permutation.
# Speedup factor ≈ N_perm / n_traces ≈ 5000 / 10 = 500×.
# ══════════════════════════════════════════════════════════════════════════════

print("\n── 6.4 Permutation Test (Pair 3, N_perm=5000) ──")
N_PERM = 5000

# 1. Collect eligible tasks and pre-compute distance matrices
perm_tasks = []   # (task_id, n_success, bg, D_matrix)
for task_id in tqdm(task_ids_sorted, desc="Pre-computing distance matrices"):
    bg   = baseline_groups.get(task_id, "?")
    tr_s = get_raw(task_id, "human_success")
    tr_f = get_raw(task_id, "human_failed")
    if len(tr_s) < 2 or len(tr_f) < 2:
        continue
    all_tr = tr_s + tr_f
    n      = len(all_tr)
    D      = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_sakoe_chiba(all_tr[i], all_tr[j])
            D[i, j] = d
            D[j, i] = d
    perm_tasks.append((task_id, all_tr, len(tr_s), bg, D))

print(f"  Tasks eligible: {len(perm_tasks)}")

def medoid_idx_from_matrix(D, indices):
    """Return the index (into D) of the medoid of the given subset indices."""
    sub = D[np.ix_(indices, indices)]
    return indices[int(np.argmin(sub.sum(axis=1)))]

# 2. Observed statistic
obs_per_task = []
for _, all_tr, n_s, _, D in perm_tasks:
    n     = len(all_tr)
    s_idx = np.arange(n_s)
    f_idx = np.arange(n_s, n)
    mi_s  = medoid_idx_from_matrix(D, s_idx)
    mi_f  = medoid_idx_from_matrix(D, f_idx)
    obs_per_task.append(D[mi_s, mi_f])
obs_stat = float(np.median(obs_per_task))
print(f"  Observed global median DTW: {obs_stat:.4f}")

# 3. Permutation loop — fast: only matrix index operations, no DTW recomputation
perm_stats = []
for _ in tqdm(range(N_PERM), desc="Permuting"):
    per_task_dtw = []
    for _, all_tr, n_s, _, D in perm_tasks:
        n       = len(all_tr)
        shuffled = rng.permutation(n)
        s_idx   = shuffled[:n_s]
        f_idx   = shuffled[n_s:]
        mi_s    = medoid_idx_from_matrix(D, s_idx)
        mi_f    = medoid_idx_from_matrix(D, f_idx)
        per_task_dtw.append(D[mi_s, mi_f])
    perm_stats.append(float(np.median(per_task_dtw)))

p_perm = float(np.mean(np.array(perm_stats) >= obs_stat))
print(f"  p_perm = {p_perm:.4f}  ({N_PERM} permutations)")

# ══════════════════════════════════════════════════════════════════════════════
# 保存 CSV
# ══════════════════════════════════════════════════════════════════════════════

med_df.to_csv(os.path.join(OUT_DIR, "medoid_dtw_per_task.csv"),   index=False)
boot_df.to_csv(os.path.join(OUT_DIR, "bootstrap_ci.csv"),          index=False)
ap_df.to_csv(os.path.join(OUT_DIR, "allpairs_dtw.csv"),            index=False)
f2_df.to_csv(os.path.join(OUT_DIR, "wilcoxon_tests.csv"),          index=False)

perm_result = pd.DataFrame([{
    "n_perm":    N_PERM,
    "obs_stat":  round(obs_stat, 4),
    "perm_mean": round(float(np.mean(perm_stats)), 4),
    "p_perm":    round(p_perm, 4),
    "n_tasks":   len(perm_tasks),
}])
perm_result.to_csv(os.path.join(OUT_DIR, "permutation_result.csv"), index=False)

print("\nAll CSVs saved.")

# ══════════════════════════════════════════════════════════════════════════════
# 绘图
# ══════════════════════════════════════════════════════════════════════════════

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

# 图1：Medoid DTW 分布（Group A / B）
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("S06_DTW_new: Medoid DTW (raw variable-length sequences, Sakoe-Chiba r=10%)",
             fontsize=11, fontweight="bold")
for ax_idx, bg in enumerate(BASELINE_GROUPS):
    ax  = axes[ax_idx]
    sub = med_df[med_df["baseline_group"] == bg]
    data, labels, colors = [], [], []
    for pname, _, _ in PAIRS:
        vals = sub[sub["pair"] == pname]["dtw_medoid"].dropna()
        if len(vals) < 2:
            continue
        data.append(vals.values)
        labels.append(PAIR_NAMES[pname] + f"\n(n={len(vals)})")
        colors.append(PAIR_COLORS[pname])
    if not data:
        continue
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("DTW distance (medoid vs medoid)")
    ax.set_title(f"Group {bg}")
    ax.grid(alpha=0.25, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "medoid_dtw_distribution.png"), dpi=130)
plt.close()

# 图2：All-pairs DTW 分布
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("S06_DTW_new: All-pairs DTW (each human trace vs each CodeIt trace)",
             fontsize=11, fontweight="bold")
for col_idx, agg in enumerate(["score_min", "score_median"]):
    agg_label = "Min (best-case)" if agg == "score_min" else "Median (typical)"
    for row_idx, (pname, _, _) in enumerate([
        ("pair1", None, None), ("pair2", None, None)
    ]):
        ax     = axes[row_idx][col_idx]
        sub    = ap_df[ap_df["pair"] == pname]
        data_A = sub[sub["baseline_group"] == "A"][agg].dropna().values
        data_B = sub[sub["baseline_group"] == "B"][agg].dropna().values
        if len(data_A) == 0 and len(data_B) == 0:
            continue
        bp = ax.boxplot([d for d in [data_A, data_B] if len(d) > 0],
                        patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=2))
        clrs = ["steelblue", "salmon"]
        for patch, c in zip(bp["boxes"], clrs):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        xticklabels = []
        if len(data_A) > 0: xticklabels.append(f"Group A\n(n={len(data_A)})")
        if len(data_B) > 0: xticklabels.append(f"Group B\n(n={len(data_B)})")
        ax.set_xticklabels(xticklabels)
        ax.set_ylabel("DTW score per human trace")
        ax.set_title(f"{PAIR_NAMES[pname].replace(chr(10),' ')}\n{agg_label}")
        ax.grid(alpha=0.25, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "allpairs_dtw_distribution.png"), dpi=130)
plt.close()

# 图3：Permutation test
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(perm_stats, bins=50, color="steelblue", alpha=0.8, edgecolor="white",
        label="Permuted global median DTW")
ax.axvline(obs_stat, color="firebrick", linewidth=2.5,
           label=f"Observed = {obs_stat:.4f}\np = {p_perm:.4f}")
ax.set_xlabel("Global median DTW(human_success medoid, human_failed medoid)")
ax.set_ylabel("Count")
ax.set_title(f"S06_DTW_new: Permutation Test — Pair 3\n"
             f"n_tasks={len(perm_tasks)}, N_perm={N_PERM}, p={p_perm:.4f}")
ax.legend(fontsize=9); ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "permutation_test.png"), dpi=130)
plt.close()

# 图4：trace 长度分布（诊断）
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("S06_DTW_new: Raw trace length distribution after MIN_FRAMES filter",
             fontsize=10, fontweight="bold")
for ax_idx, (gname, label) in enumerate([
    ("human_success", "Human success"), ("codeit_success", "CodeIt success")
]):
    lens = [len(t) for tid in task_ids_sorted for t in get_raw(tid, gname)]
    axes[ax_idx].hist(lens, bins=40, color="steelblue", alpha=0.8, edgecolor="white")
    axes[ax_idx].axvline(np.median(lens), color="firebrick", linewidth=1.5,
                         label=f"Median = {np.median(lens):.0f}")
    axes[ax_idx].set_xlabel("Trace length (frames)")
    axes[ax_idx].set_ylabel("Count")
    axes[ax_idx].set_title(label)
    axes[ax_idx].legend(fontsize=9)
    axes[ax_idx].grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "trace_length_distribution.png"), dpi=130)
plt.close()

# ── 汇总 ──────────────────────────────────────────────────────────────────────
print("\n=== Medoid DTW Wilcoxon Tests ===")
print(f2_df.to_string(index=False))
print(f"\n=== Permutation Test (Pair 3) ===")
print(f"  obs_stat={obs_stat:.4f}, p_perm={p_perm:.4f}, n_tasks={len(perm_tasks)}")

print(f"\n✓ S06_DTW_new complete.")
print(f"  Output dir: {OUT_DIR}/")
