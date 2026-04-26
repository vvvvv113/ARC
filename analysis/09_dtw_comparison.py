"""
Script 09 — DTW Curve Comparison

Extends script 08 by replacing point-wise L2 distance with Dynamic Time Warping (DTW),
which finds the optimal many-to-one temporal alignment between two progress curves.

Key metric: DTW/L2 ratio ∈ [0, 1]
  - Ratio ≈ 1 → temporal warping does not help → agents differ in level/capability
  - Ratio ≪ 1 → large L2 collapses under alignment → agents share the same shape but
    operate at different paces ("strategy-equivalent, pace-different")

DTW implementation: manual DP with Sakoe-Chiba band (window = 10 points = 10% of N=100).
Bootstrap CIs (B=200): resample individual normalised trajectories → bootstrap median →
DTW; quantifies instability from small human sample sizes. Skipped for
codeit_success_vs_codeit_failed (both groups have large stable n).

Input:
  analysis/processed/06_curves/progress_curves_v2.json
  analysis/processed/04_human_traces/human_traces.json
  analysis/processed/05_codeit_traces/codeit_traces.json
  analysis/processed/01_difficulty/task_difficulty.csv
  codelt/data/evaluation/{task_id}.json

Outputs (analysis/processed/09_dtw/):
  dtw_per_task.csv       — per-task long format: task_id, pair, dtw_distance,
                            dtw_ci_lo, dtw_ci_hi, l2_distance, dtw_l2_ratio,
                            difficulty_category, n_a, n_b
  dtw_summary.csv        — median DTW + DTW/L2 grouped by pair × difficulty_category
  dtw_stripplot.png      — per-pair strip plot (same format as script 08)
  dtw_vs_l2_scatter.png  — DTW vs L2 per (task, pair); points on or below y=x
  warp_path/             — 2-panel warping-path plots for selected tasks
"""

import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

REPO        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURVES_PATH = os.path.join(REPO, "analysis/processed/06_curves/progress_curves_v2.json")
HUMAN_PATH  = os.path.join(REPO, "analysis/processed/04_human_traces/human_traces.json")
CODEIT_PATH = os.path.join(REPO, "analysis/processed/05_codeit_traces/codeit_traces.json")
DIFF_PATH   = os.path.join(REPO, "analysis/processed/01_difficulty/task_difficulty.csv")
EVAL_DIR    = os.path.join(REPO, "codelt/data/evaluation")
OUT_DIR     = os.path.join(REPO, "analysis/processed/09_dtw")
WARP_DIR    = os.path.join(OUT_DIR, "warp_path")

os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(WARP_DIR, exist_ok=True)

N_POINTS   = 100
DTW_WINDOW = 10   # Sakoe-Chiba band: ±10 time steps (10% of N=100)
BOOT_B     = 200  # bootstrap resamples for CI

PAIRS = {
    "human_success_vs_codeit_success": ("human_success", "codeit_success"),
    "human_failed_vs_codeit_failed":   ("human_failed",  "codeit_failed"),
    "human_success_vs_human_failed":   ("human_success", "human_failed"),
    "codeit_success_vs_codeit_failed": ("codeit_success","codeit_failed"),
}
# Bootstrap skipped for this pair (both groups have large stable n)
SKIP_BOOTSTRAP = {"codeit_success_vs_codeit_failed"}

PAIR_LABELS = {
    "human_success_vs_codeit_success": "Human succ\nvs CodeIt succ",
    "human_failed_vs_codeit_failed":   "Human fail\nvs CodeIt fail",
    "human_success_vs_human_failed":   "Human succ\nvs Human fail",
    "codeit_success_vs_codeit_failed": "CodeIt succ\nvs CodeIt fail",
}
CAT_COLORS = {
    "Easy for both":        "tab:green",
    "Hard for both":        "tab:red",
    "Only hard for AI":     "tab:blue",
    "Only hard for humans": "tab:orange",
}

# ── DTW helpers ────────────────────────────────────────────────────────────────

def dtw_and_matrix(a, b, window=DTW_WINDOW):
    """
    DTW distance + accumulated cost matrix D with Sakoe-Chiba band.
    Uses anti-diagonal numpy vectorisation: O(N) iterations of O(N/band) work.
    DTW distance ≤ L2 distance by construction (L2 is the zero-warping special case).
    """
    n = len(a)
    cost = (np.asarray(a, dtype=float)[:, None] -
            np.asarray(b, dtype=float)[None, :]) ** 2   # (n, n)
    D = np.full((n, n), np.inf)

    D[0, 0] = cost[0, 0]
    for j in range(1, min(window + 1, n)):
        D[0, j] = D[0, j - 1] + cost[0, j]
    for i in range(1, min(window + 1, n)):
        D[i, 0] = D[i - 1, 0] + cost[i, 0]

    for k in range(2, 2 * n - 1):
        i_lo = max(1, k - n + 1)
        i_hi = min(n, k)
        i_arr = np.arange(i_lo, i_hi)
        j_arr = k - i_arr
        mask  = np.abs(i_arr - j_arr) <= window
        i_arr, j_arr = i_arr[mask], j_arr[mask]
        if len(i_arr) == 0:
            continue
        prev = np.minimum(np.minimum(D[i_arr - 1, j_arr],
                                     D[i_arr,     j_arr - 1]),
                          D[i_arr - 1, j_arr - 1])
        D[i_arr, j_arr] = cost[i_arr, j_arr] + prev

    return float(np.sqrt(D[-1, -1])), D


def dtw_fast(a, b, window=DTW_WINDOW):
    """DTW distance only (no matrix storage) — used for bootstrap inner loop."""
    d, _ = dtw_and_matrix(a, b, window)
    return d


def dtw_path(D):
    """Traceback the optimal warping path from the accumulated cost matrix D."""
    i, j = D.shape[0] - 1, D.shape[1] - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            step = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
            if step == 0:
                i -= 1; j -= 1
            elif step == 1:
                i -= 1
            else:
                j -= 1
        path.append((i, j))
    return list(reversed(path))


def compute_bootstrap_ci(curves_a, curves_b, B=BOOT_B, rng=None):
    """
    Bootstrap CI for DTW(median_A, median_B) by resampling individual trajectories.
    Returns (ci_lo, ci_hi) at 95% level, or (nan, nan) if either group has n < 3.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n_a, n_b = len(curves_a), len(curves_b)
    if min(n_a, n_b) < 3:
        return float("nan"), float("nan")
    arr_a = np.array(curves_a)
    arr_b = np.array(curves_b)
    boot_dtw = []
    for _ in range(B):
        idx_a = rng.integers(0, n_a, size=n_a)
        idx_b = rng.integers(0, n_b, size=n_b)
        med_a = np.median(arr_a[idx_a], axis=0)
        med_b = np.median(arr_b[idx_b], axis=0)
        boot_dtw.append(dtw_fast(med_a, med_b))
    return float(np.percentile(boot_dtw, 2.5)), float(np.percentile(boot_dtw, 97.5))

# ── grid helpers (identical to 06_progress_curves_v2.py) ──────────────────────

def parse_grid(grid_str):
    rows = grid_str.strip("|").split("|")
    return [[int(c) for c in row] for row in rows]

def progress(grid_str, target_str):
    try:
        g = parse_grid(grid_str)
        t = parse_grid(target_str)
        if len(g) != len(t) or any(len(gr) != len(tr) for gr, tr in zip(g, t)):
            return 0.0
        total = sum(len(row) for row in t)
        match = sum(g[r][c] == t[r][c] for r in range(len(t)) for c in range(len(t[r])))
        return match / total if total > 0 else 0.0
    except Exception:
        return 0.0

def resample(curve, n=N_POINTS):
    x_old = np.linspace(0, 1, len(curve))
    x_new = np.linspace(0, 1, n)
    return np.interp(x_new, x_old, curve).tolist()

def normalise_v2(raw_curve, baseline):
    denom = 1.0 - baseline
    if denom <= 0:
        return [1.0] * len(raw_curve)
    return [(v - baseline) / denom for v in raw_curve]

def load_grids(task_id):
    with open(os.path.join(EVAL_DIR, f"{task_id}.json")) as f:
        task = json.load(f)
    inp_rows = task["test_examples"][0]["input"]
    out_rows = task["test_examples"][0]["output"]
    inp_str = "|" + "|".join("".join(str(c) for c in row) for row in inp_rows) + "|"
    tgt_str = "|" + "|".join("".join(str(c) for c in row) for row in out_rows) + "|"
    return inp_str, tgt_str

# ── load data ──────────────────────────────────────────────────────────────────

with open(CURVES_PATH) as f: curves       = json.load(f)
with open(HUMAN_PATH)  as f: human_traces = json.load(f)
with open(CODEIT_PATH) as f: codeit_traces= json.load(f)

diff_df  = pd.read_csv(DIFF_PATH)[["task_id", "difficulty_category"]]
diff_map = dict(zip(diff_df["task_id"], diff_df["difficulty_category"]))

all_task_ids = sorted(curves["tasks"].keys())

# ── pre-build individual normalised curves for bootstrap ───────────────────────
# boot_data[task_id][group] = list of 100-point v2-normalised curves

print("Pre-computing individual normalised trajectories for bootstrap …")
boot_data = {}
for task_id in all_task_ids:
    _, tgt_str = load_grids(task_id)
    baseline   = curves["tasks"][task_id]["baseline"]
    td = {"human_success": [], "human_failed": [],
          "codeit_success": [], "codeit_failed": []}

    for traj in human_traces.get(task_id, []):
        raw  = resample([progress(g, tgt_str) for g in traj["grids"]])
        norm = normalise_v2(raw, baseline)
        key  = "human_success" if traj["success"] else "human_failed"
        td[key].append(norm)

    for traj in codeit_traces.get(task_id, []):
        if not traj["grids"]:
            continue
        raw  = resample([progress(g, tgt_str) for g in traj["grids"]])
        norm = normalise_v2(raw, baseline)
        key  = "codeit_success" if traj["class"] == "success" else "codeit_failed"
        td[key].append(norm)

    boot_data[task_id] = td

# ── main loop: DTW + L2 + bootstrap CI per task × pair ────────────────────────

rng  = np.random.default_rng(0)
rows = []

total_pairs = sum(
    1 for tid in all_task_ids
    for pname, (ka, kb) in PAIRS.items()
    if (curves["tasks"][tid].get(ka) or {}).get("median_curve") is not None
    and (curves["tasks"][tid].get(kb) or {}).get("median_curve") is not None
    and (curves["tasks"][tid].get(ka) or {}).get("sample_size", 0) > 0
    and (curves["tasks"][tid].get(kb) or {}).get("sample_size", 0) > 0
)
print(f"Computing DTW for {total_pairs} task×pair combinations "
      f"(bootstrap B={BOOT_B} for human-involving pairs) …")

done = 0
for task_id in all_task_ids:
    td       = curves["tasks"][task_id]
    diff_cat = diff_map.get(task_id, "Unknown")

    for pair_name, (key_a, key_b) in PAIRS.items():
        grp_a = td.get(key_a) or {}
        grp_b = td.get(key_b) or {}
        ca = grp_a.get("median_curve")
        cb = grp_b.get("median_curve")
        n_a = grp_a.get("sample_size", 0)
        n_b = grp_b.get("sample_size", 0)
        if ca is None or cb is None or n_a == 0 or n_b == 0:
            continue

        ca_arr = np.array(ca)
        cb_arr = np.array(cb)

        # DTW distance (with matrix for potential path plot)
        dtw_dist, _ = dtw_and_matrix(ca, cb)

        # L2 distance (point-wise, for DTW/L2 ratio)
        l2_dist = float(np.sqrt(np.sum((ca_arr - cb_arr) ** 2)))

        ratio = dtw_dist / l2_dist if l2_dist > 1e-9 else float("nan")

        # Bootstrap CI
        if pair_name in SKIP_BOOTSTRAP:
            ci_lo, ci_hi = float("nan"), float("nan")
        else:
            raw_a = boot_data[task_id].get(key_a, [])
            raw_b = boot_data[task_id].get(key_b, [])
            ci_lo, ci_hi = compute_bootstrap_ci(raw_a, raw_b, B=BOOT_B, rng=rng)

        rows.append({
            "task_id":             task_id,
            "difficulty_category": diff_cat,
            "pair":                pair_name,
            "dtw_distance":        round(dtw_dist, 4),
            "dtw_ci_lo":           round(ci_lo, 4) if not np.isnan(ci_lo) else float("nan"),
            "dtw_ci_hi":           round(ci_hi, 4) if not np.isnan(ci_hi) else float("nan"),
            "l2_distance":         round(l2_dist, 4),
            "dtw_l2_ratio":        round(ratio, 4) if not np.isnan(ratio) else float("nan"),
            "n_a":                 n_a,
            "n_b":                 n_b,
        })
        done += 1
        if done % 50 == 0:
            print(f"  {done}/{total_pairs} done …")

per_task_df = pd.DataFrame(rows)
out_per_task = os.path.join(OUT_DIR, "dtw_per_task.csv")
per_task_df.to_csv(out_per_task, index=False)
print(f"Saved per-task DTW -> {out_per_task}  ({len(per_task_df)} rows)")

# ── summary: median by pair × difficulty_category ─────────────────────────────

summary = (
    per_task_df.groupby(["pair", "difficulty_category"])
    .agg(
        task_count      = ("task_id",       "count"),
        median_dtw      = ("dtw_distance",  "median"),
        median_l2       = ("l2_distance",   "median"),
        median_dtw_l2   = ("dtw_l2_ratio",  "median"),
    )
    .round(4)
    .reset_index()
)
out_summary = os.path.join(OUT_DIR, "dtw_summary.csv")
summary.to_csv(out_summary, index=False)
print(f"\nSaved summary -> {out_summary}")
print(summary.to_string(index=False))

# ── DTW strip plot (same format as script 08) ──────────────────────────────────

def strip_plot(metric, ylabel, title, out_path):
    pair_names = list(PAIRS.keys())
    fig, ax    = plt.subplots(figsize=(13, 5))
    rng_plot   = np.random.default_rng(42)
    legend_added = set()
    for xi, pair in enumerate(pair_names):
        sub = per_task_df[per_task_df["pair"] == pair]
        for cat, color in CAT_COLORS.items():
            pts = sub[sub["difficulty_category"] == cat][metric].dropna().values
            if len(pts) == 0:
                continue
            jitter = rng_plot.uniform(-0.18, 0.18, len(pts))
            label  = cat if cat not in legend_added else "_nolegend_"
            ax.scatter(xi + jitter, pts, c=color, s=35, alpha=0.75, label=label)
            legend_added.add(cat)
        med = sub[metric].median()
        ax.plot([xi - 0.28, xi + 0.28], [med, med],
                color="black", linewidth=2.0, zorder=5)
    ax.set_xticks(range(len(pair_names)))
    ax.set_xticklabels([PAIR_LABELS[p] for p in pair_names], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(rect=[0, 0, 0.87, 1])
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"Saved plot -> {out_path}")

strip_plot(
    "dtw_distance",
    "DTW distance",
    "DTW distance between paired median curves (each point = one task; bar = median)",
    os.path.join(OUT_DIR, "dtw_stripplot.png"),
)

# ── DTW vs L2 scatter ──────────────────────────────────────────────────────────

pair_markers = {
    "human_success_vs_codeit_success": "o",
    "human_failed_vs_codeit_failed":   "s",
    "human_success_vs_human_failed":   "^",
    "codeit_success_vs_codeit_failed": "D",
}

fig, ax = plt.subplots(figsize=(7, 6))
rng_plot = np.random.default_rng(42)
plotted_cats = set()
plotted_pairs = set()

for _, row in per_task_df.iterrows():
    color  = CAT_COLORS.get(row["difficulty_category"], "gray")
    marker = pair_markers[row["pair"]]
    cat_label  = row["difficulty_category"] if row["difficulty_category"] not in plotted_cats else "_nolegend_"
    pair_label = PAIR_LABELS[row["pair"]] if row["pair"] not in plotted_pairs else "_nolegend_"
    ax.scatter(row["l2_distance"], row["dtw_distance"],
               c=color, marker=marker, s=45, alpha=0.7,
               label=cat_label if cat_label != "_nolegend_" else None)
    plotted_cats.add(row["difficulty_category"])
    plotted_pairs.add(row["pair"])

# y = x reference (DTW = L2, no temporal flexibility gained)
lim_max = max(per_task_df["l2_distance"].max(), per_task_df["dtw_distance"].max()) + 0.5
ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=0.8, alpha=0.5, label="DTW = L2")
ax.set_xlim(0, lim_max)
ax.set_ylim(0, lim_max)
ax.set_xlabel("L2 distance (point-wise)")
ax.set_ylabel("DTW distance (temporally aligned)")
ax.set_title("DTW vs L2 distance per task×pair\n"
             "(below diagonal = temporal warping reduces gap)")

# Colour legend (difficulty categories)
from matplotlib.lines import Line2D
cat_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                      markersize=8, label=cat)
               for cat, c in CAT_COLORS.items()]
marker_handles = [Line2D([0], [0], marker=m, color="gray", markersize=7,
                         linestyle="None", label=PAIR_LABELS[p])
                  for p, m in pair_markers.items()]
leg1 = ax.legend(handles=cat_handles,    fontsize=7, loc="upper left",
                 title="Difficulty", title_fontsize=7)
ax.add_artist(leg1)
ax.legend(handles=marker_handles, fontsize=7, loc="lower right",
          title="Pair", title_fontsize=7)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "dtw_vs_l2_scatter.png"), dpi=130)
plt.close()
print(f"Saved plot -> {os.path.join(OUT_DIR, 'dtw_vs_l2_scatter.png')}")

# ── warping-path plots ─────────────────────────────────────────────────────────
# Select for pair "human_success_vs_codeit_success":
#   - 2 tasks with lowest DTW/L2 ratio (most temporal warping)
#   - 2 tasks with highest DTW/L2 ratio (warping doesn't help)
#   - 1 task with smallest human n (illustrate wide CI)

TARGET_PAIR = "human_success_vs_codeit_success"
pair_sub = per_task_df[per_task_df["pair"] == TARGET_PAIR].dropna(subset=["dtw_l2_ratio"])

lowest_ratio  = pair_sub.nsmallest(2, "dtw_l2_ratio")["task_id"].tolist()
highest_ratio = pair_sub.nlargest(2,  "dtw_l2_ratio")["task_id"].tolist()
smallest_n    = pair_sub.nsmallest(1, "n_a")["task_id"].tolist()

warp_tasks = []
seen = set()
for tid in lowest_ratio + highest_ratio + smallest_n:
    if tid not in seen:
        warp_tasks.append((tid, TARGET_PAIR))
        seen.add(tid)

print(f"\nGenerating warping-path plots for {len(warp_tasks)} tasks …")

for task_id, pair_name in warp_tasks:
    key_a, key_b = PAIRS[pair_name]
    td  = curves["tasks"][task_id]
    grp_a = td.get(key_a) or {}
    grp_b = td.get(key_b) or {}
    ca = grp_a.get("median_curve")
    cb = grp_b.get("median_curve")
    if ca is None or cb is None:
        continue

    dtw_dist, D = dtw_and_matrix(ca, cb)
    path        = dtw_path(D)
    path_i      = [p[0] for p in path]
    path_j      = [p[1] for p in path]

    row_info = per_task_df[
        (per_task_df["task_id"] == task_id) & (per_task_df["pair"] == pair_name)
    ].iloc[0]
    l2_dist   = row_info["l2_distance"]
    ratio     = row_info["dtw_l2_ratio"]
    ci_lo     = row_info["dtw_ci_lo"]
    ci_hi     = row_info["dtw_ci_hi"]
    diff_cat  = row_info["difficulty_category"]
    n_a, n_b  = int(row_info["n_a"]), int(row_info["n_b"])

    x = np.linspace(0, 1, N_POINTS)
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7, 8),
                                          gridspec_kw={"height_ratios": [1, 2]})

    # Top panel: two median curves
    label_a = f"{key_a.replace('_', ' ')} (n={n_a})"
    label_b = f"{key_b.replace('_', ' ')} (n={n_b})"
    ax_top.plot(x, ca, color="tab:blue",   linewidth=2, label=label_a)
    ax_top.plot(x, cb, color="tab:orange", linewidth=2, label=label_b)
    ax_top.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax_top.axhline(1, color="gray",  linewidth=0.5, linestyle=":",  alpha=0.3)
    ax_top.set_xlabel("Normalised step")
    ax_top.set_ylabel("v2 normalised progress")
    ci_str = (f"[{ci_lo:.3f}, {ci_hi:.3f}]"
              if not (np.isnan(ci_lo) or np.isnan(ci_hi)) else "n/a")
    ax_top.set_title(
        f"Task {task_id}  |  {diff_cat}\n"
        f"DTW={dtw_dist:.3f}  L2={l2_dist:.3f}  DTW/L2={ratio:.3f}  "
        f"95% CI {ci_str}"
    )
    ax_top.legend(fontsize=8)
    ax_top.grid(alpha=0.3)

    # Bottom panel: DTW cost matrix + optimal path
    # Mask out inf cells for display
    D_display = np.where(np.isinf(D), np.nan, D)
    im = ax_bot.imshow(D_display.T, origin="lower", aspect="auto",
                       cmap="viridis", interpolation="nearest")
    ax_bot.plot(path_i, path_j, color="white", linewidth=1.5,
                label="Optimal warp path")
    plt.colorbar(im, ax=ax_bot, label="Accumulated cost")
    ax_bot.set_xlabel(f"Time steps — {key_a.replace('_', ' ')}")
    ax_bot.set_ylabel(f"Time steps — {key_b.replace('_', ' ')}")
    ax_bot.set_title("DTW accumulated cost matrix (white = optimal alignment path)")
    ax_bot.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    safe_pair = pair_name.replace("_vs_", "_v_")
    fname = os.path.join(WARP_DIR, f"{task_id}_{safe_pair}.png")
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"  Saved warp path -> {fname}")

print(f"\nDone. All outputs in {OUT_DIR}/")
