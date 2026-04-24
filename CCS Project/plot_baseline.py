"""
Plot the final CodeIt multi-seed baseline + check convergence.

Produces:
  CCS Project/baseline_results/baseline_multiseed_curves.png
    - all per-seed cumulative curves overlaid
    - black mean across seeds + ±1σ shaded band
    - paper CodeIt (14.75%) and Mutation d1 (10.5%) reference lines

Convergence check (printed to stdout): for every per-seed CSV found, reports
final cum, last iter where cum rose, flat window, tail cumulative range,
tail per-iter mean/std, and a YES/yes*/NO verdict.

Expected per-seed inputs (rsynced from HPC scratch):
  CCS Project/baseline_results/performance_seed17.csv
  CCS Project/baseline_results/performance_seed42.csv
  CCS Project/baseline_results/performance_seed123.csv
Also picks up CCS Project/baseline_results/performance.csv (earlier single run).

Run from repo root:
    python3 "CCS Project/plot_baseline.py"
"""
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RESULTS = HERE / "baseline_results"

PAPER = 0.1475
MUT_D1 = 0.105
TAIL = 20            # window for per-iter tail stats
PLATEAU_MIN = 20     # iters with no cum improvement (strict convergence)
BAND_PP = 0.01       # <1pp cumulative range over tail → converged (noise-tolerant)


def find_trajectories():
    """Return {label: csv_path} for every per-seed performance.csv available."""
    trajs = {}
    default = RESULTS / "performance.csv"
    if default.exists():
        trajs["earlier single run"] = default
    for p in sorted(RESULTS.glob("performance_seed*.csv")):
        m = re.search(r"seed(\d+)", p.name)
        trajs[f"seed{m.group(1)}"] = p
    return trajs


def convergence_report(label, csv_path):
    df = pd.read_csv(csv_path).sort_values("meta_iteration").reset_index(drop=True)
    cum = df["cumulative_performance"].values
    per = df["performance"].values
    iters = df["meta_iteration"].values

    diffs = np.diff(cum)
    last_rise_idx = np.where(diffs > 0)[0]
    last_rise_iter = int(iters[last_rise_idx[-1] + 1]) if len(last_rise_idx) else int(iters[0])
    final_iter = int(iters[-1])
    flat_for = final_iter - last_rise_iter

    tail = per[-TAIL:]
    tail_mean = float(tail.mean())
    tail_std = float(tail.std(ddof=1)) if len(tail) > 1 else 0.0

    tail_cum = cum[-TAIL:]
    tail_cum_range = float(tail_cum.max() - tail_cum.min())

    if flat_for >= PLATEAU_MIN:
        verdict = "YES"
    elif tail_cum_range < BAND_PP:
        verdict = "yes*"
    else:
        verdict = "NO"

    return {
        "label": label,
        "final_iter": final_iter,
        "final_cum": float(cum[-1]),
        "last_rise_iter": last_rise_iter,
        "flat_for": flat_for,
        "tail_cum_range": tail_cum_range,
        "tail_mean": tail_mean,
        "tail_std": tail_std,
        "converged": verdict,
    }


def print_convergence_table(rows):
    print(f"\n=== Convergence check (strict: cum flat ≥{PLATEAU_MIN} iters, "
          f"band: cum range <{BAND_PP*100:.1f}pp in last {TAIL}) ===")
    header = (f"{'label':<25} {'final_iter':>10} {'final_cum':>10} "
              f"{'last_rise':>10} {'flat_for':>9} {'cum_range':>10} "
              f"{'tail_mean':>10} {'tail_std':>9} {'converged':>10}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['label']:<25} {r['final_iter']:>10} {r['final_cum']:>10.4f} "
              f"{r['last_rise_iter']:>10} {r['flat_for']:>9} "
              f"{r['tail_cum_range']:>10.4f} "
              f"{r['tail_mean']:>10.4f} {r['tail_std']:>9.4f} "
              f"{r['converged']:>10}")
    print("\nYES   = strict (cum fully flat)")
    print("yes*  = noise-tolerant (cum range <1pp in last 20 iters)")
    print("NO    = still moving beyond band")


def plot_multiseed_curves():
    """Overlay all per-seed cumulative curves + mean±std band across seeds."""
    per_seed = {}
    for p in sorted(RESULTS.glob("performance_seed*.csv")):
        m = re.search(r"seed(\d+)", p.name)
        if not m:
            continue
        seed = int(m.group(1))
        df = pd.read_csv(p).sort_values("meta_iteration").reset_index(drop=True)
        per_seed[seed] = df

    if len(per_seed) < 2:
        print(f"[skip] need ≥2 per-seed CSVs, found {len(per_seed)}")
        return

    common_iters = None
    for df in per_seed.values():
        its = set(df["meta_iteration"].astype(int).tolist())
        common_iters = its if common_iters is None else common_iters & its
    common_iters = sorted(common_iters)

    seeds_sorted = sorted(per_seed.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(figsize=(9, 5.5))

    aligned = []
    for i, seed in enumerate(seeds_sorted):
        df = per_seed[seed]
        sub = df[df["meta_iteration"].astype(int).isin(common_iters)].sort_values("meta_iteration")
        y = sub["cumulative_performance"].values
        aligned.append(y)
        ax.plot(sub["meta_iteration"], y, color=colors[i], alpha=0.55, linewidth=1.2,
                label=f"seed {seed} (final {y[-1]:.2%} @ iter {int(sub['meta_iteration'].iloc[-1])})")

    arr = np.vstack(aligned)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)

    ax.plot(common_iters, mean, color="black", linewidth=2.5,
            label=f"mean (final {mean[-1]:.2%})")
    ax.fill_between(common_iters, mean - std, mean + std, color="black", alpha=0.15,
                    label="±1σ across seeds")

    ax.axhline(PAPER, color="#d62728", linestyle="--", linewidth=1,
               label=f"paper CodeIt ({PAPER:.2%})")
    ax.axhline(MUT_D1, color="#7f7f7f", linestyle=":", linewidth=1,
               label=f"paper Mutation d1 ({MUT_D1:.2%})")

    ax.set_xlabel("meta-iteration")
    ax.set_ylabel("cumulative ARC eval accuracy")
    ax.set_title(f"CodeIt baseline — {len(seeds_sorted)}-seed learning curves (H200)")
    ax.set_ylim(0, max(PAPER, arr.max()) * 1.2)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = RESULTS / "baseline_multiseed_curves.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    plot_multiseed_curves()

    trajs = find_trajectories()
    if trajs:
        rows = [convergence_report(label, path) for label, path in trajs.items()]
        print_convergence_table(rows)
    else:
        print("\n[convergence] no per-seed performance.csv found.")
