"""
Compare Method 2 (DTW-biased) multi-seed runs against the baseline.

Reads performance.csv from /scratch run directories for both conditions and
emits:
  - CCS Project/baseline_results/method2_vs_baseline_summary.csv
  - CCS Project/baseline_results/method2_vs_baseline_curves.png

Run from repo root:
    python "CCS Project/compare_method2_vs_baseline.py"
"""
from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE / "baseline_results"
OUT.mkdir(parents=True, exist_ok=True)

BASELINE_DIRS = {
    17:  "/scratch/cy2941/codeit_outputs/h200_full_6686251_seed17",
    42:  "/scratch/cy2941/codeit_outputs/h200_full_6686252_seed42",
    123: "/scratch/cy2941/codeit_outputs/h200_full_6686253_seed123",
}
METHOD2_DIRS = {
    17:  "/scratch/cy2941/codeit_outputs/method2_h200_full_7064997_seed17_lambda0.5",
    42:  "/scratch/cy2941/codeit_outputs/method2_h200_full_7064998_seed42_lambda0.5",
    123: "/scratch/cy2941/codeit_outputs/method2_h200_full_7065000_seed123_lambda0.5",
}

PAPER = 0.1475   # CodeIt paper headline
MUT_D1 = 0.105   # Mutation d=1 baseline reference


def load(run_dir):
    df = pd.read_csv(Path(run_dir) / "performance.csv")
    return df.sort_values("meta_iteration").reset_index(drop=True)


def stack(dirs):
    """Return (iters_common, cum_array[seed, iter], perf_array[seed, iter])."""
    frames = {s: load(d) for s, d in dirs.items()}
    common_max = min(df["meta_iteration"].max() for df in frames.values())
    iters = np.arange(common_max + 1)
    cum = np.full((len(frames), len(iters)), np.nan)
    perf = np.full((len(frames), len(iters)), np.nan)
    for i, s in enumerate(sorted(frames)):
        df = frames[s].set_index("meta_iteration")
        for j, it in enumerate(iters):
            if it in df.index:
                cum[i, j] = df.loc[it, "cumulative_performance"]
                perf[i, j] = df.loc[it, "performance"]
    return iters, cum, perf, common_max


def paired_t(b, m):
    n = len(b)
    diff = [m[i] - b[i] for i in range(n)]
    md = sum(diff) / n
    if n < 2:
        return md, 0.0, float("nan")
    sd = math.sqrt(sum((d - md) ** 2 for d in diff) / (n - 1))
    se = sd / math.sqrt(n)
    t = md / se if se else float("inf")
    return md, sd, t


def main():
    iters_b, cum_b, perf_b, max_b = stack(BASELINE_DIRS)
    iters_m, cum_m, perf_m, max_m = stack(METHOD2_DIRS)
    common_max = min(max_b, max_m)
    iters = np.arange(common_max + 1)
    cum_b = cum_b[:, : common_max + 1]
    cum_m = cum_m[:, : common_max + 1]
    perf_b = perf_b[:, : common_max + 1]
    perf_m = perf_m[:, : common_max + 1]

    # ---------------- Per-seed final values + paired test at common_max ----
    rows = []
    for i, seed in enumerate(sorted(BASELINE_DIRS)):
        rows.append({
            "seed": seed,
            "baseline_cum_final": cum_b[i, -1],
            "method2_cum_final": cum_m[i, -1],
            "delta_cum": cum_m[i, -1] - cum_b[i, -1],
            "baseline_perf_final": perf_b[i, -1],
            "method2_perf_final": perf_m[i, -1],
            "delta_perf": perf_m[i, -1] - perf_b[i, -1],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "method2_vs_baseline_summary.csv", index=False)

    print(f"=== Baseline vs Method 2 (λ=0.5) @ common epoch {common_max} ===\n")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n--- Aggregate (mean ± sd over 3 seeds) ---")
    for col_b, col_m, name in [
        ("baseline_cum_final", "method2_cum_final", "cumulative_performance"),
        ("baseline_perf_final", "method2_perf_final", "per-iter performance"),
    ]:
        b = summary[col_b].values
        m = summary[col_m].values
        md, sd, t = paired_t(list(b), list(m))
        sig = "*" if abs(t) > 4.303 else "n.s."  # df=2, two-sided 0.05
        print(f"\n[{name}]")
        print(f"  baseline: {b.mean()*100:.2f}% ± {b.std(ddof=1)*100:.2f}%")
        print(f"  method2 : {m.mean()*100:.2f}% ± {m.std(ddof=1)*100:.2f}%")
        print(f"  Δ (m-b) : {md*100:+.2f}pp   paired t(2)={t:.3f}  ({sig} at p=0.05)")

    # ---------------- Plot overlaid curves ---------------------------------
    mean_b = np.nanmean(cum_b, axis=0)
    sd_b = np.nanstd(cum_b, axis=0, ddof=1)
    mean_m = np.nanmean(cum_m, axis=0)
    sd_m = np.nanstd(cum_m, axis=0, ddof=1)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i in range(cum_b.shape[0]):
        ax.plot(iters, cum_b[i], color="#1f77b4", alpha=0.25, lw=1)
    for i in range(cum_m.shape[0]):
        ax.plot(iters, cum_m[i], color="#d62728", alpha=0.25, lw=1)

    ax.plot(iters, mean_b, color="#1f77b4", lw=2.4, label=f"Baseline mean (n={cum_b.shape[0]})")
    ax.fill_between(iters, mean_b - sd_b, mean_b + sd_b, color="#1f77b4", alpha=0.18)
    ax.plot(iters, mean_m, color="#d62728", lw=2.4, label=f"Method 2 (λ=0.5) mean (n={cum_m.shape[0]})")
    ax.fill_between(iters, mean_m - sd_m, mean_m + sd_m, color="#d62728", alpha=0.18)

    ax.axhline(PAPER, color="gray", ls="--", lw=1, label=f"CodeIt paper {PAPER*100:.2f}%")
    ax.axhline(MUT_D1, color="gray", ls=":", lw=1, label=f"Mutation d=1 {MUT_D1*100:.2f}%")

    ax.set_xlabel("Meta-iteration")
    ax.set_ylabel("Cumulative ARC-eval pass rate")
    ax.set_title("Method 2 (DTW-biased mutation) vs CodeIt baseline — multi-seed")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "method2_vs_baseline_curves.png", dpi=150)
    print(f"\nSaved → {OUT/'method2_vs_baseline_summary.csv'}")
    print(f"Saved → {OUT/'method2_vs_baseline_curves.png'}")


if __name__ == "__main__":
    main()
