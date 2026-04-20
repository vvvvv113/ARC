"""
Aggregate multi-seed baseline runs into mean ± std.
Usage: python aggregate_baseline.py /scratch/cy2941/codeit_outputs/h200_full_*_seed*
"""
import sys
import glob
import numpy as np
import pandas as pd
from pathlib import Path

def main(run_dirs):
    results = []
    for run_dir in run_dirs:
        csv_path = Path(run_dir) / "performance.csv"
        if not csv_path.exists():
            print(f"[skip] {run_dir}: no performance.csv")
            continue
        df = pd.read_csv(csv_path)
        seed = run_dir.split("seed")[-1].rstrip("/").split("_")[0]
        final_iter = df["meta_iteration"].max()
        final_cum = df.loc[df["meta_iteration"] == final_iter, "cumulative_performance"].iloc[0]
        final_per = df.loc[df["meta_iteration"] == final_iter, "performance"].iloc[0]
        results.append({
            "seed": int(seed),
            "run_dir": run_dir,
            "final_iter": int(final_iter),
            "cumulative_performance": float(final_cum),
            "per_iter_performance": float(final_per),
        })

    if not results:
        print("No results found.")
        return

    res_df = pd.DataFrame(results)
    print("\n=== Per-seed final results ===")
    print(res_df[["seed", "final_iter", "cumulative_performance", "per_iter_performance"]].to_string(index=False))

    cum = res_df["cumulative_performance"].values
    print("\n=== Baseline summary ===")
    print(f"N seeds:           {len(cum)}")
    print(f"Mean:              {cum.mean():.4f} ({cum.mean()*100:.2f}%)")
    print(f"Std:               {cum.std(ddof=1):.4f} ({cum.std(ddof=1)*100:.2f}%)")
    print(f"Min:               {cum.min():.4f} ({cum.min()*100:.2f}%)")
    print(f"Max:               {cum.max():.4f} ({cum.max()*100:.2f}%)")
    print(f"\nReport as: {cum.mean()*100:.2f}% ± {cum.std(ddof=1)*100:.2f}%")

    # Save aggregated result
    out_path = Path(__file__).parent / "baseline_results" / "multi_seed_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_path, index=False)
    print(f"\nSaved per-seed results → {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default glob
        run_dirs = sorted(glob.glob("/scratch/cy2941/codeit_outputs/h200_full_*_seed*"))
    else:
        run_dirs = []
        for arg in sys.argv[1:]:
            run_dirs.extend(glob.glob(arg))
        run_dirs = sorted(run_dirs)
    if not run_dirs:
        print("Usage: python aggregate_baseline.py [glob pattern ...]")
        sys.exit(1)
    main(run_dirs)
