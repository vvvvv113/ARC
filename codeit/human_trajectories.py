import json
import os
import sys

import numpy as np

# Make sure codelt environment is importable
_CODELT = os.path.join(os.path.dirname(__file__), "..", "codelt")
if _CODELT not in sys.path:
    sys.path.insert(0, _CODELT)

from codeit.policy.environment import execute_candidate_program_with_trace

N_POINTS = 100  # resolution for resampling progress curves


# ── grid helpers ──────────────────────────────────────────────────────────────

def _grid_to_str(grid):
    """Convert tuple-of-tuples grid to pipe-delimited string."""
    return "|" + "|".join("".join(str(c) for c in row) for row in grid) + "|"


def _parse_grid(grid_str):
    """Convert pipe-delimited string to list of lists of ints."""
    rows = grid_str.strip("|").split("|")
    return [[int(c) for c in row] for row in rows]


def _progress(grid_str, target_str):
    """Fraction of cells matching target. 0.0 if sizes differ or parse error."""
    try:
        g = _parse_grid(grid_str)
        t = _parse_grid(target_str)
        if len(g) != len(t) or any(len(gr) != len(tr) for gr, tr in zip(g, t)):
            return 0.0
        total = sum(len(row) for row in t)
        match = sum(g[r][c] == t[r][c] for r in range(len(t)) for c in range(len(t[r])))
        return match / total if total > 0 else 0.0
    except Exception:
        return 0.0


def _resample(curve, n=N_POINTS):
    """Linearly resample a list of floats to exactly n points."""
    x_old = np.linspace(0, 1, len(curve))
    x_new = np.linspace(0, 1, n)
    return np.interp(x_new, x_old, curve)


# ── DTW ───────────────────────────────────────────────────────────────────────

def _dtw(a, b):
    """Compute DTW distance between two 1-D arrays."""
    n, m = len(a), len(b)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return dp[n, m]


# ── public API ────────────────────────────────────────────────────────────────

def build_human_curves(human_traces_path):
    """
    Load human_traces.json and compute per-task progress curves.

    Only keeps success trajectories (success=True).

    Returns:
        dict: {task_id: [curve1, curve2, ...]}  where each curve is a np.array of length N_POINTS
    """
    with open(human_traces_path) as f:
        human_traces = json.load(f)

    human_curves = {}
    for task_id, trajectories in human_traces.items():
        # build target string from the last grid of any successful trajectory
        target_str = None
        for traj in trajectories:
            if traj["success"] and traj["grids"]:
                target_str = traj["grids"][-1]
                break
        if target_str is None:
            continue

        curves = []
        for traj in trajectories:
            if not traj["success"] or not traj["grids"]:
                continue
            raw = [_progress(g, target_str) for g in traj["grids"]]
            if len(raw) < 2:
                continue
            curves.append(_resample(raw))

        if curves:
            human_curves[task_id] = curves

    return human_curves


def get_program_curve(program_string, input_grid, target_grid):
    """
    Execute a DSL program with trace and return its progress curve.

    Args:
        program_string: DSL program as string
        input_grid: tuple-of-tuples input grid
        target_grid: tuple-of-tuples target (goal) grid

    Returns:
        np.array of length N_POINTS, or None if execution fails
    """
    target_str = _grid_to_str(target_grid)

    output, trace = execute_candidate_program_with_trace(program_string, input_grid)

    if isinstance(output, str):  # error
        return None

    if not trace:
        return None

    # build progress curve from trace grids
    raw = [_progress(_grid_to_str(g), target_str) for (_, g) in trace]

    if len(raw) < 2:
        # single-step program: prepend 0 so we have a meaningful curve
        raw = [0.0] + raw

    return _resample(raw)


def compute_dtw_similarity(program_curve, human_curves):
    """
    Compute similarity between a program's progress curve and a list of human curves.

    Takes the minimum DTW distance across all human curves, then converts to similarity.

    Returns:
        float in (0, 1]: 1 / (1 + min_dtw_distance)
        Returns 0.0 if human_curves is empty or program_curve is None.
    """
    if program_curve is None or not human_curves:
        return 0.0

    min_dist = min(_dtw(program_curve, hc) for hc in human_curves)
    return 1.0 / (1.0 + min_dist)
