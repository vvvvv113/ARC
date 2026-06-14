"""
S08_state_graph_new.py

For any task (default: bf699163):
  Section 1 — Task description: training examples + test input/target
  Section 2 — Human success traces (one row per participant)
  Section 3 — CodeIt success traces: every DSL line shown as its own box
              Grid-valued steps → actual grid image
              Non-grid steps    → empty dashed placeholder box
              Each arrow labeled with the single DSL line it represents

DSL step data recovered by executing success programs from the solutions tar.gz
files (same paths as S03). This avoids hardcoding and works for any task.

Layout: all grids at fixed HEIGHT; width scales with aspect ratio.
Positions computed in inches → converted to figure fractions.
"""

import os, sys, json, tarfile, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, FancyBboxPatch
from matplotlib.lines import Line2D

TASK_ID = "1acc24af"

REPO            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK_JSON       = os.path.join(REPO, "codelt/data/evaluation", f"{TASK_ID}.json")
HUMAN_TRACES_J  = os.path.join(REPO, "analysis_5_2/processed/S02_human_traces/human_traces_all.json")
CODEIT_TRACES_J = os.path.join(REPO, "analysis_5_2/processed/S03_codeit_traces/codeit_traces_3seeds.json")
OUT_DIR         = os.path.join(REPO, "analysis_5_2/processed/S08_state_graph_new")
os.makedirs(OUT_DIR, exist_ok=True)

HUMAN_CSV = os.path.join(REPO, "human_data/data/data.csv")
RESET_GAP = 0.50    # horizontal gap (inches) between two attempts in a row
RESET_COL = '#CC0000'  # red color for attempt-boundary markers

SEEDS = {
    "seed17":  (os.path.expanduser("~/Downloads/codeit_data/h200_full_6686251_seed17.tar.gz"),
                "h200_full_6686251_seed17/solutions_97.json"),
    "seed42":  (os.path.expanduser("~/Downloads/codeit_data/h200_full_6686252_seed42.tar.gz"),
                "h200_full_6686252_seed42/solutions_95.json"),
    "seed123": (os.path.expanduser("~/Downloads/codeit_data/h200_full_6686253_seed123.tar.gz"),
                "h200_full_6686253_seed123/solutions_96.json"),
}

ARC_PAL = {
    0:[0,0,0], 1:[0,116,217], 2:[255,65,54], 3:[46,204,64],
    4:[255,220,0], 5:[170,170,170], 6:[240,18,190],
    7:[255,133,27], 8:[127,219,255], 9:[135,12,37],
}

def parse_grid_str(s):
    rows = s.strip("|").split("|")
    if any("-" in row for row in rows):
        # Grid contains negative values (DSL intermediate with non-ARC values):
        # use regex to parse single-digit signed integers, e.g. "-1-100" → [-1,-1,0,0]
        return [[int(m) for m in re.findall(r"-?\d", row)] for row in rows]
    return [[int(c) for c in row] for row in rows]

def grid_to_img(grid):
    H, W = len(grid), len(grid[0])
    img = np.zeros((H, W, 3))
    for r in range(H):
        for c in range(W):
            img[r, c] = [v/255 for v in ARC_PAL.get(grid[r][c], [128,128,128])]
    return img

# ── DSL full-step reconstruction ───────────────────────────────────────────────

sys.path.insert(0, REPO)
try:
    from codeit.policy.environment import execute_candidate_program_with_trace as _exec_trace
    _HAS_EXEC = True
except ImportError:
    _HAS_EXEC = False
    print("Warning: codeit not importable — DSL step display will be skipped")

def load_success_programs(task_id):
    """Find all unique success program strings for task_id across the 3 seed solutions files."""
    seen, progs = set(), []
    for seed_name, (tar_path, sol_file) in SEEDS.items():
        try:
            with tarfile.open(tar_path) as tf:
                f = tf.extractfile(sol_file)
                data = json.load(f)
            policy = data.get("policy", {})
            for split_name in ("seen_example", "task_demonstration", "test"):
                for prog_str, details in policy.get(split_name, {}).get(task_id, {}).items():
                    perf = details.get("test_performance", [])
                    if perf and perf[0] is True and prog_str not in seen:
                        seen.add(prog_str)
                        progs.append(prog_str)
        except Exception as e:
            print(f"  Warning: {seed_name} load error: {e}")
    return progs

def _grid_str_from_grid(g):
    return "|" + "|".join("".join(str(c) for c in row) for row in g) + "|"

def short_dsl(line_str):
    """'x1 = vmirror(I)' → 'vmirror(I)'"""
    if "=" in line_str:
        return line_str.split("=", 1)[1].strip()
    return line_str.strip()

def reconstruct_full_steps(full_steps_meta, grids_parsed):
    """
    Reconstruct [(dsl_line, grid_or_None), ...] from S03 JSON fields without re-executing.

    full_steps_meta : t["full_steps"] = [{"dsl": str, "has_grid": bool}, ...]
    grids_parsed    : already-parsed grid list where grids_parsed[0] = input grid,
                      grids_parsed[1:] = intermediate grids in the same order as
                      the has_grid=True entries in full_steps_meta.
    """
    result = []
    grid_idx = 1   # grids_parsed[0] is the input; intermediate grids start at index 1
    for entry in full_steps_meta:
        if entry["has_grid"]:
            grid = grids_parsed[grid_idx] if grid_idx < len(grids_parsed) else None
            grid_idx += 1
        else:
            grid = None
        result.append((entry["dsl"], grid))
    return result


def build_full_steps_by_exec(task_id, test_input_list):
    """
    Fallback: execute programs from solutions tar.gz to build full step sequences.
    Used only when S03 JSON does not yet contain the 'full_steps' field
    (i.e., old JSON produced before the S03 fix).
    Returns dict: first_intermediate_grid_str → [(dsl_line, grid_or_None), ...]
    """
    if not _HAS_EXEC:
        return {}
    test_tuple = tuple(tuple(r) for r in test_input_list)
    step_map = {}
    progs = load_success_programs(task_id)
    print(f"  Found {len(progs)} unique success programs for {task_id}")
    for prog in progs:
        try:
            _, trace = _exec_trace(prog, test_tuple)
        except Exception:
            continue
        if len(trace) < 2:
            continue
        grid_by_line = {lbl: [list(row) for row in g] for lbl, g in trace[1:]}
        lines = prog.strip().split("\n")
        full_steps = [(l.strip(), grid_by_line.get(l.strip(), None)) for l in lines]
        key = _grid_str_from_grid(trace[1][1])
        step_map[key] = full_steps
    return step_map

# ── Human all-attempts loader ─────────────────────────────────────────────────

def load_all_human_attempts(task_id):
    """
    Read raw CSV directly. For each participant who solved task_id at least once,
    return ALL their attempts (sorted by attempt_number), with grids deduped within
    each attempt.  This avoids the S02 "last-attempt only" limitation and shows the
    full problem-solving journey, including carried-over state across attempts.

    Returns list of dicts (one per solved participant):
      {"hashed_id": str,
       "attempts": [{"num": int, "grids": [[...], ...], "success": bool}, ...]}
    """
    df = pd.read_csv(HUMAN_CSV)
    task_df = df[
        (df["task_name"] == f"{task_id}.json") &
        (df["task_type"] == "evaluation")
    ]

    participants = []
    for hid, p_df in task_df.groupby("hashed_id"):
        if not p_df["solved"].any():
            continue   # only keep participants who eventually solved the task

        attempts = []
        for att_num in sorted(p_df["attempt_number"].unique()):
            att_df = p_df[p_df["attempt_number"] == att_num].sort_values("action_id")
            grids, prev = [], None
            for g_str in att_df["test_output_grid"]:
                if g_str != prev:
                    try:
                        grids.append(parse_grid_str(g_str))
                        prev = g_str
                    except Exception:
                        pass
            if grids:
                attempts.append({
                    "num":     int(att_num),
                    "grids":   grids,
                    "success": bool(att_df["solved"].any()),
                })

        if attempts:
            participants.append({"hashed_id": hid, "attempts": attempts})

    return participants


MAX_PER_LINE = 10    # grids per visual line before wrapping
INNER_PAD    = 0.18  # vertical gap between wrapped lines within one participant


def _participant_grid_list(participant):
    """Flat list of (grid, is_attempt_start, attempt_num) for every grid."""
    items = []
    for att in participant["attempts"]:
        for i, g in enumerate(att["grids"]):
            items.append((g, i == 0, att["num"]))
    return items


def participant_n_lines(participant):
    total = sum(len(a["grids"]) for a in participant["attempts"])
    return max(1, -(-total // MAX_PER_LINE))   # ceiling division


def participant_block_height(participant):
    """Total vertical height (inches) this participant's block occupies."""
    n = participant_n_lines(participant)
    return n * GH + (n - 1) * INNER_PAD


def participant_max_line_width(participant):
    """Widest single wrapped line (inches) — used to set FIG_W."""
    items = _participant_grid_list(participant)
    # split into lines of MAX_PER_LINE
    max_w = 0.0
    for start in range(0, len(items), MAX_PER_LINE):
        chunk = items[start:start + MAX_PER_LINE]
        w = 0.0
        for j, (g, _, _) in enumerate(chunk):
            gw, _ = grid_display_size(g)
            w += gw + (ARROW_IN if j < len(chunk) - 1 else 0)
        max_w = max(max_w, w)
    return max_w


# ── Load task and trace data ───────────────────────────────────────────────────

with open(TASK_JSON) as f:
    task = json.load(f)
train_exs   = task.get("training_examples", [])
test_input  = task["test_examples"][0]["input"]
test_target = task["test_examples"][0]["output"]

with open(HUMAN_TRACES_J) as f:
    human_raw = json.load(f)
with open(CODEIT_TRACES_J) as f:
    codeit_raw = json.load(f)

print("Loading all human attempts from CSV...")
human_participants = load_all_human_attempts(TASK_ID)
print(f"Human participants (solved): {len(human_participants)}")
for p in human_participants:
    n_total = sum(len(a["grids"]) for a in p["attempts"])
    print(f"  {p['hashed_id'][:8]}: {len(p['attempts'])} attempt(s), "
          f"{n_total} grids total → {participant_n_lines(p)} visual line(s)")

codeit_success_traces = [t for t in codeit_raw.get(TASK_ID, []) if t.get("class") == "success"]
codeit_seqs  = [[parse_grid_str(g) for g in t["grids"]] for t in codeit_success_traces]
codeit_seeds = [t.get("seed", "") for t in codeit_success_traces]

print(f"Human participants (solved, all attempts): {len(human_participants)}")
print(f"CodeIt success traces: {len(codeit_seqs)}  seeds: {codeit_seeds}")

# Build per-line full step sequences for CodeIt.
# Prefer reading 'full_steps' stored in S03 JSON (no re-execution needed).
# Fall back to re-executing from solutions tar.gz only for old JSON without that field.
codeit_full_steps = []
need_exec = []

for i, (t, seq_parsed) in enumerate(zip(codeit_success_traces, codeit_seqs)):
    if "full_steps" in t:
        codeit_full_steps.append(reconstruct_full_steps(t["full_steps"], seq_parsed))
    else:
        codeit_full_steps.append(None)
        need_exec.append(i)

if need_exec:
    print(f"  'full_steps' missing for {len(need_exec)} trace(s) — re-executing from solutions files...")
    step_map = build_full_steps_by_exec(TASK_ID, test_input)
    for i in need_exec:
        raw_grids = codeit_success_traces[i]["grids"]
        key = raw_grids[1] if len(raw_grids) > 1 else ""
        codeit_full_steps[i] = step_map.get(key, [])
else:
    print("  full_steps loaded directly from S03 JSON (no re-execution)")

for i, steps in enumerate(codeit_full_steps):
    print(f"  C{i+1}: {len(steps)} DSL steps — "
          + ", ".join("grid" if g is not None else "none" for _, g in steps))

# ── Layout constants (inches) ──────────────────────────────────────────────────

DPI            = 130
GH             = 0.72    # fixed display HEIGHT for all trace grids
TASK_C         = 0.12    # cell size per cell for task description inputs
ARROW_IN       = 0.32    # human rows: gap between grids
ARROW_IN_C     = 0.62    # CodeIt rows: wider gap to fit longer labels
PLACEHOLDER_W  = GH      # width of empty placeholder box for non-grid steps
ROW_PAD        = 0.28    # vertical gap between rows
BETWEEN_PART   = 0.48    # extra vertical gap between consecutive participant blocks
SECT_PAD       = 0.50    # vertical gap between sections
L_MARGIN       = 1.10
R_MARGIN       = 0.30
T_MARGIN       = 0.55
B_MARGIN       = 0.30

# Extra vertical margin above each CodeIt row for labels that go above/below the grid.
# Labels sit ±0.44" from row center (= 0.08" outside the grid top/bottom at GH=0.72").
# Rows must be at least GH + 2*0.44 = 1.60" apart; 0.58 pad + 0.72 grid = 1.30", so
# adjacent-row labels have ~0.30" clearance between them.
CODEIT_ROW_PAD = 0.58    # taller than human ROW_PAD to accommodate above/below labels
codeit_row_stride = GH + CODEIT_ROW_PAD
row_stride        = GH + ROW_PAD

def grid_display_size(grid, fixed_h=GH):
    H, W = len(grid), len(grid[0])
    return (W / H) * fixed_h, fixed_h

def row_display_width(seq, arrow=ARROW_IN, fixed_h=GH):
    total = 0.0
    for i, g in enumerate(seq):
        gw, _ = grid_display_size(g, fixed_h)
        total += gw + (arrow if i < len(seq) - 1 else 0)
    return total

def codeit_full_row_width(input_grid, full_steps):
    """Total display width for a CodeIt row with placeholder boxes."""
    if not full_steps:
        return row_display_width([input_grid], ARROW_IN_C)
    gw_in, _ = grid_display_size(input_grid)
    total = gw_in
    for i, (_, grid) in enumerate(full_steps):
        total += ARROW_IN_C
        total += grid_display_size(grid)[0] if grid is not None else PLACEHOLDER_W
    return total

# Widths for layout
human_max_w  = max((participant_max_line_width(p) for p in human_participants), default=GH)
codeit_max_w = max(
    (codeit_full_row_width(seq[0], steps)
     for seq, steps in zip(codeit_seqs, codeit_full_steps)
     if steps),
    default=GH
)

all_task_rows = [(ex["input"], ex["output"], f"Example {i+1}", '#666666', '#2ECC40')
                 for i, ex in enumerate(train_exs)]
all_task_rows.append((test_input, test_target, "Test", '#0074D9', '#2ECC40'))

task_row_hs  = [len(inp) * TASK_C for inp, *_ in all_task_rows]
task_sec_h   = sum(task_row_hs) + len(all_task_rows) * (ROW_PAD + 0.32) + 0.50
task_row_w   = max(
    len(inp[0]) * TASK_C + ARROW_IN + (len(out[0]) / len(out)) * GH
    for inp, out, *_ in all_task_rows
)

FIG_W = L_MARGIN + max(human_max_w, codeit_max_w, task_row_w) + R_MARGIN
FIG_W = max(FIG_W, 14.0)

if human_participants:
    human_sec_h = (sum(participant_block_height(p) for p in human_participants)
                   + max(0, len(human_participants) - 1) * BETWEEN_PART
                   + 0.65)   # header + bottom margin
else:
    human_sec_h = 0.45
codeit_sec_h = (len(codeit_seqs) * codeit_row_stride + 0.55) if codeit_seqs else 0

FIG_H = B_MARGIN + codeit_sec_h + SECT_PAD + human_sec_h + SECT_PAD + task_sec_h + T_MARGIN + 0.40

# ── Figure primitives ─────────────────────────────────────────────────────────

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
fig.patch.set_facecolor('white')

def frac(x, y, w, h):
    return [x/FIG_W, y/FIG_H, w/FIG_W, h/FIG_H]

def add_grid(x_in, y_in, grid, fixed_h=GH, border='#444444', lw=1.5):
    gw, gh = grid_display_size(grid, fixed_h)
    ax = fig.add_axes(frac(x_in, y_in, gw, gh))
    ax.imshow(grid_to_img(grid), interpolation='nearest', aspect='auto')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(border); sp.set_linewidth(lw)
    return ax, gw, gh

def add_placeholder(x_in, y_in, var_name, fixed_h=GH, pw=PLACEHOLDER_W, border='#0074D9'):
    """Draw an empty dashed box (for non-grid intermediate steps)."""
    ax = fig.add_axes(frac(x_in, y_in, pw, fixed_h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor('#f4f7fc')
    for sp in ax.spines.values():
        sp.set_edgecolor(border)
        sp.set_linewidth(1.5)
        sp.set_linestyle((0, (5, 3)))   # dashed
    # Variable name centered; small italic label below
    ax.text(0.5, 0.60, var_name,
            ha='center', va='center', fontsize=9, color='#0074D9',
            fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.28, "(non-grid)",
            ha='center', va='center', fontsize=6, color='#888888',
            fontstyle='italic', transform=ax.transAxes)
    return ax, pw, fixed_h

def connect(ax_l, ax_r, color='#666666', lw=1.5):
    fig.add_artist(ConnectionPatch(
        xyA=(1.0, 0.5), coordsA='axes fraction', axesA=ax_l,
        xyB=(0.0, 0.5), coordsB='axes fraction', axesB=ax_r,
        arrowstyle='->', color=color, lw=lw,
        mutation_scale=13, zorder=10, shrinkA=3, shrinkB=3,
    ))

def ftxt(x_in, y_in, text, **kw):
    fig.text(x_in/FIG_W, y_in/FIG_H, text, **kw)

def draw_row(x0, y_bot, grids, fixed_h=GH, bc='#444444', ac='#666666'):
    """Draw a human trace row (grid images only, no labels)."""
    axs, x = [], x0
    for i, g in enumerate(grids):
        ax, gw, gh = add_grid(x, y_bot, g, fixed_h, border=bc)
        if axs:
            connect(axs[-1], ax, color=ac)
        axs.append(ax)
        x += gw + (ARROW_IN if i < len(grids) - 1 else 0)
    return axs

def draw_codeit_row_full(x0, y_bot, input_grid, full_steps, fixed_h=GH,
                         bc='#0074D9', ac='#0074D9'):
    """
    Draw a CodeIt trace row showing EVERY DSL line as its own box.

    input_grid : the initial test input (list-of-lists)
    full_steps : [(dsl_line_stripped, grid_or_None), ...] — one entry per DSL line
                 grid_or_None is a list-of-lists grid or None for non-grid results

    Each arrow carries exactly one DSL label.
    Labels alternate above / below the grid centerline to avoid overlap:
      even-index arrows (0, 2, 4, …) → above (+0.20")
      odd-index  arrows (1, 3, …)    → below (−0.17")
    """
    axs = []
    xs  = []
    ws  = []
    x   = x0

    # Draw input grid
    ax, gw, gh = add_grid(x, y_bot, input_grid, fixed_h, border=bc)
    axs.append(ax); xs.append(x); ws.append(gw)
    x += gw

    for arrow_idx, (dsl_line, grid) in enumerate(full_steps):
        x += ARROW_IN_C

        if grid is not None:
            ax, bw, bh = add_grid(x, y_bot, grid, fixed_h, border=bc)
        else:
            var_name = dsl_line.split("=")[0].strip() if "=" in dsl_line else "?"
            ax, bw, bh = add_placeholder(x, y_bot, var_name, fixed_h, border=bc)

        # Arrow from previous box to this box
        connect(axs[-1], ax, color=ac, lw=1.8)

        # DSL label on arrow — strip variable prefix, show only RHS call
        lbl = short_dsl(dsl_line)
        mid_x = (xs[-1] + ws[-1] + x) / 2   # horizontal center of the gap (inches)
        mid_y = y_bot + fixed_h / 2           # vertical center of the row (inches)
        # ±0.44" from row center puts labels 0.08" outside the grid top/bottom (GH/2=0.36")
        y_off = +0.44 if arrow_idx % 2 == 0 else -0.44
        ftxt(mid_x, mid_y + y_off, lbl,
             fontsize=7.5, color=ac, ha='center', va='center', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.92),
             zorder=20)

        axs.append(ax); xs.append(x); ws.append(bw)
        x += bw

    return axs


def draw_human_row_multi_attempt(x0, y_block_bot, participant, fixed_h=GH):
    """
    Draw all attempts for a participant, wrapping at MAX_PER_LINE grids per visual line.

    Layout (y increases upward in matplotlib figure coords):
      - Line 0 (first 10 grids) → top of the block (highest y)
      - Line 1 (grids 11-20)   → below line 0
      - …

    At each attempt boundary:
      - Red "Attempt X" label above the first grid of that attempt
      - Dashed red vertical separator if the boundary falls mid-line (not at x0)
      - No arrow drawn between the last grid of attempt N and first of N+1

    y_block_bot : bottom-most y coordinate (inches) of this participant's block.
    """
    items  = _participant_grid_list(participant)   # [(grid, is_attempt_start, att_num)]
    n_lines = participant_n_lines(participant)

    for line_idx in range(n_lines):
        chunk = items[line_idx * MAX_PER_LINE : (line_idx + 1) * MAX_PER_LINE]
        # Top visual line has highest y; each subsequent line is (fixed_h + INNER_PAD) lower
        line_y_bot = y_block_bot + (n_lines - 1 - line_idx) * (fixed_h + INNER_PAD)

        x       = x0
        prev_ax = None

        for item_in_line, (g, is_start, att_num) in enumerate(chunk):
            gw, _ = grid_display_size(g, fixed_h)

            if is_start:
                # Red "Attempt X" label above this grid
                ftxt(x + gw / 2, line_y_bot + fixed_h + 0.04,
                     f"Attempt {att_num}", fontsize=6.5, color=RESET_COL,
                     ha='center', va='bottom', fontweight='bold')
                # Dashed vertical separator when boundary is mid-line
                if item_in_line > 0:
                    sep_x = x - ARROW_IN / 2
                    fig.add_artist(plt.Line2D(
                        [sep_x / FIG_W, sep_x / FIG_W],
                        [(line_y_bot - 0.04) / FIG_H,
                         (line_y_bot + fixed_h + 0.04) / FIG_H],
                        color=RESET_COL, lw=1.2, linestyle='--',
                        zorder=15, transform=fig.transFigure,
                    ))

            ax, gw_actual, _ = add_grid(x, line_y_bot, g, fixed_h, border='#CC6600')

            # Arrow only within the same attempt — skip across attempt boundaries
            if prev_ax is not None and not is_start:
                connect(prev_ax, ax, color='#CC6600')

            prev_ax = ax
            x += gw_actual + (ARROW_IN if item_in_line < len(chunk) - 1 else 0)


# ── Y anchors ─────────────────────────────────────────────────────────────────

y_codeit = B_MARGIN + 0.55
y_human  = y_codeit + codeit_sec_h + SECT_PAD + 0.45
y_task   = y_human  + human_sec_h  + SECT_PAD
x0       = L_MARGIN

# ══════════════════════════════════════════════════════════════════════════════
# Section 1: Task Description
# ══════════════════════════════════════════════════════════════════════════════

fig.text(0.5, 0.997, f"Task {TASK_ID} — State Progression",
         fontsize=12, fontweight='bold', color='#111111', ha='center', va='top')

ftxt(x0, y_task + task_sec_h - 0.06,
     "Task description", fontsize=9, fontweight='bold', color='#333333', va='top')

task_rows_reversed = list(reversed(all_task_rows))
y_row = y_task + task_sec_h - 0.50
for inp, out, label, in_col, out_col in task_rows_reversed:
    in_H, in_W = len(inp), len(inp[0])
    gh_in = in_H * TASK_C
    gw_in = in_W * TASK_C
    gw_out, gh_out = grid_display_size(out, GH)

    y_bot_row = y_row - gh_in
    y_out_bot = y_bot_row + (gh_in - gh_out) / 2

    ftxt(x0 + gw_in/2, y_row + 0.03,
         f"{label}  input", fontsize=7, color=in_col, ha='center', va='bottom', fontweight='bold')
    ftxt(x0 + gw_in + ARROW_IN + gw_out/2, y_out_bot + gh_out + 0.03,
         "output" if "Example" in label else "target output",
         fontsize=7, color=out_col, ha='center', va='bottom', fontweight='bold')

    ax_in,  _, _ = add_grid(x0,                    y_bot_row, inp, gh_in, border=in_col,  lw=1.0)
    ax_out, _, _ = add_grid(x0 + gw_in + ARROW_IN, y_out_bot, out, GH,   border=out_col, lw=1.5)
    connect(ax_in, ax_out, in_col)

    y_row = y_bot_row - ROW_PAD - 0.32

# ══════════════════════════════════════════════════════════════════════════════
# Section 2: Human Success Traces
# ══════════════════════════════════════════════════════════════════════════════

ftxt(x0, y_human + human_sec_h - 0.04,
     f"Human success traces — all attempts  (n={len(human_participants)} participants)",
     fontsize=9, fontweight='bold', color='#CC6600', va='top')

# Build y_block_bot for each participant (first participant at top = highest y).
# Stack upward: last participant sits at y_human+0.05, first at the highest y.
_y_block_bots: list = []
_y_cursor = y_human + 0.05
for _p in reversed(human_participants):
    _y_block_bots.insert(0, _y_cursor)
    _y_cursor += participant_block_height(_p) + BETWEEN_PART

for p_idx, (p, y_bot_p) in enumerate(zip(human_participants, _y_block_bots)):
    n_att   = len(p["attempts"])
    n_grids = sum(len(a["grids"]) for a in p["attempts"])
    bh      = participant_block_height(p)
    # Participant info label: sit 0.22" above the block top so it clears the
    # "Attempt X" labels (which are 0.04" above the top-line grids).
    ftxt(x0, y_bot_p + bh + 0.22,
         f"P{p_idx+1} — {n_att} attempt{'s' if n_att > 1 else ''}, {n_grids} grids",
         fontsize=7.5, color='#CC6600', va='bottom', ha='left', fontweight='bold')
    draw_human_row_multi_attempt(x0, y_bot_p, p, GH)

# ══════════════════════════════════════════════════════════════════════════════
# Section 3: CodeIt Success Traces — every DSL step shown
# ══════════════════════════════════════════════════════════════════════════════

if codeit_seqs:
    n_c = len(codeit_seqs)
    ftxt(x0, y_codeit + n_c * codeit_row_stride + 0.08,
         f"CodeIt success traces  (n={n_c})  —  dashed boxes = non-grid intermediate states",
         fontsize=9, fontweight='bold', color='#0074D9', va='bottom')

    for c_idx, (seq, seed, full_steps) in enumerate(
            zip(codeit_seqs, codeit_seeds, codeit_full_steps)):
        rank      = n_c - 1 - c_idx
        y_row_bot = y_codeit + rank * codeit_row_stride
        ftxt(x0 - 0.08, y_row_bot + GH/2,
             f"C{c_idx+1}", fontsize=8, color='#0074D9', va='center', ha='right')
        ftxt(x0, y_row_bot - 0.20,
             seed, fontsize=6, color='#0074D9', va='top')

        if full_steps:
            draw_codeit_row_full(x0, y_row_bot, seq[0], full_steps)
        else:
            # Fallback: draw from S03 grids without step labels
            draw_row(x0, y_row_bot, seq, GH, '#0074D9', '#0074D9')

out_path = os.path.join(OUT_DIR, f"{TASK_ID}_trace_visualization.png")
plt.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out_path}")
