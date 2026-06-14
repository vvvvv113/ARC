"""
S08_task_panels.py

For each task in TASK_IDS, save three separate PNG panels:
  {task_id}_task_description.png  — all examples, 2 per row, input/output same scale
  {task_id}_human_traces.png      — 1 solved participant, all attempts, multi-line wrap
  {task_id}_codeit_traces.png     — first 5 CodeIt success traces, all DSL steps

Run: python3 analysis_5_2/S08_task_panels.py
"""

import os, sys, json, tarfile, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

TASK_IDS   = ["1acc24af", "34b99a2b"]
N_HUMAN    = 1     # participants to show
N_CODEIT   = 5     # CodeIt traces to show

REPO            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUMAN_TRACES_J  = os.path.join(REPO, "analysis_5_2/processed/S02_human_traces/human_traces_all.json")
CODEIT_TRACES_J = os.path.join(REPO, "analysis_5_2/processed/S03_codeit_traces/codeit_traces_3seeds.json")
HUMAN_CSV       = os.path.join(REPO, "human_data/data/data.csv")
OUT_DIR         = os.path.join(REPO, "analysis_5_2/processed/S08_state_graph_new")
os.makedirs(OUT_DIR, exist_ok=True)

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

# ── Grid helpers ──────────────────────────────────────────────────────────────

def parse_grid_str(s):
    rows = s.strip("|").split("|")
    if any("-" in row for row in rows):
        return [[int(m) for m in re.findall(r"-?\d", row)] for row in rows]
    return [[int(c) for c in row] for row in rows]

def grid_to_img(grid):
    H, W = len(grid), len(grid[0])
    img = np.zeros((H, W, 3))
    for r in range(H):
        for c in range(W):
            img[r, c] = [v/255 for v in ARC_PAL.get(grid[r][c], [128,128,128])]
    return img

def grid_wh(grid, fixed_h):
    H, W = len(grid), len(grid[0])
    return (W / H) * fixed_h, fixed_h

# ── Figure context ─────────────────────────────────────────────────────────────

class FC:
    """Thin wrapper around a matplotlib figure; holds FW/FH for inch→fraction conversion."""
    def __init__(self, W, H, dpi=130):
        self.W = W; self.H = H; self.dpi = dpi
        self.fig = plt.figure(figsize=(W, H), dpi=dpi)
        self.fig.patch.set_facecolor('white')

    def fr(self, x, y, w, h):
        return [x/self.W, y/self.H, w/self.W, h/self.H]

    def grid(self, x, y, g, fh, border='#444444', lw=1.5):
        gw, gh = grid_wh(g, fh)
        ax = self.fig.add_axes(self.fr(x, y, gw, gh))
        ax.imshow(grid_to_img(g), interpolation='nearest', aspect='auto')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(border); sp.set_linewidth(lw)
        return ax, gw, gh

    def placeholder(self, x, y, var_name, fh, pw, border='#0074D9'):
        ax = self.fig.add_axes(self.fr(x, y, pw, fh))
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor('#f4f7fc')
        for sp in ax.spines.values():
            sp.set_edgecolor(border); sp.set_linewidth(1.5); sp.set_linestyle((0,(5,3)))
        ax.text(0.5, 0.60, var_name, ha='center', va='center', fontsize=9,
                color='#0074D9', fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.28, "(non-grid)", ha='center', va='center', fontsize=6,
                color='#888888', fontstyle='italic', transform=ax.transAxes)
        return ax, pw, fh

    def arrow(self, ax_l, ax_r, color='#666666', lw=1.5):
        self.fig.add_artist(ConnectionPatch(
            xyA=(1.0, 0.5), coordsA='axes fraction', axesA=ax_l,
            xyB=(0.0, 0.5), coordsB='axes fraction', axesB=ax_r,
            arrowstyle='->', color=color, lw=lw,
            mutation_scale=13, zorder=10, shrinkA=3, shrinkB=3,
        ))

    def txt(self, x, y, s, **kw):
        self.fig.text(x/self.W, y/self.H, s, **kw)

    def save(self, path):
        plt.figure(self.fig.number)
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close(self.fig)
        print(f"Saved: {path}")

# ── DSL helpers ───────────────────────────────────────────────────────────────

sys.path.insert(0, REPO)
try:
    from codeit.policy.environment import execute_candidate_program_with_trace as _exec_trace
    _HAS_EXEC = True
except ImportError:
    _HAS_EXEC = False

def short_dsl(line_str):
    if "=" in line_str:
        return line_str.split("=", 1)[1].strip()
    return line_str.strip()

def reconstruct_full_steps(full_steps_meta, grids_parsed):
    result = []; grid_idx = 1
    for entry in full_steps_meta:
        if entry["has_grid"]:
            grid = grids_parsed[grid_idx] if grid_idx < len(grids_parsed) else None
            grid_idx += 1
        else:
            grid = None
        result.append((entry["dsl"], grid))
    return result

def load_success_programs(task_id):
    seen, progs = set(), []
    for seed_name, (tar_path, sol_file) in SEEDS.items():
        try:
            with tarfile.open(tar_path) as tf:
                data = json.load(tf.extractfile(sol_file))
            for split in ("seen_example", "task_demonstration", "test"):
                for prog, details in data.get("policy",{}).get(split,{}).get(task_id,{}).items():
                    perf = details.get("test_performance",[])
                    if perf and perf[0] is True and prog not in seen:
                        seen.add(prog); progs.append(prog)
        except Exception as e:
            print(f"  Warning: {seed_name}: {e}")
    return progs

def build_full_steps_by_exec(task_id, test_input_list):
    if not _HAS_EXEC: return {}
    test_tuple = tuple(tuple(r) for r in test_input_list)
    step_map = {}
    for prog in load_success_programs(task_id):
        try:
            _, trace = _exec_trace(prog, test_tuple)
        except Exception:
            continue
        if len(trace) < 2: continue
        grid_by_line = {lbl: [list(row) for row in g] for lbl, g in trace[1:]}
        lines = prog.strip().split("\n")
        full_steps = [(l.strip(), grid_by_line.get(l.strip())) for l in lines]
        key = "|" + "|".join("".join(str(c) for c in row) for row in trace[1][1]) + "|"
        step_map[key] = full_steps
    return step_map

# ── Human loader ──────────────────────────────────────────────────────────────

def load_all_human_attempts(task_id):
    df = pd.read_csv(HUMAN_CSV)
    task_df = df[(df["task_name"] == f"{task_id}.json") & (df["task_type"] == "evaluation")]
    participants = []
    for hid, p_df in task_df.groupby("hashed_id"):
        if not p_df["solved"].any(): continue
        attempts = []
        for att_num in sorted(p_df["attempt_number"].unique()):
            att_df = p_df[p_df["attempt_number"] == att_num].sort_values("action_id")
            grids, prev = [], None
            for g_str in att_df["test_output_grid"]:
                if g_str != prev:
                    try: grids.append(parse_grid_str(g_str)); prev = g_str
                    except: pass
            if grids:
                attempts.append({"num": int(att_num), "grids": grids,
                                  "success": bool(att_df["solved"].any())})
        if attempts:
            participants.append({"hashed_id": hid, "attempts": attempts})
    return participants

# ── Layout constants ──────────────────────────────────────────────────────────

GH            = 0.72    # trace grid fixed height
TASK_C        = 0.12    # per-cell size for task description
ARROW_IN      = 0.32    # gap between human grids
ARROW_IN_C    = 0.62    # gap between CodeIt grids
PLACEHOLDER_W = GH
MAX_PER_LINE  = 10
INNER_PAD     = 0.18    # gap between wrapped lines within a participant
ROW_PAD       = 0.28
BETWEEN_PART  = 0.48
CODEIT_ROW_PAD = 0.58
RESET_COL     = '#CC0000'
DPI           = 130
PAIRS_PER_ROW = 2       # examples per row in task description
PAIR_GAP      = 0.55    # horizontal gap between two examples in the same row

L, R, T, B = 0.80, 0.30, 0.55, 0.30  # figure margins

# ── Participant helpers ───────────────────────────────────────────────────────

def p_grid_list(p):
    items = []
    for att in p["attempts"]:
        for i, g in enumerate(att["grids"]):
            items.append((g, i == 0, att["num"]))
    return items

def p_n_lines(p):
    total = sum(len(a["grids"]) for a in p["attempts"])
    return max(1, -(-total // MAX_PER_LINE))

def p_block_h(p):
    n = p_n_lines(p)
    return n * GH + (n - 1) * INNER_PAD

def p_max_line_w(p):
    items = p_grid_list(p)
    max_w = 0.0
    for start in range(0, len(items), MAX_PER_LINE):
        chunk = items[start:start + MAX_PER_LINE]
        w = sum(grid_wh(g, GH)[0] + (ARROW_IN if j < len(chunk)-1 else 0)
                for j, (g, _, _) in enumerate(chunk))
        max_w = max(max_w, w)
    return max_w

def codeit_row_w(seq, full_steps):
    if not full_steps:
        gw, _ = grid_wh(seq[0], GH); return gw
    gw_in, _ = grid_wh(seq[0], GH)
    total = gw_in
    for _, grid in full_steps:
        total += ARROW_IN_C + (grid_wh(grid, GH)[0] if grid is not None else PLACEHOLDER_W)
    return total

# ══════════════════════════════════════════════════════════════════════════════
# Panel 1: Task Description
# ══════════════════════════════════════════════════════════════════════════════

def save_task_description(task_id, all_task_rows, out_dir):
    # Group into rows of PAIRS_PER_ROW
    pair_rows = [all_task_rows[i:i+PAIRS_PER_ROW]
                 for i in range(0, len(all_task_rows), PAIRS_PER_ROW)]

    # Per-row geometry
    def ex_geom(inp, out):
        gh_in  = len(inp)  * TASK_C; gw_in  = len(inp[0])  * TASK_C
        gh_out = len(out)  * TASK_C; gw_out = len(out[0])  * TASK_C
        row_h  = max(gh_in, gh_out)
        return gh_in, gw_in, gh_out, gw_out, row_h

    pair_geoms = []   # [(row_h, [(gh_in,gw_in,gh_out,gw_out, inp,out,label,ic,oc), ...])]
    for pair in pair_rows:
        exs = []
        max_row_h = 0.0
        for inp, out, label, ic, oc in pair:
            gh_in, gw_in, gh_out, gw_out, ex_h = ex_geom(inp, out)
            max_row_h = max(max_row_h, ex_h)
            exs.append((gh_in, gw_in, gh_out, gw_out, inp, out, label, ic, oc))
        pair_geoms.append((max_row_h, exs))

    # Max pair-row width
    def pair_row_w(exs):
        return (sum(gw_in + ARROW_IN + gw_out for gh_in,gw_in,gh_out,gw_out,*_ in exs)
                + (len(exs) - 1) * PAIR_GAP)
    max_content_w = max(pair_row_w(exs) for _, exs in pair_geoms)

    n_rows = len(pair_geoms)
    content_h = (sum(rh for rh, _ in pair_geoms)
                 + (n_rows - 1) * (ROW_PAD + 0.32))

    FW = L + max_content_w + R
    FH = T + 0.35 + content_h + B  # 0.35 for section header

    fc = FC(FW, FH)
    fc.fig.text(0.5, 0.998, f"Task {task_id} — Examples",
                fontsize=11, fontweight='bold', ha='center', va='top')
    fc.txt(L, FH - T - 0.06, "Task description",
           fontsize=9, fontweight='bold', color='#333333', va='top')

    # Draw from top down
    y_top = FH - T - 0.35   # top of content area
    y_cursor = y_top

    for r_idx, (row_h, exs) in enumerate(pair_geoms):
        y_bot = y_cursor - row_h
        x = L
        for gh_in, gw_in, gh_out, gw_out, inp, out, label, ic, oc in exs:
            y_in_bot  = y_bot + (row_h - gh_in)  / 2
            y_out_bot = y_bot + (row_h - gh_out) / 2
            out_label = "output" if "Example" in label else "target output"
            fc.txt(x + gw_in/2, y_in_bot + gh_in + 0.03,
                   f"{label}  input", fontsize=7, color=ic,
                   ha='center', va='bottom', fontweight='bold')
            fc.txt(x + gw_in + ARROW_IN + gw_out/2, y_out_bot + gh_out + 0.03,
                   out_label, fontsize=7, color=oc,
                   ha='center', va='bottom', fontweight='bold')
            ax_in,  _, _ = fc.grid(x,                    y_in_bot,  inp, gh_in,  border=ic, lw=1.0)
            ax_out, _, _ = fc.grid(x + gw_in + ARROW_IN, y_out_bot, out, gh_out, border=oc, lw=1.5)
            fc.arrow(ax_in, ax_out, ic)
            x += gw_in + ARROW_IN + gw_out + PAIR_GAP

        y_cursor = y_bot - ROW_PAD - 0.32

    fc.save(os.path.join(out_dir, f"{task_id}_task_description.png"))

# ══════════════════════════════════════════════════════════════════════════════
# Panel 2: Human Traces
# ══════════════════════════════════════════════════════════════════════════════

def save_human_panel(task_id, human_participants, total_avail, out_dir):
    if not human_participants:
        print(f"  {task_id}: no human participants, skipping human panel")
        return

    content_w = max(p_max_line_w(p) for p in human_participants)
    human_sec_h = (sum(p_block_h(p) for p in human_participants)
                   + max(0, len(human_participants) - 1) * BETWEEN_PART
                   + 0.65)

    FW = max(L + content_w + R, 10.0)
    FH = T + 0.35 + human_sec_h + B

    fc = FC(FW, FH)
    fc.fig.text(0.5, 0.998, f"Task {task_id} — Human Traces",
                fontsize=11, fontweight='bold', ha='center', va='top')
    fc.txt(L, FH - T - 0.04,
           f"Human success traces — all attempts  "
           f"(P1 of {total_avail} solved participants)",
           fontsize=9, fontweight='bold', color='#CC6600', va='top')

    # Stack participants upward from bottom margin
    y_human_base = B
    _bots = []
    cursor = y_human_base + 0.05
    for p in reversed(human_participants):
        _bots.insert(0, cursor)
        cursor += p_block_h(p) + BETWEEN_PART

    for p_idx, (p, y_bot_p) in enumerate(zip(human_participants, _bots)):
        n_att   = len(p["attempts"])
        n_grids = sum(len(a["grids"]) for a in p["attempts"])
        bh      = p_block_h(p)
        fc.txt(L, y_bot_p + bh + 0.22,
               f"P{p_idx+1} — {n_att} attempt{'s' if n_att > 1 else ''}, {n_grids} grids",
               fontsize=7.5, color='#CC6600', va='bottom', ha='left', fontweight='bold')

        # Draw all attempts with wrap
        items   = p_grid_list(p)
        n_lines = p_n_lines(p)
        for line_idx in range(n_lines):
            chunk = items[line_idx * MAX_PER_LINE : (line_idx + 1) * MAX_PER_LINE]
            line_y_bot = y_bot_p + (n_lines - 1 - line_idx) * (GH + INNER_PAD)
            x = L
            prev_ax = None
            for item_in_line, (g, is_start, att_num) in enumerate(chunk):
                gw, _ = grid_wh(g, GH)
                if is_start:
                    fc.txt(x + gw/2, line_y_bot + GH + 0.04,
                           f"Attempt {att_num}", fontsize=6.5, color=RESET_COL,
                           ha='center', va='bottom', fontweight='bold')
                    if item_in_line > 0:
                        sep_x = x - ARROW_IN / 2
                        fc.fig.add_artist(plt.Line2D(
                            [sep_x/FW, sep_x/FW],
                            [(line_y_bot - 0.04)/FH, (line_y_bot + GH + 0.04)/FH],
                            color=RESET_COL, lw=1.2, linestyle='--',
                            zorder=15, transform=fc.fig.transFigure,
                        ))
                ax, gw_actual, _ = fc.grid(x, line_y_bot, g, GH, border='#CC6600')
                if prev_ax is not None and not is_start:
                    fc.arrow(prev_ax, ax, '#CC6600')
                prev_ax = ax
                x += gw_actual + (ARROW_IN if item_in_line < len(chunk)-1 else 0)

    fc.save(os.path.join(out_dir, f"{task_id}_human_traces.png"))

# ══════════════════════════════════════════════════════════════════════════════
# Panel 3: CodeIt Traces
# ══════════════════════════════════════════════════════════════════════════════

def save_codeit_panel(task_id, codeit_seqs, codeit_seeds, codeit_full_steps,
                      total_avail, out_dir):
    if not codeit_seqs:
        print(f"  {task_id}: no CodeIt traces, skipping")
        return

    n_c = len(codeit_seqs)
    codeit_row_stride = GH + CODEIT_ROW_PAD
    content_w = max(codeit_row_w(seq, steps) for seq, steps in zip(codeit_seqs, codeit_full_steps))
    codeit_sec_h = n_c * codeit_row_stride + 0.55

    FW = max(L + content_w + R, 10.0)
    FH = T + codeit_sec_h + B

    fc = FC(FW, FH)
    fc.fig.text(0.5, 0.998, f"Task {task_id} — CodeIt Traces",
                fontsize=11, fontweight='bold', ha='center', va='top')
    fc.txt(L, FH - T - 0.04,
           f"CodeIt success traces  (n={n_c} of {total_avail})  "
           "—  dashed boxes = non-grid intermediate states",
           fontsize=9, fontweight='bold', color='#0074D9', va='top')

    y_base = B + 0.35

    for c_idx, (seq, seed, full_steps) in enumerate(
            zip(codeit_seqs, codeit_seeds, codeit_full_steps)):
        rank      = n_c - 1 - c_idx
        y_row_bot = y_base + rank * codeit_row_stride
        fc.txt(L - 0.08, y_row_bot + GH/2,
               f"C{c_idx+1}", fontsize=8, color='#0074D9', va='center', ha='right')
        fc.txt(L, y_row_bot - 0.20,
               seed, fontsize=6, color='#0074D9', va='top')

        if full_steps:
            # Draw input grid
            axs, xs, ws = [], [], []
            x = L
            ax, gw, _ = fc.grid(x, y_row_bot, seq[0], GH, border='#0074D9')
            axs.append(ax); xs.append(x); ws.append(gw)
            x += gw

            for arrow_idx, (dsl_line, grid) in enumerate(full_steps):
                x += ARROW_IN_C
                if grid is not None:
                    ax, bw, _ = fc.grid(x, y_row_bot, grid, GH, border='#0074D9')
                else:
                    var_name = dsl_line.split("=")[0].strip() if "=" in dsl_line else "?"
                    ax, bw, _ = fc.placeholder(x, y_row_bot, var_name, GH,
                                               PLACEHOLDER_W, '#0074D9')
                fc.arrow(axs[-1], ax, '#0074D9', lw=1.8)
                lbl   = short_dsl(dsl_line)
                mid_x = (xs[-1] + ws[-1] + x) / 2
                mid_y = y_row_bot + GH / 2
                y_off = +0.44 if arrow_idx % 2 == 0 else -0.44
                fc.txt(mid_x, mid_y + y_off, lbl,
                       fontsize=7.5, color='#0074D9', ha='center', va='center',
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.92),
                       zorder=20)
                axs.append(ax); xs.append(x); ws.append(bw)
                x += bw
        else:
            # Fallback: draw S03 grids without step labels
            axs = []; x = L
            for i, g in enumerate(seq):
                ax, gw, _ = fc.grid(x, y_row_bot, g, GH, border='#0074D9')
                if axs: fc.arrow(axs[-1], ax, '#0074D9')
                axs.append(ax)
                x += gw + (ARROW_IN if i < len(seq)-1 else 0)

    fc.save(os.path.join(out_dir, f"{task_id}_codeit_traces.png"))

# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════

print("Loading shared data...")
with open(HUMAN_TRACES_J) as f:
    human_raw = json.load(f)
with open(CODEIT_TRACES_J) as f:
    codeit_raw = json.load(f)

human_csv_df = pd.read_csv(HUMAN_CSV)   # load once, reuse

def load_all_human_attempts_fast(task_id):
    """Same as load_all_human_attempts but reuses the already-loaded DataFrame."""
    task_df = human_csv_df[
        (human_csv_df["task_name"] == f"{task_id}.json") &
        (human_csv_df["task_type"] == "evaluation")
    ]
    participants = []
    for hid, p_df in task_df.groupby("hashed_id"):
        if not p_df["solved"].any(): continue
        attempts = []
        for att_num in sorted(p_df["attempt_number"].unique()):
            att_df = p_df[p_df["attempt_number"] == att_num].sort_values("action_id")
            grids, prev = [], None
            for g_str in att_df["test_output_grid"]:
                if g_str != prev:
                    try: grids.append(parse_grid_str(g_str)); prev = g_str
                    except: pass
            if grids:
                attempts.append({"num": int(att_num), "grids": grids,
                                  "success": bool(att_df["solved"].any())})
        if attempts:
            participants.append({"hashed_id": hid, "attempts": attempts})
    return participants

for TASK_ID in TASK_IDS:
    print(f"\n{'='*60}")
    print(f"Processing {TASK_ID}")
    print(f"{'='*60}")

    # ── Load task JSON ────────────────────────────────────────
    with open(os.path.join(REPO, "codelt/data/evaluation", f"{TASK_ID}.json")) as f:
        task = json.load(f)
    train_exs   = task.get("training_examples", [])
    test_input  = task["test_examples"][0]["input"]
    test_target = task["test_examples"][0]["output"]

    all_task_rows = [(ex["input"], ex["output"], f"Example {i+1}", '#666666', '#2ECC40')
                     for i, ex in enumerate(train_exs)]
    all_task_rows.append((test_input, test_target, "Test", '#0074D9', '#2ECC40'))

    # ── Load human data ───────────────────────────────────────
    print("Loading human attempts...")
    all_participants = load_all_human_attempts_fast(TASK_ID)
    total_human = len(all_participants)
    participants = all_participants[:N_HUMAN]
    print(f"  {total_human} solved participants → showing {len(participants)}")
    for p in participants:
        n_g = sum(len(a["grids"]) for a in p["attempts"])
        print(f"  {p['hashed_id'][:8]}: {len(p['attempts'])} attempt(s), {n_g} grids")

    # ── Load CodeIt data ──────────────────────────────────────
    all_codeit = [t for t in codeit_raw.get(TASK_ID, []) if t.get("class") == "success"]
    total_codeit = len(all_codeit)
    codeit_traces = all_codeit[:N_CODEIT]
    codeit_seqs   = [[parse_grid_str(g) for g in t["grids"]] for t in codeit_traces]
    codeit_seeds  = [t.get("seed", "") for t in codeit_traces]
    print(f"  {total_codeit} CodeIt success traces → showing {len(codeit_traces)}")

    # Build full DSL steps
    codeit_full_steps = []
    need_exec = []
    for i, (t, seq_parsed) in enumerate(zip(codeit_traces, codeit_seqs)):
        if "full_steps" in t:
            codeit_full_steps.append(reconstruct_full_steps(t["full_steps"], seq_parsed))
        else:
            codeit_full_steps.append(None)
            need_exec.append(i)
    if need_exec:
        print(f"  Re-executing {len(need_exec)} trace(s)...")
        step_map = build_full_steps_by_exec(TASK_ID, test_input)
        for i in need_exec:
            key = codeit_traces[i]["grids"][1] if len(codeit_traces[i]["grids"]) > 1 else ""
            codeit_full_steps[i] = step_map.get(key, [])
    else:
        print("  full_steps from S03 JSON (no re-execution)")

    for i, steps in enumerate(codeit_full_steps):
        print(f"  C{i+1}: {len(steps)} DSL steps — "
              + ", ".join("grid" if g is not None else "none" for _, g in steps))

    # ── Generate 3 panels ─────────────────────────────────────
    save_task_description(TASK_ID, all_task_rows, OUT_DIR)
    save_human_panel(TASK_ID, participants, total_human, OUT_DIR)
    save_codeit_panel(TASK_ID, codeit_seqs, codeit_seeds, codeit_full_steps,
                      total_codeit, OUT_DIR)

print("\nDone — 6 panels saved.")
