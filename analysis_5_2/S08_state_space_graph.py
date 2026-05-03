#!/usr/bin/env python3
"""
S08: State Space Graph — 交互式 HTML 可视化

对每个指定 task，构建两块面板：
  左：Human 状态空间（human_success / human_failed）
  右：CodeIt 状态空间（codeit_success / codeit_failed）

节点 = 每个唯一 grid 状态（SHA1[:8] 作 ID）
边   = 相邻步骤的状态转移（有向），粗细 = 转移次数
节点大小 = log(访问次数+1) * 6

点击节点 → popup 显示对应 ARC grid

用法：
    python3 analysis_5_2/S08_state_space_graph.py
"""

import json, hashlib, os, math
from collections import defaultdict

# ── 配置 ────────────────────────────────────────────────────────────────────
SELECTED_TASKS = [
    "bf699163",  # 类型1: 人类全成功, CodeIt几乎全失败, Group A
    "34b99a2b",  # 类型2: CodeIt全成功, 人类多数失败, Group A, 状态空间最小
    "7953d61e",  # 类型2: CodeIt全成功, 人类多数失败, Group A
    "e7639916",  # 类型3: 双方高成功率, Group B
    "1acc24af",  # 类型4: Group B, baseline=0.92, 双方成功率均低
    "32e9702f",  # 类型4: Group B, baseline=0.12, 双方成功率均中
]

TASK_DATA_DIR = "codelt/data/evaluation"
S02_PATH = "analysis_5_2/processed/S02_human_traces/human_traces_all.json"
S03_PATH = "analysis_5_2/processed/S03_codeit_traces/codeit_traces_3seeds.json"
S04_PATH = "analysis_5_2/processed/S04_curves/progress_curves_400.json"
OUT_DIR  = "analysis_5_2/processed/S08_state_space"
OUT_HTML = os.path.join(OUT_DIR, "state_space_interactive.html")

# ARC 颜色映射（官方 10 色）
ARC_COLORS = [
    "#000000",  # 0 black
    "#0074D9",  # 1 blue
    "#FF4136",  # 2 red
    "#2ECC40",  # 3 green
    "#FFDC00",  # 4 yellow
    "#AAAAAA",  # 5 gray
    "#F012BE",  # 6 fuchsia
    "#FF851B",  # 7 orange
    "#7FDBFF",  # 8 light blue
    "#870C25",  # 9 maroon
]

os.makedirs(OUT_DIR, exist_ok=True)

# ── 工具函数 ─────────────────────────────────────────────────────────────────

def grid_to_str(grid_2d):
    return "|" + "|".join("".join(str(c) for c in row) for row in grid_2d) + "|"

def gid(grid_str):
    return hashlib.sha1(grid_str.encode()).hexdigest()[:8]

def parse_grid(g_str):
    return [[int(c) for c in r] for r in g_str.split("|") if r]

def dedup_consecutive(grids):
    if not grids:
        return []
    out = [grids[0]]
    for g in grids[1:]:
        if g != out[-1]:
            out.append(g)
    return out

# ── 构建一侧面板的图数据 ─────────────────────────────────────────────────────

def build_panel(trace_list_by_class, anchor_ids):
    """
    trace_list_by_class: {"human_success": [[g1,g2,...], ...], "human_failed": [...]}
    anchor_ids: set of node IDs that should be marked (input / target)
    Returns: nodes_list, links_list
    """
    node_visits  = defaultdict(lambda: defaultdict(int))  # nid -> cls -> count
    edge_counts  = defaultdict(int)                        # (src,dst) -> count
    node_grids   = {}                                      # nid -> grid_str

    for cls, traces in trace_list_by_class.items():
        for raw_grids in traces:
            grids = dedup_consecutive(raw_grids)
            prev_id = None
            for g in grids:
                nid = gid(g)
                node_grids[nid] = g
                node_visits[nid][cls] += 1
                if prev_id is not None and prev_id != nid:
                    edge_counts[(prev_id, nid)] += 1
                prev_id = nid

    nodes = []
    for nid, visits in node_visits.items():
        groups  = sorted(visits.keys())
        total   = sum(visits.values())
        size    = max(6, math.log(total + 1) * 7)

        # 节点类型（用于颜色）
        if nid in anchor_ids:
            ntype = anchor_ids[nid]        # "input" or "target"
        elif len(groups) == 1:
            ntype = groups[0]
        else:
            ntype = "multi"

        nodes.append({
            "id":    nid,
            "grid":  node_grids[nid],
            "visits": dict(visits),
            "total": total,
            "type":  ntype,
            "size":  round(size, 1),
        })

    links = []
    for (src, dst), count in edge_counts.items():
        if src in node_visits and dst in node_visits:
            links.append({
                "source": src,
                "target": dst,
                "count":  count,
                "width":  round(max(1.0, math.sqrt(count) * 1.2), 1),
            })

    return nodes, links

# ── 主流程 ───────────────────────────────────────────────────────────────────

def main():
    print("Loading trace data...")
    with open(S02_PATH) as f:
        human_data = json.load(f)
    with open(S03_PATH) as f:
        codeit_data = json.load(f)
    with open(S04_PATH) as f:
        curves = json.load(f)

    all_task_data = []

    for task_id in SELECTED_TASKS:
        print(f"  Processing {task_id}...")

        # 读取任务基本信息
        with open(f"{TASK_DATA_DIR}/{task_id}.json") as f:
            task_json = json.load(f)
        inp_grid    = task_json["test_examples"][0]["input"]
        target_grid = task_json["test_examples"][0]["output"]
        inp_str     = grid_to_str(inp_grid)
        tgt_str     = grid_to_str(target_grid)
        inp_id      = gid(inp_str)
        tgt_id      = gid(tgt_str)

        # 元数据
        meta = curves.get(task_id, {})
        baseline_group = meta.get("baseline_group", "?")
        baseline       = meta.get("baseline", None)
        n_wrong        = meta.get("n_wrong_cells", None)

        # 人类轨迹
        h_traces  = human_data.get(task_id, [])
        h_by_cls  = defaultdict(list)
        for tr in h_traces:
            cls = "human_success" if tr["success"] else "human_failed"
            h_by_cls[cls].append(tr["grids"])

        # CodeIt 轨迹
        c_traces  = codeit_data.get(task_id, [])
        c_by_cls  = defaultdict(list)
        for tr in c_traces:
            cls = "codeit_" + tr["class"]
            c_by_cls[cls].append(tr["grids"])

        # 构建图
        # 人类面板：target 是锚点（human 的目标），input 一般不在人类轨迹里
        h_anchors = {tgt_id: "target"}
        # 若 input 也出现在人类轨迹里（34b99a2b 的情况）则也标记
        h_anchors[inp_id] = "input"

        c_anchors = {inp_id: "input", tgt_id: "target"}

        h_nodes, h_links = build_panel(dict(h_by_cls), h_anchors)
        c_nodes, c_links = build_panel(dict(c_by_cls), c_anchors)

        # 两侧共享节点统计
        h_ids   = {n["id"] for n in h_nodes}
        c_ids   = {n["id"] for n in c_nodes}
        shared  = list(h_ids & c_ids)

        inp_dims = [len(inp_grid), len(inp_grid[0])]
        tgt_dims = [len(target_grid), len(target_grid[0])]

        all_task_data.append({
            "task_id":        task_id,
            "baseline_group": baseline_group,
            "baseline":       round(baseline, 4) if baseline is not None else None,
            "n_wrong_cells":  n_wrong,
            "inp_dims":       inp_dims,
            "tgt_dims":       tgt_dims,
            "inp_id":         inp_id,
            "tgt_id":         tgt_id,
            "inp_grid_str":   inp_str,
            "tgt_grid_str":   tgt_str,
            "human": {
                "nodes": h_nodes,
                "links": h_links,
                "n_success": len(h_by_cls.get("human_success", [])),
                "n_failed":  len(h_by_cls.get("human_failed",  [])),
            },
            "codeit": {
                "nodes": c_nodes,
                "links": c_links,
                "n_success": len(c_by_cls.get("codeit_success", [])),
                "n_failed":  len(c_by_cls.get("codeit_failed",  [])),
            },
            "n_shared_nodes": len(shared),
        })

        print(f"    Human  nodes={len(h_nodes)} links={len(h_links)}")
        print(f"    CodeIt nodes={len(c_nodes)} links={len(c_links)}")
        print(f"    Shared nodes: {len(shared)}")

    # ── 生成 HTML ────────────────────────────────────────────────────────────
    data_json = json.dumps(all_task_data, separators=(",", ":"))
    arc_colors_json = json.dumps(ARC_COLORS)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ARC State Space Explorer</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f0f2f5; color: #333;
       display: flex; flex-direction: column; height: 100vh; overflow: hidden; }}

/* ── Top nav ── */
#nav {{
  display: flex; align-items: center; gap: 8px; padding: 8px 14px;
  background: #fff; border-bottom: 1px solid #dde1e7; flex-wrap: wrap; flex-shrink: 0;
}}
#nav-title {{ font-size: 13px; font-weight: 700; color: #444; margin-right: 4px; white-space: nowrap; }}
.task-btn {{
  padding: 4px 12px; border: 1px solid #c5cae9; background: #fff;
  color: #5c6bc0; cursor: pointer; border-radius: 16px; font-size: 12px;
  transition: all 0.15s;
}}
.task-btn:hover {{ background: #e8eaf6; }}
.task-btn.active {{ background: #5c6bc0; color: #fff; border-color: #5c6bc0; }}

/* ── Meta bar ── */
#meta-bar {{
  padding: 4px 14px; font-size: 11px; color: #777; background: #fafbfc;
  border-bottom: 1px solid #dde1e7; flex-shrink: 0; line-height: 1.8;
}}
#meta-bar b {{ color: #444; }}

/* ── Main area: two graph panels ── */
#panels {{
  display: flex; flex: 1; min-height: 0; gap: 0;
}}

.panel {{
  flex: 1; display: flex; flex-direction: column;
  background: #fff; border-right: 1px solid #dde1e7; min-width: 0;
}}
.panel-header {{
  padding: 6px 14px; background: #fafbfc; font-size: 12px; font-weight: 700;
  color: #555; display: flex; justify-content: space-between; align-items: center;
  border-bottom: 1px solid #eee; flex-shrink: 0;
}}
.panel-header .stats {{ font-size: 11px; color: #999; font-weight: 400; }}
.panel-body {{ flex: 1; position: relative; overflow: hidden; }}
svg.graph {{ width: 100%; height: 100%; background: #fff; }}

/* ── Bottom grid strip ── */
#grid-strip {{
  display: flex; align-items: stretch; background: #fff;
  border-top: 1px solid #dde1e7; flex-shrink: 0; height: 160px;
}}
#grid-left {{
  flex: 1; padding: 8px 16px; display: flex; gap: 16px; align-items: center;
  overflow-x: auto;
}}
#grid-right {{
  width: 1px; background: #dde1e7;
}}
#grid-info {{
  min-width: 140px; max-width: 200px; padding: 8px 14px;
  font-size: 11px; color: #888; display: flex; flex-direction: column;
  justify-content: center; line-height: 1.6; border-left: 1px solid #eee;
}}
#grid-info b {{ color: #444; font-size: 12px; }}
#grid-canvas {{ image-rendering: pixelated; border: 1px solid #ddd; }}
#grid-placeholder {{
  font-size: 12px; color: #bbb; padding: 16px;
  display: flex; align-items: center; justify-content: center;
  min-width: 120px;
}}

/* ── Legend ── */
#legend {{
  padding: 8px 16px; display: flex; gap: 14px; align-items: center;
  flex-wrap: wrap; border-left: 1px solid #eee; min-width: 260px;
}}
.leg {{ display: flex; align-items: center; gap: 5px; font-size: 11px; color: #666; }}
.leg-dot {{
  width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;
  border: 1px solid rgba(0,0,0,0.15);
}}

/* ── Graph elements ── */
.link {{ fill: none; stroke-opacity: 0.45; }}
.node circle {{ cursor: pointer; }}
.node circle:hover {{ opacity: 0.8; filter: drop-shadow(0 0 3px rgba(0,0,0,0.3)); }}
.anchor-label {{
  font-size: 8px; font-weight: 700; text-anchor: middle;
  dominant-baseline: central; pointer-events: none;
}}
</style>
</head>
<body>

<div id="nav">
  <span id="nav-title">ARC State Space</span>
</div>

<div id="meta-bar" id="task-meta">Select a task above</div>

<div id="panels">
  <div class="panel" id="panel-human">
    <div class="panel-header">
      <span>Human</span>
      <span class="stats" id="human-stats"></span>
    </div>
    <div class="panel-body"><svg class="graph" id="svg-human"></svg></div>
  </div>
  <div class="panel" id="panel-codeit">
    <div class="panel-header">
      <span>CodeIt</span>
      <span class="stats" id="codeit-stats"></span>
    </div>
    <div class="panel-body"><svg class="graph" id="svg-codeit"></svg></div>
  </div>
</div>

<div id="grid-strip">
  <div id="grid-left">
    <div id="grid-placeholder">Click a node to see its grid</div>
    <canvas id="grid-canvas" style="display:none"></canvas>
  </div>
  <div id="grid-info"></div>
  <div id="legend">
    <div class="leg"><div class="leg-dot" style="background:#0074D9"></div>Input (IN)</div>
    <div class="leg"><div class="leg-dot" style="background:#2ECC40"></div>Target (TGT)</div>
    <div class="leg"><div class="leg-dot" style="background:#FF851B"></div>Human success</div>
    <div class="leg"><div class="leg-dot" style="background:#e05252"></div>Human failed</div>
    <div class="leg"><div class="leg-dot" style="background:#4da6ff"></div>CodeIt success</div>
    <div class="leg"><div class="leg-dot" style="background:#aaa"></div>CodeIt failed</div>
    <div class="leg"><div class="leg-dot" style="background:#9c27b0"></div>Multi-group</div>
    <div class="leg" style="margin-left:8px; color:#aaa; font-size:10px">Node size = visit frequency</div>
  </div>
</div>

<script>
const ALL_DATA = {data_json};
const ARC_COLORS = {arc_colors_json};

const NODE_COLOR = {{
  input:          "#0074D9",
  target:         "#2ECC40",
  human_success:  "#FF851B",
  human_failed:   "#e05252",
  codeit_success: "#4da6ff",
  codeit_failed:  "#aaaaaa",
  multi:          "#9c27b0",
}};
const NODE_STROKE = {{
  input: "#fff", target: "#fff",
  human_success: "#c8640a", human_failed: "#a03030",
  codeit_success: "#1a78cc", codeit_failed: "#777",
  multi: "#6a0080",
}};

// ── Nav ────────────────────────────────────────────────────────────────────
const nav = document.getElementById("nav");
ALL_DATA.forEach((t, i) => {{
  const btn = document.createElement("button");
  btn.className = "task-btn";
  btn.textContent = `${{t.task_id}} (Grp ${{t.baseline_group}})`;
  btn.onclick = () => loadTask(i);
  nav.appendChild(btn);
}});

let currentSims = [];
function stopSims() {{ currentSims.forEach(s => s.stop()); currentSims = []; }}

// ── Grid render ────────────────────────────────────────────────────────────
function renderGrid(gridStr, label, visits) {{
  const rows = gridStr.split("|").filter(r => r.length > 0);
  const nrows = rows.length, ncols = rows[0].length;
  const maxSide = Math.max(nrows, ncols);
  const cellSize = Math.max(6, Math.min(Math.floor(130 / maxSide), 22));
  const W = ncols * cellSize, H = nrows * cellSize;

  const placeholder = document.getElementById("grid-placeholder");
  const canvas = document.getElementById("grid-canvas");
  placeholder.style.display = "none";
  canvas.style.display = "block";
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext("2d");

  for (let r = 0; r < nrows; r++) {{
    for (let c = 0; c < ncols; c++) {{
      const v = parseInt(rows[r][c]);
      ctx.fillStyle = ARC_COLORS[v] || "#000";
      ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
      if (cellSize >= 8) {{
        ctx.strokeStyle = "rgba(255,255,255,0.1)";
        ctx.lineWidth = 0.5;
        ctx.strokeRect(c * cellSize, r * cellSize, cellSize, cellSize);
      }}
    }}
  }}

  const info = document.getElementById("grid-info");
  const visitLines = visits
    ? Object.entries(visits).map(([k,v]) =>
        `<span style="color:${{NODE_COLOR[k]||'#888'}}">${{k.replace("human_","h.").replace("codeit_","c.")}}: ${{v}}</span>`
      ).join("<br>")
    : "";
  info.innerHTML = `<b>${{label}}</b><br>${{nrows}}×${{ncols}}<br>${{visitLines}}`;
}}

// ── BFS distances from a start node id ────────────────────────────────────
function bfsDistances(nodes, links, startId) {{
  const adj = {{}};
  nodes.forEach(n => adj[n.id] = []);
  links.forEach(l => {{
    const s = typeof l.source === "object" ? l.source.id : l.source;
    const t = typeof l.target === "object" ? l.target.id : l.target;
    if (adj[s]) adj[s].push(t);
  }});
  const dist = {{}};
  if (!(startId in adj)) return dist;
  const queue = [startId];
  dist[startId] = 0;
  while (queue.length) {{
    const cur = queue.shift();
    for (const nb of (adj[cur]||[])) {{
      if (!(nb in dist)) {{ dist[nb] = dist[cur]+1; queue.push(nb); }}
    }}
  }}
  return dist;
}}

// ── Draw one graph ─────────────────────────────────────────────────────────
function drawGraph(svgId, graphData, inp_id, tgt_id, labelMap) {{
  const container = document.getElementById(svgId).parentElement;
  const W = container.clientWidth  || 500;
  const H = container.clientHeight || 450;
  const cx = W / 2, cy = H / 2;

  const svg = d3.select(`#${{svgId}}`)
    .attr("viewBox", `0 0 ${{W}} ${{H}}`);
  svg.selectAll("*").remove();

  if (!graphData || graphData.nodes.length === 0) {{
    svg.append("text").attr("x", cx).attr("y", cy)
      .attr("text-anchor","middle").attr("fill","#bbb").attr("font-size","13")
      .text("No trace data available");
    return null;
  }}

  // Clone for simulation
  const nodes = graphData.nodes.map(d => ({{ ...d }}));
  const links = graphData.links.map(d => ({{ ...d }}));

  // BFS distances from input (or first node if input absent)
  const startId = nodes.find(n => n.id === inp_id) ? inp_id
                : (nodes[0]?.id || "");
  const dist = bfsDistances(nodes, links, startId);
  const maxDist = Math.max(...Object.values(dist), 1);

  // Radial radius: closer to center = more visited AND/OR shorter BFS distance
  const radialR = n => {{
    const d = dist[n.id] ?? maxDist;
    // Scale: 0→small radius (near center), maxDist → edge of circle
    const maxR = Math.min(cx, cy) * 0.88;
    return (d / maxDist) * maxR;
  }};

  // Pin anchors
  const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));
  if (nodeMap[inp_id])  {{ nodeMap[inp_id].fx  = cx; nodeMap[inp_id].fy  = cy; }}
  if (nodeMap[tgt_id])  {{ nodeMap[tgt_id].fx  = cx + Math.min(cx,cy)*0.88; nodeMap[tgt_id].fy = cy; }}

  // ── Simulation ──
  const sim = d3.forceSimulation(nodes)
    .force("link",    d3.forceLink(links).id(d => d.id)
                        .distance(d => {{
                          const s = typeof d.source==="object"?d.source:nodeMap[d.source];
                          const t2 = typeof d.target==="object"?d.target:nodeMap[d.target];
                          return 22 + (s?.size||5) + (t2?.size||5);
                        }})
                        .strength(0.9))
    .force("charge",  d3.forceManyBody().strength(n => -18 - n.size*1.5))
    .force("radial",  d3.forceRadial(radialR, cx, cy).strength(0.55))
    .force("collide", d3.forceCollide().radius(d => d.size + 2.5).strength(0.8))
    .alphaDecay(0.018);

  // ── Defs: arrowhead ──
  svg.append("defs").append("marker")
    .attr("id", `arr-${{svgId}}`).attr("viewBox","0 -3 6 6")
    .attr("refX",6).attr("refY",0)
    .attr("markerWidth",4).attr("markerHeight",4).attr("orient","auto")
    .append("path").attr("d","M0,-3L6,0L0,3")
    .attr("fill","#bbb");

  // ── Links ──
  const linkSel = svg.append("g")
    .selectAll("line").data(links).enter().append("line")
    .attr("class","link")
    .attr("stroke","#ccc")
    .attr("stroke-width", d => Math.max(0.6, d.width * 0.55))
    .attr("marker-end", `url(#arr-${{svgId}})`);

  // ── Nodes ──
  const nodeSel = svg.append("g")
    .selectAll("g").data(nodes).enter().append("g")
    .attr("class","node")
    .call(d3.drag()
      .on("start", (e,d) => {{ if(!e.active) sim.alphaTarget(0.25).restart(); d.fx=d.x; d.fy=d.y; }})
      .on("drag",  (e,d) => {{ d.fx=e.x; d.fy=e.y; }})
      .on("end",   (e,d) => {{
        if(!e.active) sim.alphaTarget(0);
        if(d.id!==inp_id && d.id!==tgt_id) {{ d.fx=null; d.fy=null; }}
      }})
    );

  nodeSel.append("circle")
    .attr("r",      d => d.size)
    .attr("fill",   d => NODE_COLOR[d.type]  || "#ccc")
    .attr("stroke", d => NODE_STROKE[d.type] || "#aaa")
    .attr("stroke-width", d => (d.type==="input"||d.type==="target") ? 2 : 1);

  // IN / TGT labels on anchors
  nodeSel.filter(d => d.id===inp_id || d.id===tgt_id)
    .append("text").attr("class","anchor-label")
    .attr("fill","#fff")
    .text(d => d.id===inp_id ? "IN" : "TGT");

  // Click → grid popup
  nodeSel.on("click", (e, d) => {{
    renderGrid(d.grid, labelMap[d.type] || d.type, d.visits);
    e.stopPropagation();
  }});

  nodeSel.append("title")
    .text(d => `${{d.type}} | visits ${{d.total}}\n${{Object.entries(d.visits).map(([k,v])=>k+":"+v).join(", ")}}`);

  sim.on("tick", () => {{
    linkSel
      .attr("x1", d => clamp(d.source.x, d.source.size, W-d.source.size))
      .attr("y1", d => clamp(d.source.y, d.source.size, H-d.source.size))
      .attr("x2", d => {{
        const dx = d.target.x-d.source.x, dy = d.target.y-d.source.y;
        const len = Math.sqrt(dx*dx+dy*dy)||1;
        return clamp(d.target.x - dx/len*(d.target.size+2), d.target.size, W-d.target.size);
      }})
      .attr("y2", d => {{
        const dx = d.target.x-d.source.x, dy = d.target.y-d.source.y;
        const len = Math.sqrt(dx*dx+dy*dy)||1;
        return clamp(d.target.y - dy/len*(d.target.size+2), d.target.size, H-d.target.size);
      }});
    nodeSel.attr("transform", d =>
      `translate(${{clamp(d.x||cx, d.size, W-d.size)}},${{clamp(d.y||cy, d.size, H-d.size)}})`);
  }});

  return sim;
}}

function clamp(v, lo, hi) {{ return Math.max(lo, Math.min(hi, v)); }}

// ── Load task ──────────────────────────────────────────────────────────────
function loadTask(idx) {{
  stopSims();
  document.querySelectorAll(".task-btn").forEach((b,i) => b.classList.toggle("active", i===idx));

  const t = ALL_DATA[idx];

  document.getElementById("meta-bar").innerHTML =
    `<b>${{t.task_id}}</b> &nbsp;·&nbsp; Group <b>${{t.baseline_group}}</b> ` +
    `&nbsp;·&nbsp; baseline=${{t.baseline??'?'}} &nbsp;·&nbsp; n_wrong=${{t.n_wrong_cells??'?'}} ` +
    `&nbsp;·&nbsp; grid ${{t.inp_dims[0]}}×${{t.inp_dims[1]}}→${{t.tgt_dims[0]}}×${{t.tgt_dims[1]}} ` +
    `&nbsp;·&nbsp; shared nodes: <b>${{t.n_shared_nodes}}</b>`;

  document.getElementById("human-stats").textContent =
    `success=${{t.human.n_success}}  failed=${{t.human.n_failed}}  nodes=${{t.human.nodes.length}}  edges=${{t.human.links.length}}`;
  document.getElementById("codeit-stats").textContent =
    `success=${{t.codeit.n_success}}  failed=${{t.codeit.n_failed}}  nodes=${{t.codeit.nodes.length}}  edges=${{t.codeit.links.length}}`;

  // Reset grid strip
  document.getElementById("grid-placeholder").style.display = "flex";
  document.getElementById("grid-canvas").style.display = "none";
  document.getElementById("grid-info").innerHTML = "";

  const hLabelMap = {{ human_success:"Human Success", human_failed:"Human Failed",
                        input:"Input Grid", target:"Target Grid", multi:"Multi-group" }};
  const cLabelMap = {{ codeit_success:"CodeIt Success", codeit_failed:"CodeIt Failed",
                        input:"Input Grid", target:"Target Grid", multi:"Multi-group" }};

  const simH = drawGraph("svg-human",  t.human,  t.inp_id, t.tgt_id, hLabelMap);
  const simC = drawGraph("svg-codeit", t.codeit, t.inp_id, t.tgt_id, cLabelMap);
  if (simH) currentSims.push(simH);
  if (simC) currentSims.push(simC);
}}

// Click background → reset
["svg-human","svg-codeit"].forEach(id => {{
  document.getElementById(id).addEventListener("click", () => {{
    document.getElementById("grid-placeholder").style.display = "flex";
    document.getElementById("grid-canvas").style.display = "none";
    document.getElementById("grid-info").innerHTML = "";
  }});
}});

loadTask(0);
</script>
</body>
</html>
"""

    with open(OUT_HTML, "w") as f:
        f.write(html)
    print(f"\nSaved -> {OUT_HTML}")
    print(f"Open in browser: file://{os.path.abspath(OUT_HTML)}")


if __name__ == "__main__":
    main()
