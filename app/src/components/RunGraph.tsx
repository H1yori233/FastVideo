import { useEffect, useMemo, useRef, useState } from "react";
import { Maximize2, Search, X, ArrowRight } from "lucide-react";
import type { Workspace } from "../lib/dataStore";
import type { Experiment } from "../lib/types";
import { normalizeArgs } from "../lib/paramGroups";
import { colorFor, type RunColor } from "../lib/runColors";
import { Button } from "./ui/button";
import { ConfigSheet } from "./ConfigSheet";
import { cn } from "../lib/utils";

const LAYOUT_KEY = "graph-layout-v4";
const NODE_W = 286;
const NODE_H = 72;
const RES_W = 190;
const RES_H = 50;
const COL_GAP = 90;
const ROW_GAP = 26;
const LANE_GAP = 70;
const MAX_ROWS = 5;
const POSTER_W = 127;

const DS_PALETTE = ["#0e7490", "#6d28d9", "#15803d", "#a21caf", "#b45309"];

export function datasetColors(workspace: Workspace): Map<string, string> {
  const m = new Map<string, string>();
  workspace.doc.dataset_snapshots.forEach((d, i) =>
    m.set(d.id, DS_PALETTE[i % DS_PALETTE.length]),
  );
  return m;
}

interface GNode {
  id: string;
  kind: "run" | "base_model" | "dataset";
  label: string;
  full: string;
  sub: string;
  chips: string[];
  videos: number;
  poster: string;
  segments: number;
  dsId: string;
  dsName: string;
  w: number;
  h: number;
  x: number;
  y: number;
}
interface GLane {
  name: string;
  y: number;
  count: number;
}
interface GEdge {
  from: string;
  to: string;
  label: string;
  kind: "lineage" | "resource";
}

function runLabel(e: Experiment): string {
  return e.name
    .replace(/^wan_/, "")
    .replace(/_2026\d{4}_\d{4}/, "")
    .replace(/^(4bit_)?finetune_(test_)?/, "")
    .replace(/^dmd_distill_(dmd_)?/, "");
}

function startedAt(e: Experiment): string {
  const s = e.code?.started_at;
  return typeof s === "string" ? s : "";
}

function runChips(e: Experiment): string[] {
  const a = normalizeArgs(e.training_args ?? {});
  const c: string[] = [];
  if (a.max_train_steps) c.push(`${a.max_train_steps} steps`);
  if (a.learning_rate) c.push(`lr ${a.learning_rate}`);
  if (a.generator_4bit_attn === "True") c.push("4-bit");
  return c;
}

function loadSaved(): Record<string, { x: number; y: number }> {
  try {
    return JSON.parse(localStorage.getItem(LAYOUT_KEY) ?? "{}");
  } catch {
    return {};
  }
}

function isDeadRun(workspace: Workspace, e: Experiment): boolean {
  return (
    !workspace.hasVideos(e.id) &&
    !workspace.lineage.some((l) => l.from === e.id || l.to === e.id)
  );
}

interface LayoutUnit {
  id: string;
  rep: Experiment;
  members: Experiment[];
}

export function chainTokens(workspace: Workspace): Map<string, string> {
  const tokenByExp = new Map<string, string>();
  for (const chain of workspace.resumeChains()) {
    const token = chain.join("~");
    for (const id of chain) tokenByExp.set(id, token);
  }
  return tokenByExp;
}

function autoLayout(
  workspace: Workspace,
  showAll: boolean,
  mergeOn: boolean,
): { nodes: GNode[]; lanes: GLane[] } {
  const nodes: GNode[] = [];
  const lanes: GLane[] = [];
  const pipeKind = new Map(workspace.doc.pipelines.map((p) => [p.id, p.kind ?? p.name]));
  const pipeName = new Map(workspace.doc.pipelines.map((p) => [p.id, p.name]));
  const byId = new Map(workspace.experiments.map((e) => [e.id, e]));

  const exps = workspace.experiments.filter((e) => showAll || !isDeadRun(workspace, e));
  const tokenByExp = mergeOn ? chainTokens(workspace) : new Map<string, string>();
  const seen = new Set<string>();
  const units: LayoutUnit[] = [];
  for (const e of exps) {
    const token = tokenByExp.get(e.id);
    if (!token) {
      units.push({ id: e.id, rep: e, members: [e] });
      continue;
    }
    if (seen.has(token)) continue;
    seen.add(token);
    const members = token
      .split("~")
      .map((id) => byId.get(id))
      .filter((m): m is Experiment => !!m);
    units.push({ id: token, rep: members[0], members });
  }

  const byLane = new Map<string, LayoutUnit[]>();
  for (const u of units) {
    const lane = pipeKind.get(u.rep.pipeline_id) ?? "other";
    if (!byLane.has(lane)) byLane.set(lane, []);
    byLane.get(lane)!.push(u);
  }

  const initParentStart = (u: LayoutUnit): string => {
    const memberIds = new Set(u.members.map((m) => m.id));
    const edge = workspace.lineage.find(
      (l) => l.kind === "init_weights" && memberIds.has(l.to) && !memberIds.has(l.from),
    );
    const parent = edge ? byId.get(edge.from) : undefined;
    return parent ? startedAt(parent) : startedAt(u.rep);
  };

  const dedupedLabel = (lane: LayoutUnit[], u: LayoutUnit): string => {
    if (lane.length < 3) return runLabel(u.rep);
    const tokensOf = (x: LayoutUnit) => runLabel(x.rep).split(/[-_]+/).filter(Boolean);
    const freq = new Map<string, number>();
    for (const x of lane) for (const t of new Set(tokensOf(x))) freq.set(t, (freq.get(t) ?? 0) + 1);
    const kept = tokensOf(u).filter((t) => (freq.get(t) ?? 0) / lane.length < 0.7);
    return kept.length ? kept.join("-") : runLabel(u.rep);
  };

  const outlierChips = (lane: LayoutUnit[], u: LayoutUnit): string[] => {
    if (lane.length < 3) return runChips(u.rep);
    const freq = new Map<string, number>();
    for (const x of lane) for (const c of runChips(x.rep)) freq.set(c, (freq.get(c) ?? 0) + 1);
    return runChips(u.rep).filter((c) => (freq.get(c) ?? 0) / lane.length < 0.7);
  };

  const LANE_PRIORITY: Record<string, number> = { finetune: 0, dmd_distill: 1 };
  const laneOrder = [...byLane.keys()].sort(
    (a, b) => (LANE_PRIORITY[a] ?? 9) - (LANE_PRIORITY[b] ?? 9) || a.localeCompare(b),
  );
  const xStart = RES_W + 30 + COL_GAP;
  let laneTop = 56;

  for (const lane of laneOrder) {
    const list = byLane.get(lane)!;
    list.sort((a, b) => {
      const ka = `${initParentStart(a)}|${startedAt(a.rep)}`;
      const kb = `${initParentStart(b)}|${startedAt(b.rep)}`;
      return ka.localeCompare(kb) || a.rep.name.localeCompare(b.rep.name);
    });
    const rows = Math.min(MAX_ROWS, Math.max(1, list.length));
    lanes.push({
      name: pipeName.get(list[0].rep.pipeline_id) ?? lane,
      y: laneTop,
      count: list.length,
    });
    list.forEach((u, i) => {
      const col = Math.floor(i / rows);
      const row = i % rows;
      const last = u.members[u.members.length - 1];
      const videos = u.members.reduce((s, m) => s + workspace.videoCount(m.id), 0);
      const poster =
        [...u.members].reverse().map((m) => workspace.firstPoster(m.id)).find(Boolean) ?? "";
      const sub =
        u.members.length > 1
          ? `${startedAt(u.rep).slice(5, 10)} → ${startedAt(last).slice(5, 10)}`
          : startedAt(u.rep).slice(5, 16).replace("T", " ");
      const ds = workspace.doc.dataset_snapshots.find((d) => d.id === u.rep.dataset_id);
      nodes.push({
        id: u.id,
        kind: "run",
        label: dedupedLabel(list, u),
        full: u.members.map((m) => m.name).join(" → "),
        sub: `${sub}${videos > 0 ? ` · ${videos} videos` : " · no videos"}`,
        chips: outlierChips(list, u),
        videos,
        poster,
        segments: u.members.length,
        dsId: u.rep.dataset_id,
        dsName: ds?.name ?? "",
        w: NODE_W,
        h: NODE_H,
        x: xStart + col * (NODE_W + 30),
        y: laneTop + 34 + row * (NODE_H + ROW_GAP),
      });
    });
    laneTop += 34 + rows * (NODE_H + ROW_GAP) + LANE_GAP;
  }

  const resources = [
    ...workspace.doc.base_models.map((b) => ({
      id: b.id,
      kind: "base_model" as const,
      label: b.name,
      sub: "base model",
    })),
    ...workspace.doc.dataset_snapshots.map((d) => ({
      id: d.id,
      kind: "dataset" as const,
      label: d.name,
      sub: "dataset",
    })),
  ];
  resources.forEach((r, i) => {
    nodes.push({
      ...r,
      full: r.label,
      chips: [],
      videos: 0,
      poster: "",
      segments: 1,
      dsId: r.kind === "dataset" ? r.id : "",
      dsName: "",
      w: RES_W,
      h: RES_H,
      x: 30,
      y: 56 + i * (RES_H + 50),
    });
  });

  const saved = loadSaved();
  for (const n of nodes) {
    const s = saved[n.id];
    if (s) {
      n.x = s.x;
      n.y = s.y;
    }
  }
  return { nodes, lanes };
}

function buildEdges(workspace: Workspace): GEdge[] {
  const edges: GEdge[] = workspace.lineage.map((l) => ({
    from: l.from,
    to: l.to,
    label: `${l.kind === "resume" ? "resume" : "init"} ckpt-${l.step}`,
    kind: "lineage" as const,
  }));
  for (const e of workspace.experiments) {
    edges.push({ from: e.base_model_id, to: e.id, label: "base", kind: "resource" });
    edges.push({ from: e.dataset_id, to: e.id, label: "data", kind: "resource" });
  }
  return edges;
}

type Drag =
  | { mode: "pan"; sx: number; sy: number; ox: number; oy: number }
  | { mode: "node"; id: string; sx: number; sy: number; ox: number; oy: number };

export function RunGraph({
  workspace,
  colors,
  compareIds,
  detailId,
  onDetail,
  onToggleCompare,
  onOpenResults,
}: {
  workspace: Workspace;
  colors: Map<string, RunColor>;
  compareIds: string[];
  detailId: string | null;
  onDetail: (id: string | null) => void;
  onToggleCompare: (id: string) => void;
  onOpenResults: () => void;
}) {
  const [showAll, setShowAll] = useState(false);
  const [mergeOn, setMergeOn] = useState(true);
  const [layout, setLayout] = useState(() => autoLayout(workspace, false, true));
  const nodes = layout.nodes;
  const rawEdges = useMemo(() => buildEdges(workspace), [workspace]);
  const [pan, setPan] = useState({ x: 20, y: 20 });
  const [zoom, setZoom] = useState(1);
  const [query, setQuery] = useState("");
  const dragRef = useRef<Drag | null>(null);
  const canvasRef = useRef<HTMLDivElement | null>(null);

  const deadCount = useMemo(
    () => workspace.experiments.filter((e) => isDeadRun(workspace, e)).length,
    [workspace],
  );
  const chainCount = useMemo(() => workspace.resumeChains().length, [workspace]);
  const tokenByExp = useMemo(
    () => (mergeOn ? chainTokens(workspace) : new Map<string, string>()),
    [workspace, mergeOn],
  );
  const edges = useMemo(() => {
    const remap = (id: string) => tokenByExp.get(id) ?? id;
    const out: GEdge[] = [];
    const seen = new Set<string>();
    for (const e of rawEdges) {
      const from = remap(e.from);
      const to = remap(e.to);
      if (from === to) continue;
      const key = `${from}|${to}|${e.kind}|${e.label}`;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push({ ...e, from, to });
    }
    return out;
  }, [rawEdges, tokenByExp]);

  const detailUnit = useMemo(
    () => (detailId ? workspace.compareUnits([detailId])[0] : undefined),
    [workspace, detailId],
  );
  const detailExp = detailUnit?.rep;
  const nodeIds = useMemo(() => new Set(nodes.map((n) => n.id)), [nodes]);

  const focusSet = useMemo(() => {
    if (!detailId || !nodeIds.has(detailId)) return null;
    const lin = edges.filter((e) => e.kind === "lineage");
    const walk = (start: string, dir: "up" | "down") => {
      const seen = new Set([start]);
      let grew = true;
      while (grew) {
        grew = false;
        for (const e of lin) {
          if (dir === "up" && seen.has(e.to) && !seen.has(e.from)) {
            seen.add(e.from);
            grew = true;
          }
          if (dir === "down" && seen.has(e.from) && !seen.has(e.to)) {
            seen.add(e.to);
            grew = true;
          }
        }
      }
      return seen;
    };
    return new Set([...walk(detailId, "up"), ...walk(detailId, "down")]);
  }, [detailId, edges, nodeIds]);

  const showText = zoom >= 0.55;
  const showEdgeLabels = zoom >= 0.7;
  const dsColors = useMemo(() => datasetColors(workspace), [workspace]);
  const isDim = (id: string) =>
    (matched != null && !matched.has(id)) || (focusSet != null && !focusSet.has(id));

  const matched = useMemo(() => {
    if (!query.trim()) return null;
    const q = query.trim().toLowerCase();
    const byName = new Set(
      workspace.experiments
        .filter((e) => e.name.toLowerCase().includes(q))
        .map((e) => tokenByExp.get(e.id) ?? e.id),
    );
    return new Set(
      nodes.filter((n) => byName.has(n.id) || n.label.toLowerCase().includes(q)).map((n) => n.id),
    );
  }, [query, nodes, workspace, tokenByExp]);

  function setNodes(updater: (ns: GNode[]) => GNode[]) {
    setLayout((l) => ({ ...l, nodes: updater(l.nodes) }));
  }

  function fitView(ns: GNode[] = nodes) {
    const el = canvasRef.current;
    if (!el || !ns.length) return;
    const minX = Math.min(...ns.map((n) => n.x));
    const minY = Math.min(...ns.map((n) => n.y));
    const maxX = Math.max(...ns.map((n) => n.x + n.w));
    const maxY = Math.max(...ns.map((n) => n.y + n.h));
    const m = 60;
    const vw = el.clientWidth;
    const vh = el.clientHeight;
    const z = Math.min(1.4, Math.max(0.25, Math.min(vw / (maxX - minX + 2 * m), vh / (maxY - minY + 2 * m))));
    setZoom(z);
    setPan({
      x: (vw - (maxX - minX) * z) / 2 - minX * z,
      y: (vh - (maxY - minY) * z) / 2 - minY * z,
    });
  }

  function focusNode(id: string) {
    const el = canvasRef.current;
    const n = nodes.find((m) => m.id === id);
    if (!el || !n) return;
    setPan({
      x: el.clientWidth / 2 - (n.x + n.w / 2) * zoom,
      y: el.clientHeight / 2 - (n.y + n.h / 2) * zoom,
    });
  }

  useEffect(() => {
    fitView();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function toggleShowAll() {
    const next = !showAll;
    setShowAll(next);
    const fresh = autoLayout(workspace, next, mergeOn);
    setLayout(fresh);
    fitView(fresh.nodes);
  }

  function toggleMerge() {
    const next = !mergeOn;
    setMergeOn(next);
    const fresh = autoLayout(workspace, showAll, next);
    setLayout(fresh);
    fitView(fresh.nodes);
  }

  function nodeById(id: string) {
    return nodes.find((n) => n.id === id);
  }
  function edgePath(e: GEdge): string {
    const a = nodeById(e.from);
    const b = nodeById(e.to);
    if (!a || !b) return "";
    const x1 = a.x + a.w;
    const y1 = a.y + a.h / 2;
    const x2 = b.x;
    const y2 = b.y + b.h / 2;
    const mx = (x1 + x2) / 2;
    return `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`;
  }
  function edgeMid(e: GEdge): { x: number; y: number } {
    const a = nodeById(e.from);
    const b = nodeById(e.to);
    if (!a || !b) return { x: 0, y: 0 };
    return { x: (a.x + a.w + b.x) / 2, y: (a.y + a.h / 2 + b.y + b.h / 2) / 2 - 6 };
  }

  function capture(ev: React.PointerEvent) {
    try {
      (ev.currentTarget as Element).setPointerCapture(ev.pointerId);
    } catch {
      /* synthetic pointer */
    }
  }
  function onPointerDownBg(ev: React.PointerEvent) {
    dragRef.current = { mode: "pan", sx: ev.clientX, sy: ev.clientY, ox: pan.x, oy: pan.y };
    capture(ev);
  }
  function onPointerDownNode(ev: React.PointerEvent, n: GNode) {
    ev.stopPropagation();
    dragRef.current = { mode: "node", id: n.id, sx: ev.clientX, sy: ev.clientY, ox: n.x, oy: n.y };
    capture(ev);
  }
  function onPointerMove(ev: React.PointerEvent) {
    const drag = dragRef.current;
    if (!drag) return;
    if (drag.mode === "pan") {
      setPan({ x: drag.ox + ev.clientX - drag.sx, y: drag.oy + ev.clientY - drag.sy });
    } else {
      const dx = (ev.clientX - drag.sx) / zoom;
      const dy = (ev.clientY - drag.sy) / zoom;
      setNodes((ns) =>
        ns.map((n) => (n.id === drag.id ? { ...n, x: drag.ox + dx, y: drag.oy + dy } : n)),
      );
    }
  }
  function onPointerUp(ev: React.PointerEvent, n?: GNode) {
    const drag = dragRef.current;
    if (!drag) return;
    const moved = Math.hypot(ev.clientX - drag.sx, ev.clientY - drag.sy) > 4;
    if (drag.mode === "node") {
      if (moved) {
        const saved = loadSaved();
        const node = nodes.find((m) => m.id === drag.id);
        if (node) saved[node.id] = { x: node.x, y: node.y };
        localStorage.setItem(LAYOUT_KEY, JSON.stringify(saved));
      } else if (n && n.kind === "run") {
        onDetail(detailId === n.id ? null : n.id);
      }
    }
    dragRef.current = null;
  }
  function onWheel(ev: React.WheelEvent) {
    const el = canvasRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const mx = ev.clientX - rect.left;
    const my = ev.clientY - rect.top;
    const factor = ev.deltaY < 0 ? 1.1 : 0.9;
    setZoom((z) => {
      const nz = Math.min(2.5, Math.max(0.2, z * factor));
      setPan((p) => ({ x: mx - ((mx - p.x) * nz) / z, y: my - ((my - p.y) * nz) / z }));
      return nz;
    });
  }
  function resetLayout() {
    localStorage.removeItem(LAYOUT_KEY);
    const fresh = autoLayout(workspace, showAll, mergeOn);
    setLayout(fresh);
    fitView(fresh.nodes);
  }

  const selectedExps = compareIds
    .map((id) => workspace.experiments.find((e) => e.id === id))
    .filter((e): e is Experiment => !!e);

  const visibleEdges = edges.filter((e) => {
    if (!nodeIds.has(e.from) || !nodeIds.has(e.to)) return false;
    if (e.kind === "resource") return detailId != null && e.to === detailId;
    return true;
  });

  return (
    <div className="flex h-full">
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="hairline-b flex items-center gap-4 px-8 py-2">
          <label className="flex items-center gap-2 text-faint">
            <Search className="h-3.5 w-3.5" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && matched?.size) focusNode([...matched][0]);
                if (e.key === "Escape") setQuery("");
                e.stopPropagation();
              }}
              placeholder="search runs…"
              className="w-44 border-0 bg-transparent text-[12.5px] text-ink outline-none placeholder:text-faint"
            />
            {query && (
              <button className="cursor-pointer border-0 bg-transparent p-0 text-faint hover:text-ink" onClick={() => setQuery("")}>
                <X className="h-3 w-3" />
              </button>
            )}
          </label>
          <span className="text-[12px] text-faint">
            {workspace.experiments.length} runs · {workspace.lineage.length} lineage edges
          </span>
          {chainCount > 0 && (
            <label className="flex cursor-pointer items-center gap-1.5 text-[12px] text-faint">
              <input type="checkbox" checked={mergeOn} onChange={toggleMerge} className="accent-ink" />
              merge {chainCount} resume chains
            </label>
          )}
          {deadCount > 0 && (
            <label className="flex cursor-pointer items-center gap-1.5 text-[12px] text-faint">
              <input type="checkbox" checked={showAll} onChange={toggleShowAll} className="accent-ink" />
              show {deadCount} runs without videos
            </label>
          )}
          <span className="ml-auto flex gap-2">
            <Button onClick={() => fitView()}>
              <Maximize2 className="h-3 w-3" /> fit view
            </Button>
            <Button onClick={resetLayout}>reset layout</Button>
          </span>
        </div>

        <div
          ref={canvasRef}
          className="min-h-0 flex-1 cursor-grab touch-none overflow-hidden active:cursor-grabbing"
          onPointerDown={onPointerDownBg}
          onPointerMove={onPointerMove}
          onPointerUp={(e) => onPointerUp(e)}
          onWheel={onWheel}
        >
          <svg width="100%" height="100%">
            <defs>
              <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#16160f" />
              </marker>
              <marker id="arrow-dim" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#b9b7ac" />
              </marker>
            </defs>
            <g transform={`translate(${pan.x},${pan.y}) scale(${zoom})`}>
              {layout.lanes.map((lane) => (
                <g key={lane.name}>
                  <text x={RES_W + 30 + COL_GAP} y={lane.y + 10}
                    style={{ fill: "#44443c", fontSize: 15, fontFamily: "var(--font-serif)", fontWeight: 500 }}>
                    {lane.name}
                  </text>
                  <text x={RES_W + 30 + COL_GAP + 8} y={lane.y + 10} dx={lane.name.length * 8 + 16}
                    style={{ fill: "#9c9c92", fontSize: 11 }}>
                    {lane.count} runs · chronological →
                  </text>
                </g>
              ))}

              {visibleEdges.map((e) => (
                <g key={e.from + e.to + e.label} opacity={isDim(e.from) || isDim(e.to) ? 0.12 : 1}>
                  <path
                    d={edgePath(e)}
                    fill="none"
                    stroke={e.kind === "lineage" ? "#16160f" : "#c9c7bc"}
                    strokeWidth={e.kind === "lineage" ? 1.3 : 1}
                    strokeDasharray={e.kind === "resource" ? "4 4" : undefined}
                    markerEnd={e.kind === "lineage" ? "url(#arrow)" : "url(#arrow-dim)"}
                  />
                  {e.kind === "lineage" && showEdgeLabels && (
                    <text x={edgeMid(e).x} y={edgeMid(e).y} textAnchor="middle"
                      style={{ fill: "#6b6b62", fontSize: 9, fontFamily: "ui-monospace, monospace" }}>
                      {e.label}
                    </text>
                  )}
                </g>
              ))}

              {nodes.map((n) => {
                const selected = compareIds.includes(n.id);
                const rc = selected ? colorFor(colors, n.id) : null;
                const selectable = n.kind === "run" && n.videos > 0;
                return (
                  <g
                    key={n.id}
                    transform={`translate(${n.x},${n.y})`}
                    className="cursor-pointer"
                    opacity={isDim(n.id) ? 0.22 : 1}
                    onPointerDown={(e) => onPointerDownNode(e, n)}
                    onPointerUp={(e) => onPointerUp(e, n)}
                  >
                    <title>{n.full}</title>
                    <rect
                      width={n.w}
                      height={n.h}
                      rx={n.kind === "run" ? 4 : 24}
                      fill="#fffefb"
                      stroke={rc ? rc.base : detailId === n.id ? "#9c9c92" : "#dedcd2"}
                      strokeWidth={rc ? 1.5 : 1}
                    />
                    {n.kind === "run" ? (
                      <>
                        {n.poster ? (
                          <image
                            href={n.poster}
                            x={1}
                            y={1}
                            width={POSTER_W}
                            height={n.h - 2}
                            preserveAspectRatio="xMidYMid slice"
                          />
                        ) : (
                          <rect x={1} y={1} width={POSTER_W} height={n.h - 2} fill="#f1f0ea" />
                        )}
                        {n.segments > 1 && (
                          <>
                            <rect x={1} y={n.h - 19} width={26} height={18} fill="rgb(22 22 15 / 0.82)" />
                            <text x={14} y={n.h - 6} textAnchor="middle"
                              style={{ fill: "#fffefb", fontSize: 10, fontFamily: "ui-monospace, monospace" }}>
                              ×{n.segments}
                            </text>
                          </>
                        )}
                        <text x={POSTER_W + 14} y={showText ? 21 : n.h / 2 + 4}
                          style={{ fill: "#16160f", fontSize: showText ? 12 : 14, fontWeight: 500, fontFamily: "var(--font-serif)" }}>
                          {n.label.length > 22 ? n.label.slice(0, 21) + "…" : n.label}
                        </text>
                        {showText && (
                          <text x={POSTER_W + 14} y={38} style={{ fill: "#9c9c92", fontSize: 9.5 }}>
                            {n.sub}
                            {n.chips.length > 0 ? ` · ${n.chips.join(" · ")}` : ""}
                          </text>
                        )}
                        {showText && n.dsName && (
                          <>
                            <circle cx={POSTER_W + 18} cy={53} r={2.8} fill={dsColors.get(n.dsId) ?? "#9c9c92"} />
                            <text x={POSTER_W + 26} y={56.5}
                              style={{ fill: "#6b6b62", fontSize: 9.5, fontFamily: "ui-monospace, monospace" }}>
                              {n.dsName.length > 22 ? n.dsName.slice(0, 21) + "…" : n.dsName}
                            </text>
                          </>
                        )}
                        <g
                          onPointerDown={(e) => e.stopPropagation()}
                          onPointerUp={(e) => {
                            e.stopPropagation();
                            if (selectable) onToggleCompare(n.id);
                          }}
                        >
                          <circle
                            cx={n.w - 14}
                            cy={14}
                            r={6.5}
                            fill={selected && rc ? rc.base : "#fffefb"}
                            stroke={selectable ? (rc ? rc.base : "#9c9c92") : "#dedcd2"}
                            strokeWidth={1.2}
                          />
                          {selected && <circle cx={n.w - 14} cy={14} r={2.4} fill="#fffefb" />}
                          {!selectable && (
                            <line x1={n.w - 18} y1={18} x2={n.w - 10} y2={10} stroke="#dedcd2" strokeWidth={1} />
                          )}
                        </g>
                      </>
                    ) : (
                      <>
                        {n.kind === "dataset" && (
                          <circle cx={18} cy={n.h / 2} r={3.5} fill={dsColors.get(n.dsId) ?? "#9c9c92"} />
                        )}
                        <text x={n.w / 2} y={n.h / 2 + 2} textAnchor="middle"
                          style={{ fill: "#16160f", fontSize: 12.5, fontWeight: 500, fontFamily: "var(--font-serif)" }}>
                          {n.label.length > 26 ? n.label.slice(0, 24) + "…" : n.label}
                        </text>
                        <text x={n.w / 2} y={n.h / 2 + 17} textAnchor="middle" style={{ fill: "#9c9c92", fontSize: 10 }}>
                          {n.sub}
                        </text>
                      </>
                    )}
                  </g>
                );
              })}
            </g>
          </svg>
        </div>

        <div className="hairline-t flex items-center gap-4 px-8 py-2.5">
          {selectedExps.length ? (
            <>
              <span className="flex flex-wrap items-baseline gap-x-4 gap-y-1 text-[12.5px]">
                {selectedExps.map((e) => (
                  <span key={e.id} className="inline-flex items-center gap-1.5">
                    <span className="h-1.5 w-1.5 rounded-full" style={{ background: colorFor(colors, e.id).base }} />
                    <span className="font-mono text-[12px] text-ink-2">{runLabel(e)}</span>
                    <button
                      className="cursor-pointer border-0 bg-transparent p-0 text-faint hover:text-ink"
                      onClick={() => onToggleCompare(e.id)}
                      aria-label="remove"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </span>
                ))}
              </span>
              <button
                className="ml-auto inline-flex shrink-0 cursor-pointer items-center gap-1.5 border-0 bg-transparent p-0 text-[13px] text-ink underline decoration-hairline-2 underline-offset-4 hover:decoration-ink"
                onClick={onOpenResults}
              >
                Compare {selectedExps.length} {selectedExps.length === 1 ? "run" : "runs"}
                <ArrowRight className="h-3.5 w-3.5" />
              </button>
            </>
          ) : (
            <span className="text-[12px] italic text-faint">
              select runs on the graph (○) to compare their outputs
            </span>
          )}
        </div>
      </div>

      {detailExp && (
        <aside className="w-90 shrink-0 overflow-y-auto border-l border-hairline px-5 py-4">
          <div className="flex items-start justify-between gap-3">
            <h3 className="serif m-0 text-[15px] font-medium leading-snug text-ink">
              {runLabel(detailExp)}
            </h3>
            <Button size="icon" variant="ghost" onClick={() => onDetail(null)} aria-label="close">
              <X className="h-3.5 w-3.5" />
            </Button>
          </div>
          <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-faint">
            {typeof detailExp.code?.started_at === "string" && (
              <span>{detailExp.code.started_at.slice(0, 16).replace("T", " ")}</span>
            )}
            {typeof detailExp.code?.git_commit === "string" && (
              <span className="font-mono">@{detailExp.code.git_commit.slice(0, 8)}</span>
            )}
            {detailExp.environment?.gpu_count != null && (
              <span>
                {String(detailExp.environment.gpu_count)}×
                {String(detailExp.environment.gpu ?? "").replace("NVIDIA ", "")}
              </span>
            )}
          </div>

          {detailExp && (
            <div className="mt-2.5 flex flex-col gap-1 text-[11.5px]">
              <span className="text-faint">
                base{" "}
                <span className="font-mono text-ink-2">
                  {workspace.doc.base_models.find((b) => b.id === detailExp.base_model_id)?.hf_repo}
                </span>
              </span>
              <span className="flex items-center gap-1.5 text-faint">
                data
                <span
                  className="inline-block h-2 w-2 rounded-full"
                  style={{ background: dsColors.get(detailExp.dataset_id) ?? "#9c9c92" }}
                />
                <span
                  className="truncate font-mono text-ink-2"
                  title={workspace.doc.dataset_snapshots.find((d) => d.id === detailExp.dataset_id)?.path}
                >
                  {workspace.doc.dataset_snapshots.find((d) => d.id === detailExp.dataset_id)?.name}
                </span>
              </span>
            </div>
          )}

          {detailUnit && detailUnit.members.some((m) => workspace.hasVideos(m.id)) && (
            <button
              className={cn(
                "mt-3 cursor-pointer border-0 bg-transparent p-0 text-[12.5px] underline decoration-hairline-2 underline-offset-4",
                compareIds.includes(detailId!)
                  ? "text-muted hover:text-ink"
                  : "text-ink hover:decoration-ink",
              )}
              onClick={() => onToggleCompare(detailId!)}
            >
              {compareIds.includes(detailId!) ? "remove from comparison" : "add to comparison"}
            </button>
          )}

          {detailUnit && detailUnit.members.length > 1 && (
            <div className="hairline-t mt-4 pt-3">
              <div className="section-label mb-1">Segments</div>
              {detailUnit.members.map((m, i) => (
                <div key={m.id} className="py-0.5 font-mono text-[11.5px] text-ink-2">
                  {i + 1}. {m.name.replace(/^wan_/, "")}
                  <span className="ml-2 text-faint">
                    {startedAt(m).slice(5, 16).replace("T", " ")} · {workspace.videoCount(m.id)} videos
                  </span>
                </div>
              ))}
            </div>
          )}

          <div className="hairline-t mt-4 pt-3">
            <div className="section-label mb-1">Lineage</div>
            {(() => {
              const memberIds = new Set(detailUnit?.members.map((m) => m.id) ?? []);
              return workspace.lineage
                .filter(
                  (l) =>
                    (memberIds.has(l.from) || memberIds.has(l.to)) &&
                    !(memberIds.has(l.from) && memberIds.has(l.to)),
                )
                .map((l) => {
                  const otherId = memberIds.has(l.from) ? l.to : l.from;
                  const other = workspace.experiments.find((e) => e.id === otherId);
                  if (!other) return null;
                  const dir = memberIds.has(l.from) ? "→" : "←";
                  const otherToken = tokenByExp.get(otherId) ?? otherId;
                  return (
                    <button
                      key={l.from + l.to}
                      className="block cursor-pointer border-0 bg-transparent p-0 py-0.5 text-left font-mono text-[11.5px] text-ink-2 hover:text-ink"
                      onClick={() => {
                        onDetail(otherToken);
                        focusNode(otherToken);
                      }}
                    >
                      {dir} {runLabel(other)}{" "}
                      <span className="text-faint">({l.kind} ckpt-{l.step})</span>
                    </button>
                  );
                });
            })()}
          </div>

          <div className="hairline-t mt-4 pt-3">
            <ConfigSheet args={detailExp.training_args ?? {}} />
          </div>
        </aside>
      )}
    </div>
  );
}
