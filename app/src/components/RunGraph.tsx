import { useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { X } from "lucide-react";
import type { Workspace } from "../lib/dataStore";
import type { Experiment } from "../lib/types";
import { normalizeArgs } from "../lib/paramGroups";
import { RunCard } from "./RunCard";
import { Button } from "./ui/button";

const LAYOUT_KEY = "graph-layout-v1";
const NODE_W = 230;
const NODE_H = 86;
const RES_W = 190;
const RES_H = 50;
const COL_GAP = 130;
const ROW_GAP = 40;

interface GNode {
  id: string;
  kind: "run" | "base_model" | "dataset";
  label: string;
  sub: string;
  chips: string[];
  inGrid: boolean;
  w: number;
  h: number;
  x: number;
  y: number;
}
interface GEdge {
  from: string;
  to: string;
  label: string;
  kind: "lineage" | "resource";
}

function runLabel(e: Experiment): string {
  return e.name.replace(/^wan_/, "").replace(/_2026\d{4}_\d{4}/, "");
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

function autoLayout(workspace: Workspace): GNode[] {
  const nodes: GNode[] = [];
  const memo = new Map<string, number>();
  const depthOf = (id: string): number => {
    if (memo.has(id)) return memo.get(id)!;
    const parents = workspace.lineage.filter((l) => l.to === id);
    const d = parents.length ? 1 + Math.max(...parents.map((p) => depthOf(p.from))) : 0;
    memo.set(id, d);
    return d;
  };

  const byCol = new Map<number, Experiment[]>();
  for (const e of workspace.experiments) {
    const col = 1 + depthOf(e.id);
    if (!byCol.has(col)) byCol.set(col, []);
    byCol.get(col)!.push(e);
  }

  const pipe = new Map(workspace.doc.pipelines.map((p) => [p.id, p.name]));
  for (const [col, list] of byCol) {
    list.forEach((e, i) => {
      nodes.push({
        id: e.id,
        kind: "run",
        label: runLabel(e),
        sub: pipe.get(e.pipeline_id) ?? "",
        chips: runChips(e),
        inGrid: e.in_grid,
        w: NODE_W,
        h: NODE_H,
        x: col * (NODE_W + COL_GAP),
        y: 80 + i * (NODE_H + ROW_GAP),
      });
    });
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
    nodes.push({ ...r, chips: [], inGrid: false, w: RES_W, h: RES_H, x: 30, y: 40 + i * (RES_H + 60) });
  });

  const saved = loadSaved();
  for (const n of nodes) {
    const s = saved[n.id];
    if (s) {
      n.x = s.x;
      n.y = s.y;
    }
  }
  return nodes;
}

function buildEdges(workspace: Workspace): GEdge[] {
  const edges: GEdge[] = workspace.lineage.map((l) => ({
    from: l.from,
    to: l.to,
    label: `${l.kind === "resume" ? "resume" : "init"} ckpt-${l.step}`,
    kind: "lineage" as const,
  }));
  const rootRuns = new Set(
    workspace.experiments
      .filter((e) => !workspace.lineage.some((l) => l.to === e.id))
      .map((e) => e.id),
  );
  for (const e of workspace.experiments) {
    if (rootRuns.has(e.id))
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
  onCompare,
}: {
  workspace: Workspace;
  onCompare: () => void;
}) {
  const [nodes, setNodes] = useState<GNode[]>(() => autoLayout(workspace));
  const edges = useMemo(() => buildEdges(workspace), [workspace]);
  const [pan, setPan] = useState({ x: 20, y: 20 });
  const [zoom, setZoom] = useState(1);
  const [selected, setSelected] = useState<string | null>(null);
  const dragRef = useRef<Drag | null>(null);

  const selectedExp = selected
    ? workspace.experiments.find((e) => e.id === selected)
    : undefined;
  const diffKeys = useMemo(() => workspace.runDiffKeys(), [workspace]);
  const shortNames = useMemo(() => workspace.runShortNames(), [workspace]);

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
      /* synthetic or stale pointer */
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
        setSelected((s) => (s === n.id ? null : n.id));
      }
    }
    dragRef.current = null;
  }
  function onWheel(ev: React.WheelEvent) {
    const factor = ev.deltaY < 0 ? 1.1 : 0.9;
    setZoom((z) => Math.min(2.5, Math.max(0.3, z * factor)));
  }
  function resetLayout() {
    localStorage.removeItem(LAYOUT_KEY);
    setNodes(autoLayout(workspace));
    setPan({ x: 20, y: 20 });
    setZoom(1);
  }

  return (
    <div className="relative flex h-full flex-col">
      <div className="hairline-b flex items-center justify-between px-4 py-2">
        <span className="text-[11px] text-muted">
          drag canvas to pan · wheel to zoom · drag nodes · click a run for details
        </span>
        <Button onClick={resetLayout}>reset layout</Button>
      </div>

      <div
        className="flex-1 cursor-grab touch-none overflow-hidden active:cursor-grabbing"
        onPointerDown={onPointerDownBg}
        onPointerMove={onPointerMove}
        onPointerUp={(e) => onPointerUp(e)}
        onWheel={onWheel}
      >
        <svg width="100%" height="100%">
          <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#6ea8fe" />
            </marker>
            <marker id="arrow-dim" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#454552" />
            </marker>
            <linearGradient id="run-grad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#1c1c28" />
              <stop offset="100%" stopColor="#15151d" />
            </linearGradient>
          </defs>
          <g transform={`translate(${pan.x},${pan.y}) scale(${zoom})`}>
            {edges.map((e) => (
              <g key={e.from + e.to + e.label}>
                <path
                  d={edgePath(e)}
                  fill="none"
                  stroke={e.kind === "lineage" ? "#6ea8fe" : "#454552"}
                  strokeWidth={e.kind === "lineage" ? 2 : 1.2}
                  strokeDasharray={e.kind === "resource" ? "4 4" : undefined}
                  markerEnd={e.kind === "lineage" ? "url(#arrow)" : "url(#arrow-dim)"}
                />
                <text
                  x={edgeMid(e).x}
                  y={edgeMid(e).y}
                  textAnchor="middle"
                  style={{
                    fill: e.kind === "lineage" ? "#9db8e8" : "#6a6f7d",
                    fontSize: e.kind === "lineage" ? 10 : 9,
                    fontFamily: "ui-monospace, monospace",
                  }}
                >
                  {e.label}
                </text>
              </g>
            ))}

            {nodes.map((n) => (
              <g
                key={n.id}
                transform={`translate(${n.x},${n.y})`}
                className="cursor-pointer"
                onPointerDown={(e) => onPointerDownNode(e, n)}
                onPointerUp={(e) => onPointerUp(e, n)}
              >
                <rect
                  width={n.w}
                  height={n.h}
                  rx={n.kind === "run" ? 12 : 24}
                  fill={n.kind === "run" ? "url(#run-grad)" : n.kind === "base_model" ? "#181522" : "#14201a"}
                  stroke={
                    selected === n.id
                      ? "#cfe0ff"
                      : n.kind === "run" && n.inGrid
                        ? "#6ea8fe"
                        : n.kind === "base_model"
                          ? "#3a2f55"
                          : n.kind === "dataset"
                            ? "#2e4a3a"
                            : "#2b2b36"
                  }
                  strokeWidth={selected === n.id ? 2 : n.inGrid ? 1.5 : 1}
                  style={n.inGrid ? { filter: "drop-shadow(0 0 10px rgb(110 168 254 / 0.25))" } : undefined}
                />
                <text x={n.w / 2} y={n.kind === "run" ? 24 : n.h / 2 + 1} textAnchor="middle"
                  style={{ fill: "#e6e8ee", fontSize: 12, fontWeight: 600 }}>
                  {n.label.length > 30 ? n.label.slice(0, 28) + "…" : n.label}
                </text>
                {n.kind === "run" ? (
                  <>
                    <text x={n.w / 2} y={42} textAnchor="middle" style={{ fill: "#7a7a88", fontSize: 10 }}>
                      {n.sub}
                    </text>
                    <text x={n.w / 2} y={64} textAnchor="middle"
                      style={{ fill: "#9aa1b2", fontSize: 10, fontFamily: "ui-monospace, monospace" }}>
                      {n.chips.join("  ·  ")}
                    </text>
                    {n.inGrid && (
                      <text x={n.w - 10} y={14} textAnchor="end"
                        style={{ fill: "#6ea8fe", fontSize: 8, letterSpacing: "0.05em" }}>
                        IN COMPARE
                      </text>
                    )}
                  </>
                ) : (
                  <text x={n.w / 2} y={n.h / 2 + 15} textAnchor="middle" style={{ fill: "#7a7a88", fontSize: 10 }}>
                    {n.sub}
                  </text>
                )}
              </g>
            ))}
          </g>
        </svg>
      </div>

      <AnimatePresence>
        {selectedExp && (
          <motion.aside
            className="absolute bottom-3 right-3 top-12 w-[380px] overflow-y-auto rounded-2xl border border-border bg-[rgb(13_13_18/0.94)] p-3 backdrop-blur-md"
            initial={{ x: 40, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 40, opacity: 0 }}
            transition={{ type: "spring", stiffness: 380, damping: 32 }}
          >
            <div className="mb-2.5 flex items-center justify-between">
              <Button size="icon" variant="ghost" onClick={() => setSelected(null)} aria-label="close">
                <X className="h-3.5 w-3.5" />
              </Button>
              {selectedExp.in_grid && (
                <Button variant="accent" onClick={onCompare}>
                  compare videos →
                </Button>
              )}
            </div>
            <RunCard
              workspace={workspace}
              experiment={selectedExp}
              variant="full"
              shortName={shortNames.get(selectedExp.id) ?? runLabel(selectedExp)}
              diffKeys={selectedExp.in_grid ? diffKeys : []}
            />
          </motion.aside>
        )}
      </AnimatePresence>
    </div>
  );
}
