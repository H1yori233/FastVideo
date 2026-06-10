import { useEffect, useMemo } from "react";
import { AnimatePresence, motion } from "motion/react";
import { useWorkspace } from "./lib/useWorkspace";
import { useRoute } from "./lib/useRoute";
import { buildRunColorMap, colorFor } from "./lib/runColors";
import { normalizeArgs } from "./lib/paramGroups";
import { FigureList } from "./components/FigureList";
import { CaseCompareView } from "./components/CaseCompareView";
import { RunGraph } from "./components/RunGraph";
import { StepSelect } from "./components/ui/select";

function Skeleton() {
  return (
    <div className="mx-auto max-w-5xl px-8 pt-10">
      <div className="skeleton mb-8 h-5 w-1/3 rounded-sm" />
      <div className="skeleton h-72 w-full rounded-sm" />
    </div>
  );
}

export default function App() {
  const { workspace, error } = useWorkspace();
  const { route, navigate } = useRoute();

  const view = route.view;
  const caseId = route.caseId;
  const effectiveCompare = useMemo(() => route.runs ?? [], [route.runs]);
  const effectiveStep = route.step ?? workspace?.latestStep(effectiveCompare) ?? 0;
  const colors = useMemo(() => buildRunColorMap(effectiveCompare), [effectiveCompare]);
  const shortNames = useMemo(
    () => workspace?.runShortNames(effectiveCompare),
    [workspace, effectiveCompare],
  );
  const diffKeys = useMemo(
    () => workspace?.runDiffKeys(effectiveCompare) ?? [],
    [workspace, effectiveCompare],
  );
  const currentCase = caseId
    ? workspace?.doc.validation_cases.find((c) => c.id === caseId)
    : undefined;
  const project = workspace?.doc.projects[0];
  const keyDiff = diffKeys[0];

  function toggleCompare(id: string) {
    const next = effectiveCompare.includes(id)
      ? effectiveCompare.filter((x) => x !== id)
      : [...effectiveCompare, id];
    navigate({ runs: next, step: null }, { replace: true });
  }

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key !== "Escape") return;
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) return;
      if (route.view !== "graph" || route.run) history.back();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [route.view, route.run]);

  return (
    <div className="flex h-screen flex-col">
      <header className="mx-auto w-full max-w-[1400px] px-8 pt-7">
        <div className="flex items-baseline justify-between">
          <h1 className="serif m-0 text-[22px] font-medium tracking-tight text-ink">
            Output Video Workspace
          </h1>
          <span className="font-mono text-[12px] text-faint">{project?.name}</span>
        </div>

        <div className="mt-2 flex flex-wrap items-baseline gap-x-5 gap-y-1 pb-4 text-[12.5px]">
          {view === "graph" || !effectiveCompare.length ? (
            <span className="text-muted">
              Every run, its lineage, and its configuration — select runs to compare their outputs.
            </span>
          ) : (
            workspace && (
              <span className="text-muted">
                Comparing{" "}
                {workspace.compareUnits(effectiveCompare).map((u, i) => (
                  <span key={u.token}>
                    {i > 0 && <span className="mx-1.5 text-faint">vs</span>}
                    <span
                      className="mx-1 mb-0.5 inline-block h-1.5 w-1.5 rounded-full"
                      style={{ background: colorFor(colors, u.token).base }}
                    />
                    <span className="font-mono text-[12px] text-ink-2">
                      {shortNames?.get(u.token) ?? u.rep.name}
                    </span>
                  </span>
                ))}
                {keyDiff && (
                  <span className="ml-3 text-faint">
                    differs in{" "}
                    <span className="font-mono text-[11.5px]">
                      {keyDiff}:{" "}
                      {workspace
                        .compareUnits(effectiveCompare)
                        .map((u) => normalizeArgs(u.rep.training_args ?? {})[keyDiff] ?? "default")
                        .join(" / ")}
                    </span>
                  </span>
                )}
              </span>
            )
          )}

          <span className="ml-auto flex items-baseline gap-5">
            {view !== "graph" && (
              <button
                className="cursor-pointer border-0 bg-transparent p-0 text-[12px] text-muted underline decoration-hairline-2 underline-offset-4 transition-colors hover:text-ink"
                onClick={() => navigate({ view: "graph" })}
              >
                ← Graph
              </button>
            )}
            {view === "case" && (
              <button
                className="cursor-pointer border-0 bg-transparent p-0 text-[12px] text-muted underline decoration-hairline-2 underline-offset-4 transition-colors hover:text-ink"
                onClick={() => navigate({ view: "results" })}
              >
                ← All cases
              </button>
            )}
            {view === "case" && currentCase && (
              <span className="font-mono text-[12px] tabular-nums text-muted">
                case {String(currentCase.index + 1).padStart(2, "0")}
              </span>
            )}
            {workspace && view !== "graph" && (
              <StepSelect
                label="step"
                value={effectiveStep}
                options={workspace.sharedSteps(effectiveCompare)}
                onChange={(s) => navigate({ step: s }, { replace: true })}
              />
            )}
          </span>
        </div>
        <div className="hairline-b" />
      </header>

      {workspace?.schemaStale && (
        <div className="mx-auto w-full max-w-[1400px] px-8 py-2 text-[12px] text-[#9a3412]">
          data.json 是旧版 schema（缺 training_args）— 在 app/ 目录运行{" "}
          <code className="font-mono">python3 tools/gen_demo_data.py</code> 后刷新
        </div>
      )}

      <main className="min-h-0 flex-1 overflow-hidden">
        {error ? (
          <div className="mx-auto max-w-5xl px-8 pt-10 text-[#9a3412]">Failed to load: {error}</div>
        ) : !workspace ? (
          <Skeleton />
        ) : (
          <AnimatePresence mode="wait">
            <motion.div
              key={view === "case" ? `case:${caseId}` : view}
              className="h-full"
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15, ease: "easeOut" }}
            >
              {view !== "graph" && !effectiveCompare.length ? (
                <div className="mx-auto max-w-5xl px-8 pt-12 text-[13px] italic text-muted">
                  No runs selected —{" "}
                  <button
                    className="cursor-pointer border-0 bg-transparent p-0 text-[13px] not-italic text-ink underline decoration-hairline-2 underline-offset-4"
                    onClick={() => navigate({ view: "graph" })}
                  >
                    pick runs on the graph
                  </button>
                  .
                </div>
              ) : view === "case" && caseId ? (
                <CaseCompareView
                  workspace={workspace}
                  colors={colors}
                  caseId={caseId}
                  step={effectiveStep}
                  compareIds={effectiveCompare}
                />
              ) : view === "results" ? (
                <FigureList
                  workspace={workspace}
                  colors={colors}
                  step={effectiveStep}
                  compareIds={effectiveCompare}
                  onSelectCase={(id) => navigate({ view: "case", caseId: id })}
                />
              ) : (
                <RunGraph
                  workspace={workspace}
                  colors={colors}
                  compareIds={effectiveCompare}
                  detailId={route.run}
                  onDetail={(id) => navigate({ run: id }, { replace: true })}
                  onToggleCompare={toggleCompare}
                  onOpenResults={() => navigate({ view: "results" })}
                />
              )}
            </motion.div>
          </AnimatePresence>
        )}
      </main>
    </div>
  );
}
