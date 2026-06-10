import { useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Clapperboard } from "lucide-react";
import { useWorkspace } from "./lib/useWorkspace";
import { RunCompareGrid } from "./components/RunCompareGrid";
import { CaseCompareView } from "./components/CaseCompareView";
import { RunGraph } from "./components/RunGraph";
import { Tabs, TabsList, TabsTrigger } from "./components/ui/tabs";

type View = "grid" | "graph";

export default function App() {
  const { workspace, error } = useWorkspace();
  const [view, setView] = useState<View>("grid");
  const [caseId, setCaseId] = useState<string | null>(null);
  const [step, setStep] = useState<number | null>(null);

  const effectiveStep = step ?? workspace?.latestStep() ?? 0;
  const screen = view === "graph" ? "graph" : caseId ? `case:${caseId}` : "grid";

  return (
    <div className="flex h-screen flex-col">
      <header className="hairline-b flex items-center gap-5 bg-panel/80 px-4 py-2.5 backdrop-blur-md">
        <div className="flex items-baseline gap-2.5">
          <span className="inline-flex items-center gap-2 text-sm font-bold tracking-tight">
            <Clapperboard className="h-4 w-4 text-accent" />
            Output Video Workspace
          </span>
          <span className="hidden text-[11px] text-muted sm:inline">
            a video result workspace · not a run tracker
          </span>
        </div>

        <Tabs
          value={view}
          onValueChange={(v) => {
            setView(v as View);
            if (v === "graph") setCaseId(null);
          }}
        >
          <TabsList>
            <TabsTrigger value="grid" active={view === "grid"}>
              Compare
            </TabsTrigger>
            <TabsTrigger value="graph" active={view === "graph"}>
              Graph
            </TabsTrigger>
          </TabsList>
        </Tabs>

        {workspace && (
          <div className="ml-auto text-xs tabular-nums text-muted">
            {workspace.gridExperiments.length} runs compared · {workspace.experiments.length} in graph
          </div>
        )}
      </header>

      {workspace?.schemaStale && (
        <div className="border-b border-danger/40 bg-danger/12 px-4 py-1.5 text-xs text-[#f0b6b5]">
          data.json 是旧版 schema（缺 training_args）— 在 app/ 目录运行{" "}
          <code className="font-mono text-[#ffd9d8]">python3 tools/gen_demo_data.py</code> 后刷新
        </div>
      )}

      <main className="min-h-0 flex-1 overflow-hidden">
        {error ? (
          <div className="p-10 text-danger">Failed to load: {error}</div>
        ) : !workspace ? (
          <div className="p-10 text-muted">Loading…</div>
        ) : (
          <AnimatePresence mode="wait">
            <motion.div
              key={screen}
              className="h-full"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.18, ease: "easeOut" }}
            >
              {view === "graph" ? (
                <RunGraph workspace={workspace} onCompare={() => setView("grid")} />
              ) : caseId ? (
                <CaseCompareView
                  workspace={workspace}
                  caseId={caseId}
                  step={effectiveStep}
                  onStep={setStep}
                  onBack={() => setCaseId(null)}
                />
              ) : (
                <RunCompareGrid
                  workspace={workspace}
                  step={effectiveStep}
                  onStep={setStep}
                  onSelectCase={setCaseId}
                />
              )}
            </motion.div>
          </AnimatePresence>
        )}
      </main>
    </div>
  );
}
