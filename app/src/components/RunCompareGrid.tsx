import { useMemo } from "react";
import { motion } from "motion/react";
import type { Workspace } from "../lib/dataStore";
import { VideoTile } from "./VideoTile";
import { RunCard } from "./RunCard";
import { StepSelect } from "./ui/select";

function shortCaption(t: string): string {
  return t.length > 64 ? t.slice(0, 62) + "…" : t;
}

export function RunCompareGrid({
  workspace,
  step,
  onStep,
  onSelectCase,
}: {
  workspace: Workspace;
  step: number;
  onStep: (s: number) => void;
  onSelectCase: (caseId: string) => void;
}) {
  const exps = workspace.gridExperiments;
  const steps = useMemo(() => workspace.sharedSteps(), [workspace]);
  const rows = useMemo(() => workspace.gridRows(step), [workspace, step]);
  const shortNames = useMemo(() => workspace.runShortNames(), [workspace]);
  const diffKeys = useMemo(() => workspace.runDiffKeys(), [workspace]);

  return (
    <div className="flex h-full flex-col">
      <div className="hairline-b flex items-baseline gap-3.5 px-4 py-2.5">
        <span className="text-sm font-semibold tracking-tight">Compare runs</span>
        <span className="text-[11px] text-muted">rows = validation cases · columns = runs</span>
        <StepSelect className="ml-auto" label="checkpoint step" value={step} options={steps} onChange={onStep} />
      </div>

      <div className="flex-1 overflow-auto">
        <div
          className="grid items-start gap-2.5 p-4"
          style={{ gridTemplateColumns: `210px repeat(${exps.length}, minmax(200px, 1fr))` }}
        >
          <div className="sticky top-0 z-3 flex items-end bg-bg/90 px-2 py-1.5 text-[11px] uppercase tracking-wider text-muted backdrop-blur-sm">
            validation case
          </div>
          {exps.map((e) => (
            <div key={e.id} className="sticky top-0 z-2 bg-bg/90 pb-2 backdrop-blur-sm">
              <RunCard
                workspace={workspace}
                experiment={e}
                variant="compact"
                shortName={shortNames.get(e.id) ?? e.name}
                diffKeys={diffKeys}
              />
            </div>
          ))}

          {rows.map((row) => (
            <div key={row.case.id} className="contents">
              <button
                className="sticky left-0 z-1 flex cursor-pointer flex-col gap-0.5 border-r border-border bg-bg/90 px-2 py-1 text-left backdrop-blur-sm"
                onClick={() => onSelectCase(row.case.id)}
              >
                <span className="text-[11px] font-semibold text-accent">#{row.case.index}</span>
                <span className="text-[11px] leading-snug text-[#b8bcc6]" title={row.case.caption}>
                  {shortCaption(row.case.caption)}
                </span>
              </button>
              {row.cells.map((cell) => (
                <motion.button
                  key={cell.experiment.id}
                  className="aspect-video cursor-pointer overflow-hidden rounded-lg border border-border bg-black p-0 [content-visibility:auto] [contain-intrinsic-size:200px_112px]"
                  whileHover={{
                    scale: 1.025,
                    boxShadow: "0 0 0 1px rgb(110 168 254 / 0.4), 0 12px 32px -8px rgb(0 0 0 / 0.8)",
                  }}
                  whileTap={{ scale: 0.99 }}
                  transition={{ type: "spring", stiffness: 420, damping: 28 }}
                  onClick={() => onSelectCase(row.case.id)}
                >
                  <VideoTile workspace={workspace} video={cell.video} mode="static" />
                </motion.button>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
