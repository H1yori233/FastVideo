import { Fragment, useMemo, useState } from "react";
import { motion } from "motion/react";
import { ArrowLeft } from "lucide-react";
import type { Workspace } from "../lib/dataStore";
import { groupOf, normalizeArgs } from "../lib/paramGroups";
import { RunCard } from "./RunCard";
import { Button } from "./ui/button";
import { StepSelect } from "./ui/select";
import { cn } from "../lib/utils";

interface DiffRow {
  key: string;
  values: string[];
  differ: boolean;
}

export function CaseCompareView({
  workspace,
  caseId,
  step,
  onStep,
  onBack,
}: {
  workspace: Workspace;
  caseId: string;
  step: number;
  onStep: (s: number) => void;
  onBack: () => void;
}) {
  const steps = useMemo(() => workspace.sharedSteps(), [workspace]);
  const runs = useMemo(() => workspace.caseRuns(caseId, step), [workspace, caseId, step]);
  const shortNames = useMemo(() => workspace.runShortNames(), [workspace]);
  const diffKeys = useMemo(() => workspace.runDiffKeys(), [workspace]);
  const validationCase = workspace.doc.validation_cases.find((c) => c.id === caseId);

  const [onlyDiff, setOnlyDiff] = useState(true);

  const diffGroups = useMemo(() => {
    const argSets = runs.map((r) => normalizeArgs(r.experiment.training_args ?? {}));
    const keys = Array.from(new Set(argSets.flatMap((a) => Object.keys(a))));
    const rows: DiffRow[] = keys.map((k) => {
      const values = argSets.map((a) => a[k] ?? "default");
      return { key: k, values, differ: new Set(values).size > 1 };
    });
    const shown = rows.filter((r) => (onlyDiff ? r.differ : true));
    const byGroup = new Map<string, DiffRow[]>();
    for (const r of shown) {
      const g = groupOf(r.key);
      if (!byGroup.has(g)) byGroup.set(g, []);
      byGroup.get(g)!.push(r);
    }
    return [...byGroup.entries()].map(([group, rows]) => ({ group, rows }));
  }, [runs, onlyDiff]);

  return (
    <div className="h-full overflow-y-auto px-4 pb-10 pt-3.5">
      <div className="mb-3.5 flex items-baseline gap-3.5">
        <Button onClick={onBack}>
          <ArrowLeft className="h-3.5 w-3.5" /> all cases
        </Button>
        {validationCase && (
          <span className="flex min-w-0 flex-1 items-baseline gap-2">
            <span className="text-[13px] font-semibold text-accent">#{validationCase.index}</span>
            <span className="text-[13px] leading-relaxed text-[#c4c8d2]">{validationCase.caption}</span>
          </span>
        )}
        <StepSelect className="ml-auto shrink-0" label="step" value={step} options={steps} onChange={onStep} />
      </div>

      <div className="grid gap-3.5" style={{ gridTemplateColumns: `repeat(${runs.length}, 1fr)` }}>
        {runs.map((r) => (
          <div key={r.experiment.id} className="flex flex-col gap-1.5">
            <div className="flex items-baseline gap-2">
              <span className="text-[13px] font-semibold text-accent">
                {shortNames.get(r.experiment.id) ?? r.experiment.name}
              </span>
              <span className="text-[11px] text-muted">step {step}</span>
            </div>
            <motion.div
              className="glass aspect-video w-full overflow-hidden rounded-2xl"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.25, ease: "easeOut" }}
            >
              {r.video ? (
                <video
                  key={r.video.id}
                  className="h-full w-full object-contain"
                  src={workspace.mediaSrc(r.video)}
                  poster={workspace.posterSrc(r.video)}
                  controls
                  autoPlay
                  loop
                  muted
                  playsInline
                />
              ) : (
                <div className="grid h-full w-full place-items-center text-xs text-[#666]">
                  no video at this step
                </div>
              )}
            </motion.div>
          </div>
        ))}
      </div>

      <section className="mt-6 max-w-4xl">
        <div className="mb-2 flex items-center justify-between">
          <h3 className="m-0 text-[11px] uppercase tracking-wider text-muted">Training setup diff</h3>
          <div className="flex overflow-hidden rounded-lg border border-border">
            {(["only differences", "all params"] as const).map((label, i) => {
              const active = onlyDiff === (i === 0);
              return (
                <button
                  key={label}
                  className={cn(
                    "cursor-pointer border-0 bg-transparent px-3 py-1 text-[11px] text-muted transition-colors",
                    active && "bg-accent/12 text-accent-soft",
                  )}
                  onClick={() => setOnlyDiff(i === 0)}
                >
                  {label}
                </button>
              );
            })}
          </div>
        </div>

        <table className="w-full border-collapse text-xs">
          <thead>
            <tr>
              <th className="hairline-b px-2.5 py-1.5 text-left text-[11px] font-semibold text-muted">param</th>
              {runs.map((r) => (
                <th key={r.experiment.id} className="hairline-b px-2.5 py-1.5 text-left text-[11px] font-semibold text-muted">
                  {shortNames.get(r.experiment.id) ?? r.experiment.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {diffGroups.map((g) => (
              <Fragment key={g.group}>
                <tr>
                  <td
                    colSpan={runs.length + 1}
                    className="border-b border-border-2 px-2.5 pb-1 pt-3 text-[10px] uppercase tracking-wider text-[#707689]"
                  >
                    {g.group}
                  </td>
                </tr>
                {g.rows.map((row) => (
                  <tr key={row.key} className={cn(row.differ && "bg-accent/10")}>
                    <td className="hairline-b whitespace-nowrap px-2.5 py-1 font-mono text-muted">{row.key}</td>
                    {row.values.map((v, i) => (
                      <td
                        key={i}
                        className={cn(
                          "hairline-b break-words px-2.5 py-1 font-mono text-[11px] text-[#d6d9e0]",
                          row.differ && "text-accent-soft",
                          v === "default" && "italic text-[#8a93a8]",
                        )}
                      >
                        {v}
                      </td>
                    ))}
                  </tr>
                ))}
              </Fragment>
            ))}
          </tbody>
        </table>
        {onlyDiff && diffGroups.length === 0 && (
          <p className="mt-2 text-[11px] text-muted">these runs have identical training args</p>
        )}
      </section>

      <section className="mt-7">
        <h3 className="m-0 text-[11px] uppercase tracking-wider text-muted">Run cards</h3>
        <div className="mt-2.5 grid gap-3.5" style={{ gridTemplateColumns: `repeat(${runs.length}, 1fr)` }}>
          {runs.map((r) => (
            <RunCard
              key={r.experiment.id}
              workspace={workspace}
              experiment={r.experiment}
              variant="full"
              shortName={shortNames.get(r.experiment.id) ?? r.experiment.name}
              diffKeys={diffKeys}
            />
          ))}
        </div>
      </section>
    </div>
  );
}
