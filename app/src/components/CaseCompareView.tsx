import { Fragment, useMemo, useState } from "react";
import { motion } from "motion/react";
import type { Workspace } from "../lib/dataStore";
import { groupOf, normalizeArgs } from "../lib/paramGroups";
import { colorFor, type RunColor } from "../lib/runColors";
import { cn } from "../lib/utils";
import { ConfigSheet } from "./ConfigSheet";

interface DiffRow {
  key: string;
  values: string[];
  differ: boolean;
}

export function CaseCompareView({
  workspace,
  colors,
  caseId,
  step,
  compareIds,
}: {
  workspace: Workspace;
  colors: Map<string, RunColor>;
  caseId: string;
  step: number;
  compareIds: string[];
}) {
  const runs = useMemo(
    () => workspace.caseRuns(caseId, step, compareIds),
    [workspace, caseId, step, compareIds],
  );
  const shortNames = useMemo(() => workspace.runShortNames(compareIds), [workspace, compareIds]);
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
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-5xl px-8 pb-24 pt-7">
        <div
          className="grid gap-6"
          style={{ gridTemplateColumns: `repeat(${runs.length}, 1fr)` }}
        >
          {runs.map((r) => {
            const c = colorFor(colors, r.token);
            return (
              <figure key={r.token} className="m-0">
                <motion.div
                  className="aspect-video w-full overflow-hidden bg-ink"
                  initial={{ opacity: 0, y: 6 }}
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
                    <div className="grid h-full w-full place-items-center bg-wash text-xs text-faint">
                      no video at this step
                    </div>
                  )}
                </motion.div>
                <figcaption className="mt-2.5 flex items-baseline gap-2 text-[12.5px] text-muted">
                  <span
                    className="inline-block h-1.5 w-1.5 self-center rounded-full"
                    style={{ background: c.base }}
                  />
                  <span className="font-mono text-ink-2">
                    {shortNames.get(r.token) ?? r.experiment.name}
                  </span>
                  <span className="text-faint">step {step}</span>
                </figcaption>
              </figure>
            );
          })}
        </div>

        {validationCase && (
          <p className="serif mx-auto mt-8 max-w-3xl text-[15.5px] leading-[1.75] text-ink-2">
            <span className="mr-3 font-mono text-[13px] tabular-nums text-faint">
              {String(validationCase.index + 1).padStart(2, "0")}
            </span>
            {validationCase.caption}
          </p>
        )}

        <section className="mx-auto mt-14 max-w-3xl">
          <div className="flex items-baseline justify-between">
            <h3 className="serif m-0 text-[17px] font-medium text-ink">
              Table 1 — {onlyDiff ? "differing hyperparameters" : "full hyperparameters"}
            </h3>
            <button
              className="cursor-pointer border-0 bg-transparent p-0 text-[12px] text-muted underline decoration-hairline-2 underline-offset-4 transition-colors hover:text-ink"
              onClick={() => setOnlyDiff((v) => !v)}
            >
              {onlyDiff ? "show all" : "show differences only"}
            </button>
          </div>

          <table className="mt-4 w-full border-collapse text-[13px]">
            <thead>
              <tr className="hairline-b">
                <th className="pb-2 pr-4 text-left font-normal text-muted">parameter</th>
                {runs.map((r) => {
                  const c = colorFor(colors, r.token);
                  return (
                    <th key={r.token} className="pb-2 pr-4 text-left font-normal">
                      <span
                        className="mr-1.5 inline-block h-1.5 w-1.5 rounded-full align-middle"
                        style={{ background: c.base }}
                      />
                      <span className="font-mono text-[12px] text-ink-2">
                        {shortNames.get(r.token)}
                      </span>
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {diffGroups.map((g) => (
                <Fragment key={g.group}>
                  <tr>
                    <td colSpan={runs.length + 1} className="section-label pb-1 pt-5">
                      {g.group}
                    </td>
                  </tr>
                  {g.rows.map((row) => (
                    <tr key={row.key} className="hairline-b">
                      <td className="py-1.5 pr-4 font-mono text-[12px] text-muted">{row.key}</td>
                      {row.values.map((v, i) => (
                        <td
                          key={i}
                          className={cn(
                            "py-1.5 pr-4 font-mono text-[12px] tabular-nums",
                            row.differ ? "font-medium text-ink" : "text-ink-2",
                            v === "default" && "font-normal italic text-faint",
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
            <p className="mt-3 text-[13px] italic text-muted">
              These runs have identical training configurations.
            </p>
          )}
        </section>

        <section className="mx-auto mt-14 max-w-3xl">
          <h3 className="serif m-0 text-[17px] font-medium text-ink">
            Appendix — full configuration
          </h3>
          <div
            className="mt-4 grid gap-x-10"
            style={{ gridTemplateColumns: `repeat(${runs.length}, 1fr)` }}
          >
            {runs.map((r) => {
              const c = colorFor(colors, r.token);
              const code = r.experiment.code ?? {};
              const env = r.experiment.environment ?? {};
              return (
                <div key={r.token}>
                  <div className="hairline-b flex items-baseline gap-2 pb-2 text-[12px]">
                    <span
                      className="inline-block h-1.5 w-1.5 self-center rounded-full"
                      style={{ background: c.base }}
                    />
                    <span className="font-mono text-ink-2">{shortNames.get(r.token)}</span>
                    {typeof code.git_commit === "string" && (
                      <span className="font-mono text-faint">@{code.git_commit.slice(0, 8)}</span>
                    )}
                    {env.gpu_count != null && (
                      <span className="text-faint">
                        {String(env.gpu_count)}×{String(env.gpu ?? "").replace("NVIDIA ", "")}
                      </span>
                    )}
                  </div>
                  <ConfigSheet args={r.experiment.training_args ?? {}} />
                </div>
              );
            })}
          </div>
        </section>
      </div>
    </div>
  );
}
