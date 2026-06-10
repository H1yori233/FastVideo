import { useMemo } from "react";
import { GitCommitHorizontal, Cpu, CalendarClock } from "lucide-react";
import type { Workspace } from "../lib/dataStore";
import type { Experiment } from "../lib/types";
import { groupArgs, normalizeArgs, DEFAULT_OPEN_GROUPS } from "../lib/paramGroups";
import { Badge } from "./ui/badge";
import { cn } from "../lib/utils";

export function RunCard({
  workspace,
  experiment,
  variant = "compact",
  shortName,
  diffKeys = [],
}: {
  workspace: Workspace;
  experiment: Experiment;
  variant?: "compact" | "full";
  shortName: string;
  diffKeys?: string[];
}) {
  const args = useMemo(() => normalizeArgs(experiment.training_args ?? {}), [experiment]);
  const env = experiment.environment ?? {};
  const code = experiment.code ?? {};
  const baseModel = workspace.doc.base_models.find((b) => b.id === experiment.base_model_id);

  const badges = useMemo(() => {
    const b: string[] = [];
    if (baseModel?.name) b.push(baseModel.name);
    if (env.gpu_count && env.gpu) b.push(`${env.gpu_count}×${String(env.gpu).replace("NVIDIA ", "")}`);
    if (args.max_train_steps) b.push(`${args.max_train_steps} steps`);
    if (args.num_height && args.num_width) b.push(`${args.num_height}×${args.num_width}`);
    if (args.num_frames) b.push(`${args.num_frames}f`);
    if (args.generator_4bit_attn === "True" && args.generator_4bit_linear === "True")
      b.push("4-bit attn+lin");
    return b;
  }, [args, env, baseModel]);

  const grouped = useMemo(() => groupArgs(experiment.training_args ?? {}), [experiment]);

  return (
    <div className="glass flex min-w-0 flex-col gap-2.5 rounded-2xl border-t-2 border-t-accent/70 px-3.5 py-3">
      <div className="truncate text-sm font-semibold tracking-tight" title={experiment.name}>
        {shortName}
      </div>

      {variant === "full" && (
        <div className="flex flex-wrap gap-x-3.5 gap-y-1 text-[10px] text-muted">
          {typeof code.started_at === "string" && (
            <span className="inline-flex items-center gap-1">
              <CalendarClock className="h-3 w-3" />
              {code.started_at.slice(0, 16).replace("T", " ")}
            </span>
          )}
          {experiment.wandb_run_id && (
            <span className="inline-flex items-center gap-1 font-mono">
              <Cpu className="h-3 w-3" />
              wandb {experiment.wandb_run_id}
            </span>
          )}
          {typeof code.git_commit === "string" && (
            <span className="inline-flex items-center gap-1 font-mono">
              <GitCommitHorizontal className="h-3 w-3" />@{code.git_commit.slice(0, 8)}
            </span>
          )}
        </div>
      )}

      <div className="flex flex-wrap gap-1.5">
        {badges.map((b) => (
          <Badge key={b}>{b}</Badge>
        ))}
        {diffKeys.map((k) => (
          <Badge key={k} variant="accent" title="differs between runs">
            {k}
            <b className={cn("font-semibold", !(k in args) && "font-normal italic opacity-70")}>
              {args[k] ?? "default"}
            </b>
          </Badge>
        ))}
      </div>

      {variant === "full" && (
        <div className="flex flex-col gap-0.5 border-t border-border pt-2">
          {grouped.map((g) => (
            <details key={g.group} open={DEFAULT_OPEN_GROUPS.has(g.group)}>
              <summary className="cursor-pointer py-1 text-[11px] uppercase tracking-wider text-muted transition-colors open:text-[#b9bece] hover:text-[#b9bece]">
                {g.group} <span className="text-[10px] text-[#50505c]">{g.entries.length}</span>
              </summary>
              <dl className="mx-0 my-1 ml-3.5 grid grid-cols-[minmax(150px,auto)_1fr] gap-x-3 gap-y-0.5 font-mono text-[11px]">
                {g.entries.map(([k, v]) => {
                  const hl = diffKeys.includes(k);
                  return (
                    <div key={k} className="contents">
                      <dt className={cn("text-muted", hl && "rounded-sm bg-accent/10 text-accent-soft")}>
                        {k}
                      </dt>
                      <dd
                        className={cn(
                          "m-0 break-all text-[#d6d9e0]",
                          hl && "rounded-sm bg-accent/10 text-accent-soft",
                        )}
                      >
                        {v}
                      </dd>
                    </div>
                  );
                })}
              </dl>
            </details>
          ))}
          {typeof code.program === "string" && (
            <div className="truncate pt-1.5 font-mono text-[10px] text-[#6a6f7d]" title={code.program}>
              {code.program}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
