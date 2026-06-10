import { useMemo } from "react";
import { motion } from "motion/react";
import type { Workspace } from "../lib/dataStore";
import { colorFor, type RunColor } from "../lib/runColors";
import { VideoTile } from "./VideoTile";

export function FigureList({
  workspace,
  colors,
  step,
  compareIds,
  onSelectCase,
}: {
  workspace: Workspace;
  colors: Map<string, RunColor>;
  step: number;
  compareIds: string[];
  onSelectCase: (caseId: string) => void;
}) {
  const rows = useMemo(() => workspace.gridRows(step, compareIds), [workspace, step, compareIds]);
  const shortNames = useMemo(() => workspace.runShortNames(compareIds), [workspace, compareIds]);

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-5xl px-8 pb-24">
        {rows.map((row, ri) => (
          <motion.section
            key={row.case.id}
            className="hairline-b py-9"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: Math.min(ri * 0.03, 0.25), duration: 0.3, ease: "easeOut" }}
          >
            <button
              className="group block w-full cursor-pointer text-left"
              onClick={() => onSelectCase(row.case.id)}
            >
              <p className="serif m-0 mb-5 max-w-3xl text-[15px] leading-relaxed text-ink-2">
                <span className="mr-3 font-mono text-[13px] tabular-nums text-faint">
                  {String(row.case.index + 1).padStart(2, "0")}
                </span>
                <span className="transition-colors group-hover:text-ink">
                  {row.case.caption.length > 180
                    ? row.case.caption.slice(0, 178) + "…"
                    : row.case.caption}
                </span>
              </p>
              <div
                className="grid gap-5"
                style={{ gridTemplateColumns: `repeat(${row.cells.length}, 1fr)` }}
              >
                {row.cells.map((cell) => {
                  const c = colorFor(colors, cell.token);
                  return (
                    <figure key={cell.token} className="m-0">
                      <div className="aspect-video w-full overflow-hidden bg-ink transition-opacity group-hover:opacity-95 [content-visibility:auto] [contain-intrinsic-size:480px_270px]">
                        <VideoTile workspace={workspace} video={cell.video} mode="static" />
                      </div>
                      <figcaption className="mt-2 flex items-baseline gap-2 text-[12px] text-muted">
                        <span
                          className="inline-block h-1.5 w-1.5 self-center rounded-full"
                          style={{ background: c.base }}
                        />
                        <span className="font-mono">
                          {shortNames.get(cell.token) ?? cell.experiment.name}
                        </span>
                        <span className="text-faint">step {step}</span>
                      </figcaption>
                    </figure>
                  );
                })}
              </div>
            </button>
          </motion.section>
        ))}
      </div>
    </div>
  );
}
