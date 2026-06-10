import type { Bag } from "../lib/types";
import { groupArgs } from "../lib/paramGroups";

export function ConfigSheet({ args }: { args: Bag }) {
  const grouped = groupArgs(args ?? {});
  return (
    <div>
      {grouped.map((g) => (
        <section key={g.group} className="mt-6 first:mt-0">
          <div className="sticky top-0 z-1 flex items-baseline gap-3 bg-paper pb-1.5 pt-1">
            <span className="section-label whitespace-nowrap">{g.group}</span>
            <span className="hairline-b mb-1 flex-1 self-end" />
          </div>
          <dl className="m-0">
            {g.entries.map(([k, v]) => (
              <div key={k} className="flex items-baseline justify-between gap-8 py-1">
                <dt className="shrink-0 font-mono text-[12px] text-muted">{k}</dt>
                <dd className="m-0 min-w-0 break-all text-right font-mono text-[12px] tabular-nums leading-relaxed text-ink">
                  {v}
                </dd>
              </div>
            ))}
          </dl>
        </section>
      ))}
    </div>
  );
}
