import * as SelectPrimitive from "@radix-ui/react-select";
import { Check, ChevronDown } from "lucide-react";
import { cn } from "../../lib/utils";

export function StepSelect({
  value,
  options,
  onChange,
  label,
  className,
}: {
  value: number;
  options: number[];
  onChange: (v: number) => void;
  label?: string;
  className?: string;
}) {
  return (
    <label className={cn("flex items-center gap-2 text-[12px] text-muted", className)}>
      {label}
      <SelectPrimitive.Root value={String(value)} onValueChange={(v) => onChange(Number(v))}>
        <SelectPrimitive.Trigger
          className="inline-flex h-7 min-w-18 cursor-pointer items-center justify-between gap-2 rounded-sm border border-hairline-2 bg-transparent px-2.5 font-mono text-[12px] tabular-nums text-ink transition-colors hover:border-faint focus-visible:outline-1 focus-visible:outline-ink/40 data-[state=open]:border-faint"
          aria-label={label}
        >
          <SelectPrimitive.Value />
          <SelectPrimitive.Icon>
            <ChevronDown className="h-3.5 w-3.5 text-faint" />
          </SelectPrimitive.Icon>
        </SelectPrimitive.Trigger>
        <SelectPrimitive.Portal>
          <SelectPrimitive.Content
            position="popper"
            sideOffset={6}
            className="z-50 max-h-72 overflow-y-auto rounded-sm border border-hairline-2 bg-paper p-1 shadow-[0_8px_24px_-8px_rgb(22_22_15/0.18)]"
          >
            <SelectPrimitive.Viewport>
              {options.map((s) => (
                <SelectPrimitive.Item
                  key={s}
                  value={String(s)}
                  className="flex cursor-pointer items-center justify-between gap-5 rounded-[2px] px-2.5 py-1 font-mono text-[12px] tabular-nums text-ink-2 outline-none data-[highlighted]:bg-wash data-[highlighted]:text-ink"
                >
                  <SelectPrimitive.ItemText>{s}</SelectPrimitive.ItemText>
                  <SelectPrimitive.ItemIndicator>
                    <Check className="h-3 w-3 text-ink" />
                  </SelectPrimitive.ItemIndicator>
                </SelectPrimitive.Item>
              ))}
            </SelectPrimitive.Viewport>
          </SelectPrimitive.Content>
        </SelectPrimitive.Portal>
      </SelectPrimitive.Root>
    </label>
  );
}
