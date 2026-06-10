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
    <label className={cn("flex items-center gap-2 text-xs text-muted", className)}>
      {label}
      <SelectPrimitive.Root value={String(value)} onValueChange={(v) => onChange(Number(v))}>
        <SelectPrimitive.Trigger
          className="inline-flex h-7 min-w-20 items-center justify-between gap-2 rounded-lg border border-border bg-panel px-3 text-xs text-fg cursor-pointer transition-colors hover:border-border-2 focus-visible:outline-2 focus-visible:outline-accent/60 data-[state=open]:border-accent/50"
          aria-label={label}
        >
          <SelectPrimitive.Value />
          <SelectPrimitive.Icon>
            <ChevronDown className="h-3.5 w-3.5 text-muted" />
          </SelectPrimitive.Icon>
        </SelectPrimitive.Trigger>
        <SelectPrimitive.Portal>
          <SelectPrimitive.Content
            position="popper"
            sideOffset={6}
            className="z-50 max-h-72 overflow-y-auto rounded-xl border border-border-2 bg-panel-2/95 p-1 shadow-2xl backdrop-blur-md"
          >
            <SelectPrimitive.Viewport>
              {options.map((s) => (
                <SelectPrimitive.Item
                  key={s}
                  value={String(s)}
                  className="flex cursor-pointer items-center justify-between gap-4 rounded-lg px-3 py-1.5 text-xs text-fg outline-none data-[highlighted]:bg-accent/15 data-[highlighted]:text-accent-soft"
                >
                  <SelectPrimitive.ItemText>{s}</SelectPrimitive.ItemText>
                  <SelectPrimitive.ItemIndicator>
                    <Check className="h-3 w-3 text-accent" />
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
