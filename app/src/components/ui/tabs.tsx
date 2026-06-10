import * as TabsPrimitive from "@radix-ui/react-tabs";
import { motion } from "motion/react";
import { cn } from "../../lib/utils";

export const Tabs = TabsPrimitive.Root;

export function TabsList({
  className,
  ...props
}: React.ComponentProps<typeof TabsPrimitive.List>) {
  return (
    <TabsPrimitive.List
      className={cn(
        "inline-flex items-center gap-1 rounded-xl border border-border bg-panel p-1",
        className,
      )}
      {...props}
    />
  );
}

export function TabsTrigger({
  className,
  active,
  children,
  ...props
}: React.ComponentProps<typeof TabsPrimitive.Trigger> & { active?: boolean }) {
  return (
    <TabsPrimitive.Trigger
      className={cn(
        "relative rounded-lg px-4 py-1.5 text-xs font-medium text-muted transition-colors cursor-pointer data-[state=active]:text-fg",
        className,
      )}
      {...props}
    >
      {active && (
        <motion.span
          layoutId="tab-pill"
          className="absolute inset-0 rounded-lg bg-accent/14 border border-accent/35"
          transition={{ type: "spring", stiffness: 500, damping: 35 }}
        />
      )}
      <span className="relative z-10">{children}</span>
    </TabsPrimitive.Trigger>
  );
}
