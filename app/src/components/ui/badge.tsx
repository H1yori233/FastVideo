import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "../../lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-[10px] leading-4 whitespace-nowrap font-medium transition-colors",
  {
    variants: {
      variant: {
        default: "border-border-2 bg-panel-2 text-[#aeb3c0]",
        accent:
          "border-accent/45 bg-accent/10 text-accent-soft shadow-[0_0_12px_-4px_rgb(110_168_254/0.5)]",
        violet: "border-violet/40 bg-violet/10 text-violet",
        mint: "border-mint/35 bg-mint/10 text-mint",
        outline: "border-border text-muted",
      },
    },
    defaultVariants: { variant: "default" },
  },
);

export function Badge({
  className,
  variant,
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & VariantProps<typeof badgeVariants>) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}
