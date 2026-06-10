import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "../../lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-1.5 rounded-lg text-xs font-medium cursor-pointer transition-all duration-150 disabled:opacity-50 focus-visible:outline-2 focus-visible:outline-accent/60",
  {
    variants: {
      variant: {
        default: "border border-border bg-panel text-muted hover:text-fg hover:border-border-2",
        accent:
          "border border-accent/40 bg-accent/12 text-accent-soft hover:bg-accent/20 hover:shadow-[0_0_16px_-6px_rgb(110_168_254/0.6)]",
        ghost: "text-muted hover:text-fg hover:bg-white/5",
      },
      size: {
        sm: "h-7 px-3",
        md: "h-8 px-4",
        icon: "h-7 w-7",
      },
    },
    defaultVariants: { variant: "default", size: "sm" },
  },
);

export function Button({
  className,
  variant,
  size,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & VariantProps<typeof buttonVariants>) {
  return <button className={cn(buttonVariants({ variant, size }), className)} {...props} />;
}
