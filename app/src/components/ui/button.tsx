import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "../../lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-1.5 text-[12px] cursor-pointer transition-colors duration-150 disabled:opacity-50 focus-visible:outline-1 focus-visible:outline-ink/40",
  {
    variants: {
      variant: {
        default:
          "border border-hairline-2 bg-transparent text-muted hover:text-ink hover:border-faint rounded-sm",
        ghost: "border-0 bg-transparent text-muted hover:text-ink rounded-sm",
        link: "border-0 bg-transparent p-0 text-muted underline decoration-hairline-2 underline-offset-4 hover:text-ink",
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
