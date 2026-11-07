import * as Dialog from "@radix-ui/react-dialog";
import clsx from "clsx";
import { AlertTriangle, CheckCircle2, CircleDashed, CircleX, Clock, Inbox, Loader2, PauseCircle, ShieldAlert, X } from "lucide-react";
import { forwardRef, type ButtonHTMLAttributes, type InputHTMLAttributes, type ReactNode, type SelectHTMLAttributes, type TextareaHTMLAttributes } from "react";
import { CLAIM_LABEL, CLAIM_SHORT, titleCase } from "../lib/format";

export const Card = ({ className, children, ...p }: { className?: string; children: ReactNode } & React.HTMLAttributes<HTMLDivElement>) => (
  <div className={clsx("rounded-lg border border-line bg-surface p-4", className)} {...p}>{children}</div>
);
export const CardTitle = ({ children, right }: { children: ReactNode; right?: ReactNode }) => (
  <div className="mb-3 flex items-center justify-between gap-2"><h2 className="text-[13px] font-semibold tracking-tight text-ink">{children}</h2>{right}</div>
);
export const PageHeader = ({ title, subtitle, actions }: { title: string; subtitle?: ReactNode; actions?: ReactNode }) => (
  <div className="mb-4 flex flex-wrap items-start justify-between gap-3 border-b border-line pb-3">
    <div className="min-w-0"><h1 className="text-lg font-semibold tracking-tight text-ink">{title}</h1>{subtitle && <p className="mt-0.5 max-w-3xl text-sm text-muted">{subtitle}</p>}</div>
    {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
  </div>
);

type BtnProps = ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" | "danger" | "ghost"; busy?: boolean; size?: "sm" | "md" };
export const Button = forwardRef<HTMLButtonElement, BtnProps>(({ variant = "secondary", busy, size = "md", className, children, disabled, ...p }, ref) => (
  <button ref={ref} disabled={disabled || busy} aria-busy={busy || undefined}
    className={clsx("inline-flex items-center justify-center gap-1.5 whitespace-nowrap rounded-lg font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50",
      size === "sm" ? "h-7 px-2 text-xs" : "h-8 px-3 text-sm",
      variant === "primary" && "bg-accent text-on-accent hover:brightness-110",
      variant === "secondary" && "border border-line-strong bg-surface text-ink hover:bg-surface-2",
      variant === "danger" && "border border-bad/50 bg-surface text-bad hover:bg-bad/10",
      variant === "ghost" && "text-muted hover:bg-surface-2 hover:text-ink", className)} {...p}>
    {busy && <Loader2 className="h-4 w-4 animate-spin" aria-hidden />}{children}
  </button>
));
Button.displayName = "Button";

const fieldCls = "w-full rounded-lg border border-line-strong bg-surface px-2.5 py-1.5 text-sm text-ink placeholder:text-muted focus:border-accent aria-[invalid=true]:border-bad";
export const Field = ({ label, error, hint, children, htmlFor }: { label: string; error?: string; hint?: string; children: ReactNode; htmlFor: string }) => (
  <div className="space-y-1">
    <label htmlFor={htmlFor} className="block text-xs font-medium text-ink">{label}</label>
    {children}
    {hint && !error && <p className="text-xs text-muted" id={`${htmlFor}-hint`}>{hint}</p>}
    {error && <p className="text-xs text-bad" role="alert" id={`${htmlFor}-err`}>{error}</p>}
  </div>
);
export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(({ className, ...p }, ref) => <input ref={ref} className={clsx(fieldCls, className)} {...p} />);
Input.displayName = "Input";
export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(({ className, ...p }, ref) => <textarea ref={ref} className={clsx(fieldCls, "min-h-[72px]", className)} {...p} />);
Textarea.displayName = "Textarea";
export const Select = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement>>(({ className, ...p }, ref) => <select ref={ref} className={clsx(fieldCls, className)} {...p} />);
Select.displayName = "Select";

export const Skeleton = ({ className }: { className?: string }) => <div className={clsx("skeleton h-4 w-full", className)} aria-hidden />;
export const SkeletonBlock = ({ rows = 4, label = "Loading" }: { rows?: number; label?: string }) => (
  <div role="status" aria-live="polite" aria-label={label} className="space-y-3 p-1">
    <span className="sr-only">{label}…</span>
    {Array.from({ length: rows }).map((_, i) => <Skeleton key={i} className={i % 3 === 2 ? "w-2/3" : ""} />)}
  </div>
);

export const EmptyState = ({ title, hint, action, icon }: { title: string; hint?: ReactNode; action?: ReactNode; icon?: ReactNode }) => (
  <div className="flex flex-col items-center gap-1.5 rounded-lg border border-dashed border-line-strong px-6 py-8 text-center">
    <div className="text-muted" aria-hidden>{icon ?? <Inbox className="h-8 w-8" />}</div>
    <p className="text-sm font-medium text-ink">{title}</p>
    {hint && <p className="max-w-md text-sm text-muted">{hint}</p>}
    {action}
  </div>
);
export const ErrorState = ({ error, onRetry }: { error: unknown; onRetry?: () => void }) => (
  <div role="alert" className="flex flex-wrap items-center gap-3 rounded-lg border border-bad/40 bg-bad/5 px-4 py-3 text-sm text-bad">
    <AlertTriangle className="h-4 w-4 shrink-0" aria-hidden /><span className="flex-1">{error instanceof Error ? error.message : "Something went wrong."}</span>
    {onRetry && <Button size="sm" onClick={onRetry}>Retry</Button>}
  </div>
);
export const Notice = ({ tone = "info", children }: { tone?: "info" | "warn"; children: ReactNode }) => (
  <div role="note" className={clsx("flex gap-2 rounded-lg border px-3 py-2 text-xs", tone === "warn" ? "border-warn/40 bg-warn/5 text-ink" : "border-line bg-surface-2 text-muted")}>
    <ShieldAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden /><div>{children}</div>
  </div>
);

/** Status is always icon + text (never colour alone). */
type Tone = "ok" | "warn" | "bad" | "info" | "neutral" | "violet";
const toneCls: Record<Tone, string> = {
  ok: "border-ok/30 bg-ok/10 text-ok", warn: "border-warn/30 bg-warn/10 text-warn", bad: "border-bad/30 bg-bad/10 text-bad",
  info: "border-accent/30 bg-accent-soft text-accent", neutral: "border-line bg-surface-2 text-muted", violet: "border-violet/30 bg-violet/10 text-violet",
};
export function Badge({ tone = "neutral", icon, children, title }: { tone?: Tone; icon?: ReactNode; children: ReactNode; title?: string }) {
  return <span title={title} className={clsx("inline-flex items-center gap-1 whitespace-nowrap rounded-md border px-1.5 py-px text-[11px] font-medium", toneCls[tone])}>{icon}{children}</span>;
}
const STATUS_TONE: Record<string, [Tone, ReactNode]> = {
  succeeded: ["ok", <CheckCircle2 key="i" className="h-3 w-3" aria-hidden />], completed: ["ok", <CheckCircle2 key="i" className="h-3 w-3" aria-hidden />],
  approved: ["ok", <CheckCircle2 key="i" className="h-3 w-3" aria-hidden />],
  running: ["info", <Loader2 key="i" className="h-3 w-3 animate-spin" aria-hidden />], queued: ["neutral", <Clock key="i" className="h-3 w-3" aria-hidden />],
  pending: ["neutral", <Clock key="i" className="h-3 w-3" aria-hidden />], blocked: ["warn", <PauseCircle key="i" className="h-3 w-3" aria-hidden />],
  awaiting_approval: ["warn", <ShieldAlert key="i" className="h-3 w-3" aria-hidden />], paused: ["warn", <PauseCircle key="i" className="h-3 w-3" aria-hidden />],
  failed: ["bad", <CircleX key="i" className="h-3 w-3" aria-hidden />], error: ["bad", <CircleX key="i" className="h-3 w-3" aria-hidden />],
  timed_out: ["bad", <CircleX key="i" className="h-3 w-3" aria-hidden />], cancelled: ["neutral", <CircleX key="i" className="h-3 w-3" aria-hidden />],
  rejected: ["bad", <CircleX key="i" className="h-3 w-3" aria-hidden />], draft: ["neutral", <CircleDashed key="i" className="h-3 w-3" aria-hidden />],
};
export const StatusBadge = ({ status }: { status: string }) => {
  const [tone, icon] = STATUS_TONE[status] ?? ["neutral", <CircleDashed key="i" className="h-3 w-3" aria-hidden />];
  return <Badge tone={tone} icon={icon}>{titleCase(status)}</Badge>;
};
const CLAIM_TONE: Record<string, Tone> = { SUPPORTED: "ok", PARTIALLY_SUPPORTED: "warn", CONTRADICTED: "bad", INSUFFICIENT_EVIDENCE: "neutral", UNVERIFIED: "neutral" };
const CLAIM_ICON: Record<string, ReactNode> = {
  SUPPORTED: <CheckCircle2 className="h-3 w-3" aria-hidden />, PARTIALLY_SUPPORTED: <AlertTriangle className="h-3 w-3" aria-hidden />,
  CONTRADICTED: <CircleX className="h-3 w-3" aria-hidden />, INSUFFICIENT_EVIDENCE: <CircleDashed className="h-3 w-3" aria-hidden />, UNVERIFIED: <Clock className="h-3 w-3" aria-hidden />,
};
export const ClaimBadge = ({ status }: { status: string }) => (
  <Badge tone={CLAIM_TONE[status] ?? "neutral"} icon={CLAIM_ICON[status]} title={CLAIM_LABEL[status]}>{CLAIM_SHORT[status] ?? titleCase(status)}</Badge>
);

export function Modal({ open, onOpenChange, title, description, children }: { open: boolean; onOpenChange: (o: boolean) => void; title: string; description?: string; children: ReactNode }) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/40" />
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 max-h-[90vh] w-[min(92vw,34rem)] -translate-x-1/2 -translate-y-1/2 overflow-auto rounded-lg border border-line-strong bg-surface p-5 shadow-lg">
          <div className="mb-3 flex items-start justify-between gap-3">
            <Dialog.Title className="text-base font-semibold text-ink">{title}</Dialog.Title>
            <Dialog.Close asChild><Button variant="ghost" size="sm" aria-label="Close dialog"><X className="h-4 w-4" aria-hidden /></Button></Dialog.Close>
          </div>
          {description ? <Dialog.Description className="mb-3 text-sm text-muted">{description}</Dialog.Description> : <Dialog.Description className="sr-only">{title}</Dialog.Description>}
          {children}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
export function ConfirmDialog({ open, onOpenChange, title, description, confirmLabel = "Confirm", onConfirm, busy, danger = true }: {
  open: boolean; onOpenChange: (o: boolean) => void; title: string; description: string; confirmLabel?: string; onConfirm: () => void; busy?: boolean; danger?: boolean }) {
  return (
    <Modal open={open} onOpenChange={onOpenChange} title={title} description={description}>
      <div className="flex justify-end gap-2">
        <Button onClick={() => onOpenChange(false)}>Cancel</Button>
        <Button variant={danger ? "danger" : "primary"} busy={busy} onClick={onConfirm}>{confirmLabel}</Button>
      </div>
    </Modal>
  );
}

export function Tabs<T extends string>({ tabs, value, onChange, label }: { tabs: { id: T; label: string }[]; value: T; onChange: (t: T) => void; label: string }) {
  return (
    <div role="tablist" aria-label={label} className="mb-4 flex flex-wrap gap-1 border-b border-line">
      {tabs.map((t) => (
        <button key={t.id} role="tab" aria-selected={value === t.id} tabIndex={value === t.id ? 0 : -1} onClick={() => onChange(t.id)}
          onKeyDown={(e) => { const i = tabs.findIndex((x) => x.id === value); if (e.key === "ArrowRight") onChange(tabs[(i + 1) % tabs.length].id); if (e.key === "ArrowLeft") onChange(tabs[(i - 1 + tabs.length) % tabs.length].id); }}
          className={clsx("-mb-px border-b-2 px-2.5 py-1.5 text-sm font-medium", value === t.id ? "border-accent text-accent" : "border-transparent text-muted hover:text-ink")}>{t.label}</button>
      ))}
    </div>
  );
}

export const Table = ({ caption, head, children }: { caption: string; head: string[]; children: ReactNode }) => (
  <div className="max-h-[70vh] overflow-auto rounded-lg border border-line bg-surface">
    <table className="w-full text-left text-sm"><caption className="sr-only">{caption}</caption>
      <thead className="sticky top-0 bg-surface-2 text-[11px] uppercase tracking-wide text-muted"><tr>{head.map((h) => <th key={h} scope="col" className="px-3 py-1.5 font-medium">{h}</th>)}</tr></thead>
      <tbody className="divide-y divide-line [&_td]:px-3 [&_td]:py-1.5 [&_tr:hover]:bg-surface-2/60">{children}</tbody></table>
  </div>
);
export const KV = ({ items }: { items: [string, ReactNode][] }) => (
  <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 text-sm">
    {items.map(([k, v]) => <div key={k} className="contents"><dt className="text-xs text-muted">{k}</dt><dd className="min-w-0 break-words text-ink">{v ?? "-"}</dd></div>)}
  </dl>
);

/* ---- Toasts: transient confirmations (never the only place an error is shown) ---- */
import { createContext, useCallback, useContext, useState } from "react";
interface Toast { id: number; tone: "ok" | "bad" | "info"; text: string }
const ToastCtx = createContext<(tone: Toast["tone"], text: string) => void>(() => undefined);
export const useToast = () => useContext(ToastCtx);
export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<Toast[]>([]);
  const push = useCallback((tone: Toast["tone"], text: string) => {
    const id = Date.now() + Math.random();
    setItems((l) => [...l.slice(-3), { id, tone, text }]);
    setTimeout(() => setItems((l) => l.filter((t) => t.id !== id)), 5000);
  }, []);
  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div className="no-print pointer-events-none fixed bottom-4 right-4 z-[60] flex w-[min(92vw,22rem)] flex-col gap-2" role="status" aria-live="polite">
        {items.map((t) => (
          <div key={t.id} className={clsx("pointer-events-auto flex items-start gap-2 rounded-lg border bg-surface px-3 py-2 text-sm shadow-lg", t.tone === "ok" ? "border-ok/40" : t.tone === "bad" ? "border-bad/40" : "border-line-strong")}>
            <span className="flex-1">{t.text}</span>
            <button aria-label="Dismiss notification" className="text-muted hover:text-ink" onClick={() => setItems((l) => l.filter((x) => x.id !== t.id))}><X className="h-3.5 w-3.5" aria-hidden /></button>
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  );
}

/** Segmented control for mutually exclusive modes (e.g. search level). Radio semantics, arrow-key navigable. */
export function Segmented<T extends string>({ label, value, onChange, options }: { label: string; value: T; onChange: (v: T) => void; options: { id: T; label: string; hint?: string }[] }) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-lg border border-line-strong bg-surface-2 p-0.5">
      {options.map((o) => (
        <button key={o.id} role="radio" aria-checked={value === o.id} title={o.hint} tabIndex={value === o.id ? 0 : -1} onClick={() => onChange(o.id)}
          onKeyDown={(e) => { const i = options.findIndex((x) => x.id === value); if (e.key === "ArrowRight") onChange(options[(i + 1) % options.length].id); if (e.key === "ArrowLeft") onChange(options[(i - 1 + options.length) % options.length].id); }}
          className={clsx("rounded-md px-2.5 py-1 text-xs font-medium", value === o.id ? "bg-surface text-ink shadow-sm" : "text-muted hover:text-ink")}>{o.label}</button>
      ))}
    </div>
  );
}

/** Highlights matched query terms (case-insensitive, literal). Terms come from the server's matched_terms or the query. */
export function Highlight({ text, terms }: { text: string; terms: string[] }) {
  const clean = terms.map((t) => t.trim()).filter((t) => t.length >= 2).map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (!clean.length) return <>{text}</>;
  const re = new RegExp(`(${clean.join("|")})`, "gi");
  return <>{text.split(re).map((part, i) => (i % 2 ? <mark key={i} className="hl">{part}</mark> : <span key={i}>{part}</span>))}</>;
}

/** Technical identifier (document / passage / task ids): monospace, selectable, truncated with a full-text tooltip. */
export const Mono = ({ children }: { children: ReactNode }) => <code className="rounded bg-surface-2 px-1 py-px text-[11px] text-muted">{children}</code>;

/** Collapsible section for progressive disclosure. Native <details>: keyboard and screen-reader support for free. */
export function Section({ title, count, defaultOpen = false, children, right }: { title: string; count?: number | string; defaultOpen?: boolean; children: ReactNode; right?: ReactNode }) {
  return (
    <details open={defaultOpen} className="group rounded-lg border border-line bg-surface">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-2 px-4 py-2.5 text-[13px] font-semibold marker:hidden [&::-webkit-details-marker]:hidden">
        <span className="flex items-center gap-2"><span aria-hidden className="text-muted transition-transform group-open:rotate-90">›</span>{title}{count !== undefined && <span className="rounded bg-surface-2 px-1.5 text-[11px] font-medium text-muted">{count}</span>}</span>{right}
      </summary>
      <div className="border-t border-line px-4 py-3">{children}</div>
    </details>
  );
}

/** Right-hand sheet for inspecting one item (document, claim) without leaving the list. Focus-trapped, Esc closes. */
export function Sheet({ open, onOpenChange, title, subtitle, children, wide }: { open: boolean; onOpenChange: (o: boolean) => void; title: ReactNode; subtitle?: ReactNode; children: ReactNode; wide?: boolean }) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/30" />
        <Dialog.Content className={clsx("fixed inset-y-0 right-0 z-50 flex w-full flex-col border-l border-line-strong bg-surface shadow-lg", wide ? "sm:w-[44rem]" : "sm:w-[34rem]")}>
          <div className="flex items-start justify-between gap-3 border-b border-line px-5 py-3">
            <div className="min-w-0"><Dialog.Title className="text-base font-semibold leading-snug text-ink">{title}</Dialog.Title>{subtitle && <div className="mt-0.5 text-xs text-muted">{subtitle}</div>}</div>
            <Dialog.Close asChild><Button variant="ghost" size="sm" aria-label="Close panel"><X className="h-4 w-4" aria-hidden /></Button></Dialog.Close>
          </div>
          <Dialog.Description className="sr-only">Details panel</Dialog.Description>
          <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">{children}</div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
