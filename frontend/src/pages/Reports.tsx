import { useMutation, useQuery } from "@tanstack/react-query";
import { Copy, Download, Printer } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Markdown } from "../components/Markdown";
import { Q, RequireProject } from "../components/common";
import { Badge, Button, Card, CardTitle, ClaimBadge, EmptyState, Mono, Notice, PageHeader, Select, Tabs, useToast } from "../components/ui";
import { api, saveFile } from "../lib/api";
import { CLAIM_LABEL, fmtDate } from "../lib/format";
import { QualityPanel } from "../components/QualityPanel";
import { ISSUE_LABEL, blockKey, issuesByBlock, unsupportedBlocks } from "../lib/quality";
import type { CitationIssue, ReportBlock, ReportSummary, ReportVersion, StructuredReport } from "../lib/types";

type Cit = ReportBlock["citations"][number];
export const citationLabel = (c: Cit) => `${c.document_id}${c.passage_id ? ` · ${c.passage_id}` : ""}${c.page_number != null ? ` · p.${c.page_number}` : ""}`;
const SIZE_KEY = "researchos.read-size"; // reading preference only
const SIZES = [0.95, 1.0625, 1.1875, 1.3];

function CitationDetail({ c }: { c: Cit }) {
  return (
    <div className="rounded-lg border border-line bg-surface p-3 text-sm">
      <p className="flex flex-wrap items-center gap-2 text-xs text-muted"><Mono>{c.document_id}</Mono>{c.passage_id && <Mono>{c.passage_id}</Mono>}<span>page {c.page_number ?? "-"}</span>{c.resolvable ? <Badge tone="ok">resolves to a stored passage</Badge> : <Badge tone="bad">does not resolve</Badge>}</p>
      {c.quote && <blockquote className="mt-2 border-l-2 border-line-strong pl-3 font-serif text-[15px] leading-relaxed">{c.quote}</blockquote>}
      {c.resolvable && <Link className="mt-2 inline-block text-xs text-accent underline" to={`/library?doc=${encodeURIComponent(c.document_id)}&passage=${encodeURIComponent(c.passage_id)}`}>Open the passage in the library</Link>}
    </div>
  );
}

const EPISTEMIC_TAG: Record<string, string> = { evidence: "evidence", fact: "fact", inference: "inference (untested)", absence: "nothing found" };

function Block({ b, sel, setSel, id, issues = [], unsupported = false }: { b: ReportBlock; sel: string | null; setSel: (k: string | null) => void; id: string; issues?: CitationIssue[]; unsupported?: boolean }) {
  return (
    <div className="mb-4">
      <p className="!mb-1">{b.text}</p>
      <p className="flex flex-wrap items-center gap-1.5 !font-sans text-xs">
        {b.epistemic && <span className="rounded border border-line px-1 text-[10.5px] text-muted" title="What kind of statement this is">{EPISTEMIC_TAG[b.epistemic] ?? b.epistemic}</span>}
        {issues.length > 0 && <span title={issues.map((i) => i.detail).join("; ")}><Badge tone="bad">citation issue: {issues.map((i) => ISSUE_LABEL[i.issue] ?? i.issue).join(", ")}</Badge></span>}
        {unsupported && <Badge tone="warn" title="Shown as supported, but no valid citation backs it">no valid citation</Badge>}
        {b.status && <span title={CLAIM_LABEL[b.status]}><ClaimBadge status={b.status} /></span>}
        {b.strength && <span className="text-muted">strength: {b.strength}</span>}
        {b.citations.map((c, i) => { const key = `${id}:${i}`; return (
          <button key={key} onClick={() => setSel(sel === key ? null : key)} aria-pressed={sel === key} title={citationLabel(c)}
            className={`rounded border px-1.5 py-px font-mono text-[11px] ${sel === key ? "border-accent bg-accent-soft text-accent" : c.resolvable ? "border-line-strong text-muted hover:text-ink" : "border-bad/50 text-bad"}`}>[{i + 1}]{!c.resolvable && "!"}</button>); })}
      </p>
      {b.citations.map((c, i) => sel === `${id}:${i}` && <div key={i} className="mt-2 xl:hidden"><CitationDetail c={c} /></div>)}
    </div>
  );
}

function Viewer({ id }: { id: string }) {
  const toast = useToast();
  const [tab, setTab] = useState<"sections" | "markdown" | "json" | "versions">("sections");
  const [size, setSize] = useState<number>(() => { try { const v = Number(localStorage.getItem(SIZE_KEY)); return SIZES.includes(v) ? v : SIZES[1]; } catch { return SIZES[1]; } });
  const [sel, setSel] = useState<string | null>(null); const [active, setActive] = useState<string>("");
  useEffect(() => { try { localStorage.setItem(SIZE_KEY, String(size)); } catch { /* preference only */ } }, [size]);
  const s = useQuery({ queryKey: ["report-structured", id], queryFn: ({ signal }) => api.get<StructuredReport>(`/reports/${id}/structured`, { signal }) });
  const md = useQuery({ queryKey: ["report-md", id], enabled: tab === "markdown", queryFn: ({ signal }) => api.text(`/reports/${id}/structured.md`, { signal }) });
  const json = useQuery({ queryKey: ["report-json", id], enabled: tab === "json", queryFn: ({ signal }) => api.text(`/reports/${id}/export.json`, { signal }) });
  const vers = useQuery({ queryKey: ["report-versions", id], enabled: tab === "versions", queryFn: ({ signal }) => api.get<ReportVersion[]>(`/reports/${id}/versions`, { signal }) });
  const dl = useMutation({
    mutationFn: async (kind: "md" | "pdf" | "json") => {
      if (kind === "pdf") saveFile(`report-${id}.pdf`, await api.blob(`/reports/${id}/export.pdf`));
      else saveFile(`report-${id}.${kind}`, await api.text(`/reports/${id}/export.${kind}`), kind === "md" ? "text/markdown" : "application/json");
    },
    onSuccess: (_d, k) => toast("ok", `Downloaded ${k.toUpperCase()}.`),
  });
  const copy = async () => { try { await navigator.clipboard.writeText(await api.text(`/reports/${id}/structured.md`)); toast("ok", "Markdown copied to the clipboard."); } catch { toast("bad", "Copy failed: the browser blocked clipboard access."); } };
  const sections = useMemo(() => s.data?.structured.sections ?? [], [s.data]);
  const blockIssues = useMemo(() => issuesByBlock(s.data?.citation_integrity), [s.data]);
  const unsupported = useMemo(() => unsupportedBlocks(s.data?.citation_integrity), [s.data]);
  useEffect(() => {
    if (tab !== "sections" || !sections.length || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver((es) => { const v = es.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0]; if (v) setActive(v.target.id); }, { rootMargin: "0px 0px -70% 0px" });
    sections.forEach((x) => { const el = document.getElementById(`sec-${x.id}`); if (el) io.observe(el); });
    return () => io.disconnect();
  }, [sections, tab]);
  const selected = useMemo(() => { if (!sel) return null; const [bid, i] = sel.split(":"); const [si, bi] = bid.split("-").map(Number); return sections[si]?.blocks[bi]?.citations[Number(i)] ?? null; }, [sel, sections]);
  return (
    <Q q={s} label="report">{(r) => (
      <div>
        <PageHeader title={r.structured.title} subtitle={`Version ${r.version} · content hash ${r.content_sha256.slice(0, 12)}…`} actions={<>
          <div className="no-print flex items-center gap-0.5 rounded-lg border border-line-strong p-0.5" role="group" aria-label="Reading size">
            <Button size="sm" variant="ghost" aria-label="Smaller text" disabled={size === SIZES[0]} onClick={() => setSize(SIZES[Math.max(0, SIZES.indexOf(size) - 1)])}>A−</Button>
            <Button size="sm" variant="ghost" aria-label="Larger text" disabled={size === SIZES[SIZES.length - 1]} onClick={() => setSize(SIZES[Math.min(SIZES.length - 1, SIZES.indexOf(size) + 1)])}>A+</Button></div>
          <Button onClick={copy}><Copy className="h-3.5 w-3.5" aria-hidden />Copy Markdown</Button>
          <Button onClick={() => dl.mutate("md")} busy={dl.isPending && dl.variables === "md"}><Download className="h-3.5 w-3.5" aria-hidden />Markdown</Button>
          <Button onClick={() => dl.mutate("pdf")} busy={dl.isPending && dl.variables === "pdf"}><Download className="h-3.5 w-3.5" aria-hidden />PDF</Button>
          <Button onClick={() => dl.mutate("json")} busy={dl.isPending && dl.variables === "json"}><Download className="h-3.5 w-3.5" aria-hidden />JSON</Button>
          <Button onClick={() => window.print()}><Printer className="h-3.5 w-3.5" aria-hidden />Print</Button></>} />
        {dl.error && <p role="alert" className="mb-2 text-sm text-bad">{dl.error.message}</p>}
        <div className="no-print"><Tabs label="Report views" value={tab} onChange={setTab} tabs={[{ id: "sections", label: "Report" }, { id: "markdown", label: "Markdown" }, { id: "json", label: "JSON" }, { id: "versions", label: "Versions" }]} /></div>
        {tab === "sections" && (
          <div className="grid gap-6 lg:grid-cols-[13rem_minmax(0,1fr)] xl:grid-cols-[13rem_minmax(0,1fr)_20rem]">
            <nav aria-label="Report sections" className="no-print hidden lg:block"><div className="sticky top-0 max-h-[calc(100vh-9rem)] overflow-y-auto pr-2">
              <p className="mb-1 text-[10.5px] font-medium uppercase tracking-wider text-muted">Sections</p>
              <ol className="space-y-px text-[13px]">{sections.map((x) => <li key={x.id}><a href={`#sec-${x.id}`} aria-current={active === `sec-${x.id}` ? "location" : undefined} className={`block rounded px-2 py-1 ${active === `sec-${x.id}` ? "bg-surface-2 font-medium text-ink" : "text-muted hover:text-ink"}`}>{x.number}. {x.title}{x.blocks.length === 0 && <span className="text-muted"> ·</span>}</a></li>)}</ol></div></nav>
            <div className="min-w-0">
              <div className="no-print mb-3 lg:hidden"><label htmlFor="jump" className="sr-only">Jump to section</label><Select id="jump" onChange={(e) => { document.getElementById(e.target.value)?.scrollIntoView(); }} defaultValue=""><option value="" disabled>Jump to section…</option>{sections.map((x) => <option key={x.id} value={`sec-${x.id}`}>{x.number}. {x.title}</option>)}</Select></div>
              <Card className="mb-4 p-3"><div className="flex flex-wrap items-center gap-2 text-sm">{Object.entries(r.structured.claim_counts).map(([k, v]) => <span key={k} className="flex items-center gap-1"><ClaimBadge status={k} /><b className="tabular-nums">{v}</b></span>)}<span className="text-xs text-muted">· {r.structured.verification_rounds} verification round(s) · {r.structured.unresolved_claims.length} unresolved</span></div>
                {r.structured.rules && <p className="mt-1.5 text-xs text-muted">{r.structured.rules}</p>}</Card>
              <QualityPanel quality={r.structured.quality} integrity={r.citation_integrity} />
              <article className="reading" style={{ ["--read-size" as string]: `${size}rem` }}>
                {sections.map((sec, si) => (
                  <section key={sec.id} id={`sec-${sec.id}`} aria-labelledby={`h-${sec.id}`} className="scroll-mt-4">
                    <h2 id={`h-${sec.id}`}>{sec.number}. {sec.title}</h2>
                    {sec.blocks.length === 0 ? <p className="!font-sans text-sm text-muted">This section has no content for this run. Nothing has been filled in.</p> : sec.blocks.map((b, bi) => <Block key={bi} b={b} id={`${si}-${bi}`} sel={sel} setSel={setSel} issues={blockIssues.get(blockKey(sec.number, bi))} unsupported={unsupported.has(blockKey(sec.number, bi))} />)}
                  </section>))}
              </article>
            </div>
            <aside aria-label="Citation details" className="no-print hidden xl:block"><div className="sticky top-0 space-y-2">
              <p className="text-[10.5px] font-medium uppercase tracking-wider text-muted">Citation</p>
              {selected ? <CitationDetail c={selected} /> : <p className="rounded-lg border border-dashed border-line-strong p-3 text-sm text-muted">Select a citation marker such as [1] to inspect its quote, page and source here.</p>}
            </div></aside>
          </div>)}
        {tab === "markdown" && <Q q={md} label="markdown">{(t) => <Card><div className="reading"><Markdown>{t}</Markdown></div></Card>}</Q>}
        {tab === "json" && <Q q={json} label="JSON">{(t) => <pre tabIndex={0} className="max-h-[70vh] overflow-auto rounded-lg border border-line bg-surface-2 p-3 font-mono text-xs">{t}</pre>}</Q>}
        {tab === "versions" && <Q q={vers} label="versions">{(v) => <ul className="divide-y divide-line rounded-lg border border-line bg-surface">{v.map((x) => <li key={x.version} className="flex flex-wrap items-center justify-between gap-2 p-3 text-sm"><span>Version {x.version} · {x.summary || "edit"}</span><span className="text-xs text-muted">{fmtDate(x.at)}</span></li>)}</ul>}</Q>}
        <div className="no-print mt-4"><Notice>PDF, Markdown and JSON are generated by the server from the stored report; this page does not alter them. Statuses are lexical verification results over the project library.</Notice></div>
      </div>)}</Q>
  );
}

function List({ projectId }: { projectId: string }) {
  const nav = useNavigate();
  const q = useQuery({ queryKey: ["reports", projectId], queryFn: ({ signal }) => api.get<ReportSummary[]>(`/projects/${projectId}/reports`, { signal }) });
  return (
    <div>
      <PageHeader title="Research Reports" subtitle="Eighteen-section reports generated from completed research runs, with citations and verification status for every statement." />
      <Card><CardTitle>Reports in this project</CardTitle>
        <Q q={q} label="reports">{(l) => l.length === 0 ? <EmptyState title="No reports yet" hint="Open a completed run in the Research Workspace and choose Generate report." action={<Button onClick={() => nav("/workspace")}>Go to workspace</Button>} /> : (
          <ul className="divide-y divide-line">{l.map((r) => <li key={r.id} className="flex flex-wrap items-center justify-between gap-2 py-2.5"><Link className="font-medium hover:text-accent" to={`/reports/${r.id}`}>{r.title}</Link><span className="flex items-center gap-2 text-xs text-muted"><Badge>{r.status.replace(/_/g, " ")}</Badge>v{r.version} · {fmtDate(r.updated_at)}</span></li>)}</ul>)}</Q></Card>
    </div>
  );
}

export default function Reports() {
  const { reportId } = useParams();
  if (reportId) return <div><Link to="/reports" className="no-print mb-2 inline-block text-xs text-accent underline">← All reports</Link><Viewer id={reportId} /></div>;
  return <RequireProject>{(p) => <List projectId={p.id} />}</RequireProject>;
}
