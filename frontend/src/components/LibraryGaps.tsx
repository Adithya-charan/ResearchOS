import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { Q } from "./common";
import { Badge, Card, EmptyState, Mono, Notice, Section } from "./ui";
import { api } from "../lib/api";
import { titleCase } from "../lib/format";
import { BASIS_LABEL } from "../lib/quality";
import type { LibraryGap, LibraryGapReport } from "../lib/types";

function GapCard({ g }: { g: LibraryGap }) {
  return (
    <li><Card className="h-full">
      <div className="mb-1 flex flex-wrap items-center gap-2"><Badge tone={g.basis === "observed" ? "ok" : "violet"} title={BASIS_LABEL[g.basis]}>{g.basis === "observed" ? "Observed" : "Hypothesis"}</Badge><Badge>{titleCase(g.type)}</Badge></div>
      <p className="text-sm">{g.statement}</p>
      {g.caveat && <p className="mt-1.5 text-xs text-muted">Caution: {g.caveat}</p>}
      <details className="mt-2 text-xs"><summary className="cursor-pointer text-accent">Linked passages ({g.evidence.length})</summary>
        <ul className="mt-1 space-y-1.5">{g.evidence.map((e, i) => (
          <li key={i}><Mono>{e.document_id}</Mono> <Mono>{e.passage_id}</Mono>{e.page_number != null && <span className="text-muted"> p.{e.page_number}</span>}
            <blockquote className="mt-0.5 border-l-2 border-line-strong pl-2 text-muted">{e.quote}</blockquote>
            <Link className="text-accent underline" to={`/library?doc=${encodeURIComponent(e.document_id)}&passage=${encodeURIComponent(e.passage_id)}`}>Open in the library</Link></li>))}</ul></details>
    </Card></li>
  );
}

/** Gaps found by comparing the documents of the whole library (not one run). Observed gaps are quotations; hypotheses are labelled suggestions. */
export function LibraryGaps({ projectId }: { projectId: string }) {
  const q = useQuery({ queryKey: ["library-gaps", projectId], queryFn: ({ signal }) => api.get<LibraryGapReport>(`/projects/${projectId}/gaps`, { signal }) });
  return (
    <Q q={q} label="library gap analysis">{(r) => {
      const observed = r.gaps.filter((g) => g.basis === "observed");
      const hypotheses = r.gaps.filter((g) => g.basis === "hypothesis");
      return (
        <div className="space-y-3">
          <Notice>{r.scope}</Notice>
          {r.documents === 0 ? <EmptyState title="Nothing to analyse" hint={r.note ?? "Upload documents to the library first."} /> : (
            <>
              <p className="text-sm text-muted">{r.documents} document(s) analysed · {r.counts.observed} observed · {r.counts.hypothesis} hypothesis{r.excluded_not_quotable > 0 && ` · ${r.excluded_not_quotable} left out because they are not licensed for quoting`}</p>
              {r.traceability_problems.length > 0 && <p role="alert" className="text-sm text-bad">{r.traceability_problems.length} gap(s) failed the traceability check: {r.traceability_problems[0]}</p>}
              <Section title="Observed in the library" count={observed.length} defaultOpen>
                {observed.length === 0 ? <p className="text-sm text-muted">No stated limitations, conflicts or single-document themes were found.</p> : <ul className="grid gap-3 md:grid-cols-2">{observed.map((g) => <GapCard key={g.id} g={g} />)}</ul>}
              </Section>
              <Section title="Hypotheses to check" count={hypotheses.length}>
                {hypotheses.length === 0 ? <p className="text-sm text-muted">No missing comparison was suggested.</p> : <ul className="grid gap-3 md:grid-cols-2">{hypotheses.map((g) => <GapCard key={g.id} g={g} />)}</ul>}
                {r.follow_ups.length > 0 && <ul className="mt-3 list-disc space-y-1 pl-5 text-sm">{r.follow_ups.map((f, i) => <li key={i}><span className="text-muted">{f.for_gap}:</span> {f.suggestion}</li>)}</ul>}
              </Section>
              <Section title="Themes" count={r.themes.frequent.length + r.themes.single_document.length}>
                <p className="mb-1 text-xs font-medium">Recurring in several documents</p>
                <ul className="mb-3 flex flex-wrap gap-1.5">{r.themes.frequent.map((t) => <li key={t.theme}><Badge title={t.documents.join(", ")}>{t.theme} · {t.document_count}</Badge></li>)}</ul>
                <p className="mb-1 text-xs font-medium">Appearing in one document only</p>
                <ul className="flex flex-wrap gap-1.5">{r.themes.single_document.map((t) => <li key={`${t.document}-${t.theme}`}><Badge title={t.title}>{t.theme}</Badge></li>)}</ul>
              </Section>
              <Section title="How to read this" defaultOpen={false}><ul className="list-disc space-y-1 pl-5 text-sm text-muted">{r.limitations.map((x) => <li key={x}>{x}</li>)}</ul></Section>
            </>)}
        </div>);
    }}</Q>
  );
}
