import { Badge, ClaimBadge, KV, Notice, Section } from "./ui";
import { ISSUE_LABEL, pct } from "../lib/quality";
import type { CitationIntegrity, ReportQuality } from "../lib/types";

const MIX_LABEL: Record<string, string> = { evidence: "Evidence (from a passage)", fact: "Facts (from run data)", inference: "Inferences (untested)", absence: "Absences (nothing found)" };

/** Evidence coverage, the mix of statement types, and the citation integrity check for one report. Every number comes from the API. */
export function QualityPanel({ quality, integrity }: { quality?: ReportQuality; integrity?: CitationIntegrity }) {
  if (!quality && !integrity) return null;
  const cov = quality?.evidence_coverage;
  return (
    <section aria-label="Evidence and citation quality" className="mb-4 space-y-2">
      <div className="grid gap-3 md:grid-cols-2">
        {quality && cov && (
          <div className="rounded-lg border border-line bg-surface p-3">
            <h2 className="mb-2 text-[13px] font-semibold">Evidence coverage</h2>
            <div className="mb-2 flex flex-wrap gap-1.5">{Object.entries(cov.by_status).map(([k, v]) => <span key={k} className="flex items-center gap-1 text-xs"><ClaimBadge status={k} /><b className="tabular-nums">{v}</b></span>)}</div>
            <KV items={[
              ["Claims checked", cov.claims],
              ["Supported or partly supported", cov.claims === 0 ? "none to measure" : `${cov.supported_or_partial} of ${cov.claims} (${pct(cov.fraction_supported_or_partial)})`],
              ["Not fully supported", cov.claims_not_fully_supported],
              ["Contradictions", quality.contradictions],
              ["Documents cited", cov.documents_cited],
              ["Limitations listed", quality.limitations_listed],
              ["Recommendations listed", quality.recommendations_listed]]} />
            <p className="mt-2 text-xs text-muted">{quality.reading_note}</p>
          </div>)}
        {integrity && (
          <div className="rounded-lg border border-line bg-surface p-3">
            <h2 className="mb-2 flex items-center gap-2 text-[13px] font-semibold">Citation integrity
              {integrity.citations_checked === 0 ? <Badge>no citations</Badge> : integrity.ok ? <Badge tone="ok">all citations resolve</Badge> : <Badge tone="bad">{integrity.citations_with_issues} with issues</Badge>}</h2>
            <KV items={[
              ["Citations checked", integrity.citations_checked],
              ["Valid", integrity.citations_checked === 0 ? "none to check" : `${integrity.citations_valid} (${pct(integrity.integrity_rate)})`],
              ["Claims backed by a valid citation", `${integrity.claim_blocks_with_a_valid_citation} of ${integrity.claim_blocks}`],
              ["Unsupported claims", integrity.unsupported_claims.length]]} />
            {Object.keys(integrity.by_issue).length > 0 && <ul className="mt-2 flex flex-wrap gap-1.5" aria-label="Citation problems by kind">{Object.entries(integrity.by_issue).map(([k, v]) => <li key={k}><Badge tone="bad">{ISSUE_LABEL[k] ?? k}: {v}</Badge></li>)}</ul>}
            <p className="mt-2 text-xs text-muted">{integrity.scope}</p>
          </div>)}
      </div>
      {quality && (
        <Section title="What kind of statement is each block?" count={Object.values(quality.statement_mix).reduce((a, b) => a + b, 0)}>
          <dl className="grid gap-x-6 gap-y-1.5 text-sm md:grid-cols-2">{Object.entries(quality.statement_mix).map(([k, n]) => (
            <div key={k}><dt className="font-medium">{MIX_LABEL[k] ?? k}: <span className="tabular-nums">{n}</span></dt><dd className="text-xs text-muted">{quality.statement_meaning[k]}</dd></div>))}</dl>
          <p className="mt-2 text-xs text-muted">{quality.assumptions}</p>
        </Section>)}
      {integrity && integrity.issues.length > 0 && (
        <Section title="Citation problems" count={integrity.issues.length} defaultOpen>
          <ul className="space-y-1.5 text-sm">{integrity.issues.map((i, n) => <li key={n}><Badge tone="bad">{ISSUE_LABEL[i.issue] ?? i.issue}</Badge> <span className="text-muted">section {i.section}, block {i.block + 1}{i.passage_id ? ` · ${i.passage_id}` : ""}</span> <span>{i.detail}</span></li>)}</ul>
        </Section>)}
      {integrity && integrity.unsupported_claims.length > 0 && (
        <Notice tone="warn">{integrity.unsupported_claims.length} claim(s) are shown as supported but have no valid citation: {integrity.unsupported_claims.slice(0, 3).map((u) => `"${u.text}"`).join("; ")}{integrity.unsupported_claims.length > 3 ? "…" : ""}</Notice>)}
    </section>
  );
}
