import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Q, LEXICAL_NOTE, RequireProject } from "../components/common";
import { Badge, Button, Card, ClaimBadge, EmptyState, Field, Input, Mono, Notice, PageHeader, Section, Select, Textarea, useToast } from "../components/ui";
import { api } from "../lib/api";
import { CLAIM_LABEL, CLAIM_SHORT, fmtDate } from "../lib/format";
import { CLAIM_STATUSES, type Claim, type ClaimEvidence, type ClaimList, type ClaimStatus, type ClaimVerification, type DocumentRec, type Run } from "../lib/types";
import { useRuns } from "../project";

function EvidenceItem({ e, doc, onRemove }: { e: ClaimEvidence; doc?: DocumentRec; onRemove?: () => void }) {
  return (
    <li className="rounded-lg border border-line p-3">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 text-xs">
          <p className="truncate text-[13px] font-medium">{doc ? (doc.title || doc.filename) : <Mono>{e.document_id}</Mono>}</p>
          <p className="text-muted">{doc?.authors?.length ? `${doc.authors.join("; ")} · ` : ""}{doc?.year ?? ""}</p>
        </div>
        {onRemove && <Button size="sm" variant="ghost" aria-label="Remove this evidence" onClick={onRemove}><Trash2 className="h-3.5 w-3.5" aria-hidden /></Button>}
      </div>
      <blockquote className="mt-2 border-l-2 border-line-strong pl-3 font-serif text-[15px] leading-relaxed">{e.quote}</blockquote>
      <p className="mt-2 flex flex-wrap items-center gap-2 text-xs text-muted">
        <Mono>{e.passage_id}</Mono><span>page {e.page_number ?? "-"}</span><span>{e.directness}</span><span>coverage {e.coverage}</span>
        {e.citation_valid ? <Badge tone="ok">citation resolves</Badge> : <Badge tone="bad">citation does not resolve</Badge>}
        <Link className="text-accent underline" to={`/library?doc=${encodeURIComponent(e.document_id)}&passage=${encodeURIComponent(e.passage_id)}`}>Open in library</Link>
      </p>
      {e.reason && <p className="mt-1 text-xs text-muted">{e.reason}</p>}
    </li>
  );
}

function Center({ projectId, id, docs }: { projectId: string; id: string; docs: DocumentRec[] }) {
  const qc = useQueryClient(); const toast = useToast(); const [passage, setPassage] = useState("");
  const q = useQuery({ queryKey: ["claim", projectId, id], queryFn: ({ signal }) => api.get<Claim>(`/projects/${projectId}/claims/${id}`, { signal }) });
  const hist = useQuery({ queryKey: ["claim-verifs", projectId, id], queryFn: ({ signal }) => api.get<ClaimVerification[]>(`/projects/${projectId}/claims/${id}/verifications`, { signal }) });
  const refresh = () => { for (const k of ["claim", "claim-verifs", "claims"]) void qc.invalidateQueries({ queryKey: [k, projectId] }); void qc.invalidateQueries({ queryKey: ["overview"] }); };
  const verify = useMutation({ mutationFn: () => api.post(`/projects/${projectId}/claims/${id}/verify`), onSuccess: () => { refresh(); toast("ok", "Claim re-verified."); } });
  const add = useMutation({ mutationFn: () => api.post(`/projects/${projectId}/claims/${id}/evidence`, { passage_id: passage.trim() }), onSuccess: () => { setPassage(""); refresh(); toast("ok", "Evidence attached; re-verify to update the status."); } });
  const rm = useMutation({ mutationFn: (eid: number) => api.del(`/projects/${projectId}/claims/${id}/evidence/${eid}`), onSuccess: () => { refresh(); toast("ok", "Evidence removed."); } });
  const err = verify.error ?? add.error ?? rm.error;
  const byId = useMemo(() => new Map(docs.map((d) => [d.id, d])), [docs]);
  return (
    <Q q={q} label="claim">{(c) => {
      const ev = c.evidence ?? [];
      const group = (role: string) => ev.filter((e) => e.role === role);
      return (
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <div className="min-w-0 space-y-3">
            <Card>
              <div className="flex flex-wrap items-center gap-2"><ClaimBadge status={c.status} /><span className="text-xs text-muted">{CLAIM_LABEL[c.status]}</span></div>
              <p className="mt-2 font-serif text-[17px] leading-snug">{c.text}</p>
              <p className="mt-2 flex flex-wrap items-center gap-2 text-xs text-muted">{c.claim_key && <Mono>{c.claim_key}</Mono>}<span>{c.claim_type}</span><span>{c.source === "agent" ? `agent: ${c.origin_agent}` : "added by you"}</span><span>updated {fmtDate(c.updated_at)}</span></p>
              <div className="mt-3 flex justify-end"><Button variant="primary" size="sm" busy={verify.isPending} onClick={() => verify.mutate()}>Re-verify</Button></div>
            </Card>
            <Card><h3 className="mb-1 text-[13px] font-semibold">Why this status</h3>
              {c.explanation?.length ? <ul className="list-disc space-y-0.5 pl-4 text-sm">{c.explanation.map((x, i) => <li key={i}>{x}</li>)}</ul> : <p className="text-sm text-muted">No explanation recorded.</p>}
              <p className="mt-2 text-xs text-muted">Strength: <b className="text-ink">{c.strength?.level ?? "-"}</b>{c.strength?.basis ? ` (${c.strength.basis})` : ""}. {c.strength?.note}</p>
              <p className="mt-1 text-xs text-muted">This is a lexical check against the uploaded library. It is not formal proof or peer review.</p></Card>
            <Section title="Verification history" count={hist.data?.length} defaultOpen>
              <Q q={hist} label="history">{(h) => h.length === 0 ? <p className="text-sm text-muted">No verification rounds recorded.</p> : <ol className="space-y-1.5">{h.map((v) => <li key={v.id} className="flex flex-wrap items-center gap-2 text-sm"><span className="text-xs text-muted">Round {v.round} · {fmtDate(v.created_at)}</span><ClaimBadge status={v.status} /><span className="text-xs text-muted">by {v.requested_by}</span></li>)}</ol>}</Q></Section>
          </div>
          <div className="min-w-0 space-y-3" aria-label="Evidence">
            {(["supporting", "contradicting", "contextual"] as const).map((role) => (
              <section key={role}><h3 className="mb-1.5 flex items-center gap-2 text-[13px] font-semibold capitalize">{role}<span className="rounded bg-surface-2 px-1.5 text-[11px] font-medium text-muted">{group(role).length}</span></h3>
                {group(role).length === 0 ? <p className="rounded-lg border border-dashed border-line-strong px-3 py-2 text-sm text-muted">No {role} evidence.</p> : <ul className="space-y-2">{group(role).map((e) => <EvidenceItem key={e.id} e={e} doc={byId.get(e.document_id)} onRemove={e.origin === "user" ? () => rm.mutate(e.id) : undefined} />)}</ul>}</section>))}
            <Card><div className="flex items-end gap-2"><div className="flex-1"><Field label="Attach a passage by id" htmlFor="ev-pid" hint="Passage ids appear in the library preview, e.g. doc-…:p000"><Input id="ev-pid" className="font-mono text-xs" value={passage} onChange={(e) => setPassage(e.target.value)} /></Field></div><Button disabled={!passage.trim()} busy={add.isPending} onClick={() => add.mutate()}>Add</Button></div>
              {err && <p role="alert" className="mt-2 text-sm text-bad">{err.message}</p>}</Card>
          </div>
        </div>
      );
    }}</Q>
  );
}

function Body({ projectId }: { projectId: string }) {
  const qc = useQueryClient(); const toast = useToast();
  const [status, setStatus] = useState<"" | ClaimStatus>(""); const [search, setSearch] = useState(""); const [sel, setSel] = useState<string | null>(null);
  const [text, setText] = useState(""); const [runPick, setRunPick] = useState("");
  const runs = useRuns(projectId);
  const docs = useQuery({ queryKey: ["documents", projectId], queryFn: ({ signal }) => api.get<DocumentRec[]>(`/projects/${projectId}/documents`, { signal }) });
  const claims = useQuery({ queryKey: ["claims", projectId, "all"], queryFn: ({ signal }) => api.get<ClaimList>(`/projects/${projectId}/claims`, { signal, query: { limit: 500 } }) });
  const done = () => { void qc.invalidateQueries({ queryKey: ["claims", projectId] }); void qc.invalidateQueries({ queryKey: ["overview"] }); };
  const create = useMutation({ mutationFn: () => api.post<Claim>(`/projects/${projectId}/claims`, { text: text.trim() }), onSuccess: (c) => { setText(""); done(); setSel(c.id); toast("ok", "Claim added and verified."); } });
  const extract = useMutation({ mutationFn: () => api.post(`/runs/${runPick}/claims/extract`), onSuccess: () => { done(); toast("ok", "Claims extracted and verified."); } });
  const list = useMemo(() => claims.data?.claims ?? [], [claims.data]);
  const counts = useMemo(() => Object.fromEntries(CLAIM_STATUSES.map((s) => [s, list.filter((c) => c.status === s).length])) as Record<ClaimStatus, number>, [list]);
  const shown = list.filter((c) => (!status || c.status === status) && c.text.toLowerCase().includes(search.toLowerCase()));
  useEffect(() => { if (!sel && shown.length) setSel(shown[0].id); }, [shown, sel]);
  const runList: Run[] = runs.data ?? [];
  return (
    <div>
      <PageHeader title="Evidence Verification" subtitle="Each claim is checked against stored passages. A status describes lexical agreement with the library, not whether the claim is true." />
      <div className="mb-3"><Notice>{LEXICAL_NOTE}</Notice></div>
      <div className="grid gap-4 lg:grid-cols-[19rem_minmax(0,1fr)]">
        <div className="min-w-0 space-y-3">
          <Card className="p-3">
            <label htmlFor="c-search" className="sr-only">Search claims</label><Input id="c-search" placeholder="Search claims" value={search} onChange={(e) => setSearch(e.target.value)} />
            <div className="mt-2 flex flex-wrap gap-1" role="group" aria-label="Filter by status">
              <button className={`rounded-md border px-1.5 py-0.5 text-xs ${status === "" ? "border-accent bg-accent-soft text-accent" : "border-line-strong text-muted hover:text-ink"}`} aria-pressed={status === ""} onClick={() => setStatus("")}>All {list.length}</button>
              {CLAIM_STATUSES.map((s) => <button key={s} aria-pressed={status === s} onClick={() => setStatus(status === s ? "" : s)} className={`rounded-md border px-1.5 py-0.5 text-xs ${status === s ? "border-accent bg-accent-soft text-accent" : "border-line-strong text-muted hover:text-ink"}`}>{CLAIM_SHORT[s]} {counts[s]}</button>)}
            </div>
          </Card>
          <Q q={claims} label="claims">{() => shown.length === 0 ? <EmptyState title={list.length ? "No claims match" : "No claims yet"} hint={list.length ? "Clear the filters." : "Add a claim or extract claims from a run below."} /> : (
            <ul className="max-h-[60vh] divide-y divide-line overflow-y-auto rounded-lg border border-line bg-surface" aria-label="Claims">{shown.map((c) => (
              <li key={c.id}><button onClick={() => { setSel(c.id); if (window.matchMedia("(max-width: 1023px)").matches) setTimeout(() => document.getElementById("claim-detail")?.scrollIntoView({ block: "start" }), 50); }} aria-current={sel === c.id} className={`block w-full px-3 py-2 text-left text-sm hover:bg-surface-2/60 ${sel === c.id ? "bg-accent-soft" : ""}`}>
                <span className="line-clamp-3">{c.text}</span><span className="mt-1 flex items-center justify-between gap-2"><ClaimBadge status={c.status} /><span className="text-xs text-muted tabular-nums">{c.evidence_counts.supporting}↑ {c.evidence_counts.contradicting}↓</span></span></button></li>))}</ul>)}</Q>
          <Section title="Add or extract claims">
            <div className="space-y-3"><Field label="Check a claim of your own" htmlFor="c-text" hint="At least 8 characters. Verified against this project's library."><Textarea id="c-text" value={text} onChange={(e) => setText(e.target.value)} /></Field>
              {create.error && <p role="alert" className="text-sm text-bad">{create.error.message}</p>}
              <Button variant="primary" size="sm" disabled={text.trim().length < 8} busy={create.isPending} onClick={() => create.mutate()}>Add and verify</Button>
              {runList.length > 0 && <div className="border-t border-line pt-3"><label htmlFor="ex-run" className="mb-1 block text-xs font-medium">Extract claims from a run</label><Select id="ex-run" value={runPick} onChange={(e) => setRunPick(e.target.value)}><option value="">Choose a run</option>{runList.map((r) => <option key={r.id} value={r.id}>{r.question.slice(0, 50)} · {r.state.status}</option>)}</Select>
                {extract.error && <p role="alert" className="mt-1 text-sm text-bad">{extract.error.message}</p>}<div className="mt-2"><Button size="sm" disabled={!runPick} busy={extract.isPending} onClick={() => extract.mutate()}>Extract and verify</Button></div></div>}</div>
          </Section>
        </div>
        <div className="min-w-0 scroll-mt-2" id="claim-detail">{sel && list.some((c) => c.id === sel) ? <Center projectId={projectId} id={sel} docs={docs.data ?? []} key={sel} /> : <EmptyState title="Select a claim" hint="Its status, explanation, history and evidence appear here." />}</div>
      </div>
    </div>
  );
}
export default function Evidence() { return <RequireProject>{(p) => <Body projectId={p.id} />}</RequireProject>; }
