import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { NOVELTY_NOTE, Q, RequireProject } from "../components/common";
import { Badge, Button, Card, CardTitle, EmptyState, Field, KV, Modal, Notice, PageHeader, Textarea } from "../components/ui";
import { api } from "../lib/api";
import { fmtDate } from "../lib/format";
import type { NoveltyRow } from "../lib/types";

function Detail({ row, onClose }: { row: NoveltyRow; onClose: () => void }) {
  const a = row.assessment;
  return (
    <Modal open onOpenChange={(o) => !o && onClose()} title="Novelty assessment">
      <div className="space-y-3 text-sm">
        <p className="font-medium">{row.statement}</p>
        <Badge tone="violet">{row.label}</Badge>
        <ul className="list-disc pl-5 text-muted">{a.reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>
        <KV items={[["Documents searched", a.search_coverage.corpus_documents], ["Passages read", a.search_coverage.passages_read], ["Queries", a.search_coverage.queries.join(" | ")], ["Statement kind", a.statement_kind]]} />
        <div><h3 className="mb-1 font-medium">Closest documents</h3>
          {a.closest_documents.length === 0 ? <p className="text-muted">None found in the searched library.</p> : <ul className="space-y-2">{a.closest_documents.map((d) => <li key={d.passage_id} className="rounded border border-line p-2"><p className="font-medium">{d.title}{d.year ? ` (${d.year})` : ""}</p><p className="text-xs text-muted">{d.passage_id} · page {d.page_number ?? "-"} · coverage {d.best_coverage}</p><blockquote className="mt-1 border-l-2 border-line pl-2">{d.quote}</blockquote></li>)}</ul>}</div>
        {a.differences_from_closest.statement_terms_not_found_in_closest_documents.length > 0 && <p><b>Terms not found in the closest documents:</b> {a.differences_from_closest.statement_terms_not_found_in_closest_documents.join(", ")}</p>}
        {a.unsearched.length > 0 && <div><h3 className="font-medium">Not searched</h3><ul className="list-disc pl-5 text-muted">{a.unsearched.map((u, i) => <li key={i}>{u}</li>)}</ul></div>}
        <div><h3 className="font-medium">Limitations</h3><ul className="list-disc pl-5 text-muted">{a.limitations.map((u, i) => <li key={i}>{u}</li>)}</ul></div>
        <Notice tone="warn">{a.disclaimer}</Notice>
      </div>
    </Modal>
  );
}

function Body({ projectId }: { projectId: string }) {
  const qc = useQueryClient(); const [text, setText] = useState(""); const [open, setOpen] = useState<NoveltyRow | null>(null);
  const list = useQuery({ queryKey: ["novelty", projectId], queryFn: ({ signal }) => api.get<NoveltyRow[]>(`/projects/${projectId}/novelty`, { signal }) });
  const assess = useMutation({ mutationFn: () => api.post<NoveltyRow>(`/projects/${projectId}/novelty`, { statement: text.trim() }), onSuccess: (r) => { setText(""); void qc.invalidateQueries({ queryKey: ["novelty", projectId] }); setOpen(r); } });
  return (
    <div className="space-y-4">
      <PageHeader title="Novelty Assessment" subtitle="Compares a statement with the documents in this project's library and reports how much comparable work was found there." />
      <Notice tone="warn">{NOVELTY_NOTE}</Notice>
      <Card><CardTitle>Assess a statement</CardTitle>
        <Field label="Statement or hypothesis (at least 10 characters)" htmlFor="n-text"><Textarea id="n-text" value={text} onChange={(e) => setText(e.target.value)} /></Field>
        {assess.error && <p role="alert" className="mt-2 text-sm text-bad">{assess.error.message}</p>}
        <div className="mt-2"><Button variant="primary" disabled={text.trim().length < 10} busy={assess.isPending} onClick={() => assess.mutate()}>Assess</Button></div></Card>
      <Card><CardTitle>Previous assessments</CardTitle>
        <Q q={list} label="assessments">{(l) => l.length === 0 ? <EmptyState title="No assessments yet" /> : (
          <ul className="divide-y divide-line">{l.map((r) => <li key={r.id} className="flex flex-wrap items-center justify-between gap-2 py-2"><button className="min-w-0 flex-1 text-left text-sm hover:underline" onClick={() => setOpen(r)}>{r.statement}</button><Badge tone="violet">{r.label}</Badge><span className="text-xs text-muted">{fmtDate(r.created_at)}</span></li>)}</ul>)}</Q></Card>
      {open && <Detail row={open} onClose={() => setOpen(null)} />}
    </div>
  );
}
export default function Novelty() { return <RequireProject>{(p) => <Body projectId={p.id} />}</RequireProject>; }
