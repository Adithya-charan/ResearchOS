import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Upload, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useSearchParams } from "react-router-dom";
import { Q, LEXICAL_NOTE, RequireProject } from "../components/common";
import { Badge, Button, Card, ConfirmDialog, EmptyState, Field, Highlight, Input, Modal, Mono, Notice, PageHeader, Segmented, Select, Sheet, StatusBadge, useToast } from "../components/ui";
import { api } from "../lib/api";
import { fmtBytes, fmtDate } from "../lib/format";
import type { CorpusStats, DocumentRec, Passage, SearchResponse } from "../lib/types";

export const ALLOWED_EXT = [".pdf", ".docx", ".txt", ".md"];
export const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;
/** Client-side pre-check only; the server validates content type and size again. */
export function validateUpload(f: { name: string; size: number }): string | null {
  const ext = f.name.slice(f.name.lastIndexOf(".")).toLowerCase();
  if (!ALLOWED_EXT.includes(ext)) return `Unsupported file type. Allowed: ${ALLOWED_EXT.join(", ")}.`;
  if (f.size === 0) return "The file is empty.";
  if (f.size > MAX_UPLOAD_BYTES) return `The file is larger than ${fmtBytes(MAX_UPLOAD_BYTES)}.`;
  return null;
}

function UploadForm({ projectId, onDone }: { projectId: string; onDone: () => void }) {
  const qc = useQueryClient(); const file = useRef<HTMLInputElement>(null);
  const [err, setErr] = useState<string | null>(null); const [meta, setMeta] = useState({ title: "", authors: "", publication_date: "", source: "", license: "", url: "", doi: "", document_type: "" });
  const [quote, setQuote] = useState(true);
  const m = useMutation({
    mutationFn: (fd: FormData) => api.upload<{ status: string; message?: string; warnings?: string[]; document?: DocumentRec }>(`/projects/${projectId}/documents`, fd),
    onSuccess: () => { void qc.invalidateQueries({ queryKey: ["documents", projectId] }); void qc.invalidateQueries({ queryKey: ["corpus", projectId] }); void qc.invalidateQueries({ queryKey: ["overview"] }); },
  });
  const submit = (e: React.FormEvent) => {
    e.preventDefault(); setErr(null);
    const f = file.current?.files?.[0];
    if (!f) return setErr("Choose a file to upload.");
    const v = validateUpload(f); if (v) return setErr(v);
    if (!meta.source.trim() || !meta.license.trim()) return setErr("Source and license are required: only upload material you are allowed to use.");
    const fd = new FormData(); fd.set("file", f);
    for (const [k, val] of Object.entries(meta)) if (val.trim()) fd.set(k, val.trim());
    fd.set("allowed_use", quote ? "search,quote" : "search");
    m.mutate(fd, { onSuccess: () => { if (file.current) file.current.value = ""; } });
  };
  const set = (k: keyof typeof meta) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => setMeta({ ...meta, [k]: e.target.value });
  const result = m.data as { status?: string; message?: string; warnings?: string[] } | undefined;
  return (
    <form onSubmit={submit} className="space-y-3" noValidate>
      <Field label="File (PDF, DOCX, TXT or Markdown)" htmlFor="u-file"><input id="u-file" ref={file} type="file" accept={ALLOWED_EXT.join(",")} className="block w-full text-sm" /></Field>
      <div className="grid gap-3 sm:grid-cols-2">
        <Field label="Source (required)" htmlFor="u-source" hint="Where you obtained it"><Input id="u-source" value={meta.source} onChange={set("source")} /></Field>
        <Field label="License (required)" htmlFor="u-license" hint="e.g. CC-BY-4.0, own work"><Input id="u-license" value={meta.license} onChange={set("license")} /></Field>
        <Field label="Title" htmlFor="u-title"><Input id="u-title" value={meta.title} onChange={set("title")} /></Field>
        <Field label="Authors (separate with ;)" htmlFor="u-authors"><Input id="u-authors" value={meta.authors} onChange={set("authors")} /></Field>
        <Field label="Publication date (YYYY-MM-DD)" htmlFor="u-date"><Input id="u-date" value={meta.publication_date} onChange={set("publication_date")} placeholder="2024-05-01" /></Field>
        <Field label="DOI" htmlFor="u-doi"><Input id="u-doi" value={meta.doi} onChange={set("doi")} /></Field>
        <Field label="URL" htmlFor="u-url"><Input id="u-url" value={meta.url} onChange={set("url")} /></Field>
        <Field label="Document type" htmlFor="u-type"><Select id="u-type" value={meta.document_type} onChange={set("document_type")}><option value="">Unspecified</option><option value="article">Article</option><option value="preprint">Preprint</option><option value="thesis">Thesis</option><option value="report">Report</option><option value="book">Book</option><option value="other">Other</option></Select></Field>
      </div>
      <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={quote} onChange={(e) => setQuote(e.target.checked)} />Allow quoting passages in reports</label>
      {err && <p role="alert" className="text-sm text-bad">{err}</p>}
      {m.error && <p role="alert" className="text-sm text-bad">{m.error.message}</p>}
      {m.isSuccess && <p role="status" className={result?.status === "ingested" ? "text-sm text-ok" : "text-sm text-warn"}>{result?.status === "ingested" ? "Uploaded and indexed. " : `Upload result: ${result?.status ?? "unknown"}. `}{result?.message}{result?.warnings?.length ? ` Warnings: ${result.warnings.join(" ")}` : ""}</p>}
      <div className="flex justify-end gap-2"><Button type="button" onClick={onDone}>Close</Button><Button type="submit" variant="primary" busy={m.isPending}><Upload className="h-4 w-4" aria-hidden />Upload</Button></div>
    </form>
  );
}

const termsOf = (q: string) => Array.from(q.matchAll(/"([^"]+)"|(\S+)/g)).map((m) => m[1] ?? m[2]).filter((t) => t && !/^(AND|OR|NOT)$/i.test(t));

function DocPreview({ projectId, doc, focus, terms, onClose, onReprocess, onDelete }: { projectId: string; doc: DocumentRec; focus?: string; terms: string[]; onClose: () => void; onReprocess: () => void; onDelete: () => void }) {
  const q = useQuery({ queryKey: ["passages", projectId, doc.id], queryFn: ({ signal }) => api.get<Passage[]>(`/projects/${projectId}/documents/${doc.id}/passages`, { signal }) });
  const focusRef = useRef<HTMLLIElement | null>(null);
  useEffect(() => { focusRef.current?.scrollIntoView({ block: "center" }); }, [q.data, focus]);
  return (
    <Sheet open onOpenChange={(o) => !o && onClose()} wide title={doc.title || doc.filename}
      subtitle={<>{doc.authors?.length ? doc.authors.join("; ") : "Authors not recorded"}{doc.year ? ` · ${doc.year}` : ""}{doc.doi && <> · DOI <Mono>{doc.doi}</Mono></>}</>}>
      <div className="mb-3 flex flex-wrap gap-2"><Button size="sm" onClick={onReprocess}>Reprocess</Button><Button size="sm" variant="danger" onClick={onDelete}>Delete</Button></div>
      <dl className="mb-4 grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 text-sm">
        {([["File", doc.filename], ["Type", `${doc.file_type}${doc.document_type ? ` · ${doc.document_type}` : ""}`], ["Size", fmtBytes(doc.size_bytes)], ["Pages", doc.pages], ["Source", doc.source], ["License", doc.license], ["Allowed use", doc.allowed_use?.join(", ")], ["Extraction", doc.extraction_status], ["Quality", doc.quality?.status ?? "-"], ["Added", fmtDate(doc.created_at)]] as [string, ReactNode][]).map(([k, v]) => <div key={k} className="contents"><dt className="text-xs text-muted">{k}</dt><dd className="break-words">{v}</dd></div>)}
      </dl>
      {doc.warnings?.length > 0 && <div className="mb-3"><Notice tone="warn">{doc.warnings.join(" ")}</Notice></div>}
      {doc.untrusted_content_notice && <div className="mb-3"><Notice>{doc.untrusted_content_notice}</Notice></div>}
      <h3 className="mb-2 text-[13px] font-semibold">Passages</h3>
      <Q q={q} label="passages">{(p) => p.length === 0 ? <p className="text-sm text-muted">No extracted passages.</p> : (
        <ol className="space-y-3">{p.map((x) => (
          <li key={x.passage_id} ref={x.passage_id === focus ? focusRef : undefined} className={`rounded-lg border p-3 ${x.passage_id === focus ? "border-accent bg-accent-soft" : "border-line"}`}>
            <p className="mb-1 flex flex-wrap items-center gap-2 text-xs text-muted"><Mono>{x.passage_id}</Mono><span>page {x.page_number ?? "-"}</span>{x.section && <span>{x.section}</span>}{x.quality !== "ok" && <Badge tone="warn">quality: {x.quality}</Badge>}{x.passage_id === focus && <Badge tone="info">matched passage</Badge>}</p>
            <p className="whitespace-pre-wrap font-serif text-[15px] leading-relaxed"><Highlight text={x.text} terms={terms} /></p>
          </li>))}</ol>)}</Q>
    </Sheet>
  );
}

const LEVELS = [
  { id: "chunk" as const, label: "Passages", hint: "Passage-level BM25 matches with page and passage citations" },
  { id: "document" as const, label: "Documents", hint: "Whole-document ranking" },
  { id: "metadata" as const, label: "Metadata", hint: "Title, author and year fields only" },
];
type Level = (typeof LEVELS)[number]["id"];

function SearchPanel({ projectId, initial, docs, onOpen }: { projectId: string; initial: string; docs: DocumentRec[]; onOpen: (d: DocumentRec, passage?: string, terms?: string[]) => void }) {
  const [q, setQ] = useState(initial); const [level, setLevel] = useState<Level>("chunk"); const [phrase, setPhrase] = useState(false);
  const [yearMin, setYearMin] = useState(""); const [yearMax, setYearMax] = useState(""); const [author, setAuthor] = useState(""); const [docType, setDocType] = useState(""); const [hq, setHq] = useState(false);
  const [run, setRun] = useState<string | null>(initial.trim().length >= 2 ? initial.trim() : null);
  const sent = run ? (phrase && !run.startsWith('"') ? `"${run}"` : run) : null;
  const search = useQuery({
    queryKey: ["search", projectId, sent, level, yearMin, yearMax, author, docType, hq], enabled: !!sent,
    queryFn: ({ signal }) => api.get<SearchResponse>(`/projects/${projectId}/search`, { signal, query: { q: sent, level, k: 15, year_min: yearMin, year_max: yearMax, author, document_type: docType, exclude_low_quality: hq || undefined } }),
  });
  const chips: [string, () => void][] = [
    ...(yearMin ? [[`From ${yearMin}`, () => setYearMin("")] as [string, () => void]] : []), ...(yearMax ? [[`To ${yearMax}`, () => setYearMax("")] as [string, () => void]] : []),
    ...(author ? [[`Author: ${author}`, () => setAuthor("")] as [string, () => void]] : []), ...(docType ? [[`Type: ${docType}`, () => setDocType("")] as [string, () => void]] : []),
    ...(hq ? [["Low quality excluded", () => setHq(false)] as [string, () => void]] : []), ...(phrase ? [["Exact phrase", () => setPhrase(false)] as [string, () => void]] : []),
  ];
  const clearAll = () => { setYearMin(""); setYearMax(""); setAuthor(""); setDocType(""); setHq(false); setPhrase(false); };
  const terms = termsOf(sent ?? "");
  return (
    <Card>
      <form role="search" onSubmit={(e) => { e.preventDefault(); if (q.trim().length >= 2) setRun(q.trim()); }} className="space-y-3">
        <div className="flex gap-2"><div className="flex-1"><label htmlFor="s-q" className="sr-only">Search query</label><Input id="s-q" className="h-9 text-[14px]" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search passages, documents or metadata in this library" /></div><Button type="submit" variant="primary" className="h-9" disabled={q.trim().length < 2}>Search</Button></div>
        <div className="flex flex-wrap items-center gap-3">
          <Segmented label="Search level" value={level} onChange={setLevel} options={LEVELS} />
          <label className="flex items-center gap-1.5 text-xs"><input type="checkbox" checked={phrase} onChange={(e) => setPhrase(e.target.checked)} />Exact phrase</label>
          <label className="flex items-center gap-1.5 text-xs"><input type="checkbox" checked={hq} onChange={(e) => setHq(e.target.checked)} />Exclude low-quality text</label>
          <Badge tone="info" title="BM25 term matching. It does not understand synonyms or meaning.">Lexical search, not semantic</Badge>
        </div>
        <div className="grid gap-2 sm:grid-cols-4">
          <Field label="Year from" htmlFor="s-ymin"><Input id="s-ymin" inputMode="numeric" value={yearMin} onChange={(e) => setYearMin(e.target.value)} /></Field>
          <Field label="Year to" htmlFor="s-ymax"><Input id="s-ymax" inputMode="numeric" value={yearMax} onChange={(e) => setYearMax(e.target.value)} /></Field>
          <Field label="Author" htmlFor="s-author"><Input id="s-author" value={author} onChange={(e) => setAuthor(e.target.value)} /></Field>
          <Field label="Document type" htmlFor="s-type"><Input id="s-type" value={docType} onChange={(e) => setDocType(e.target.value)} /></Field>
        </div>
        {chips.length > 0 && <div className="flex flex-wrap items-center gap-1.5" aria-label="Active filters">{chips.map(([label, off]) => <button type="button" key={label} onClick={off} className="inline-flex items-center gap-1 rounded-md border border-line-strong bg-surface-2 px-1.5 py-0.5 text-xs hover:bg-surface" aria-label={`Remove filter ${label}`}>{label}<X className="h-3 w-3" aria-hidden /></button>)}<button type="button" className="text-xs text-accent underline" onClick={clearAll}>Clear all</button></div>}
      </form>
      <div className="mt-2"><Notice>{LEXICAL_NOTE}</Notice></div>
      {sent && <div className="mt-3" aria-live="polite"><Q q={search} label="results">{(r) => (
        <>
          <p className="mb-2 text-xs text-muted">{r.hits.length} result{r.hits.length === 1 ? "" : "s"} · level: {level} · method: {r.method}</p>
          {r.hits.length === 0 ? <EmptyState title="No matching text" hint="Lexical search matches words, not meaning. Try other terms used in the documents, or remove filters." /> : (
            <ol className="divide-y divide-line">{r.hits.map((h, i) => {
              const d = docs.find((x) => x.id === h.document_id);
              const snippet = h.text || h.excerpt || h.citation?.excerpt;
              const mt = h.matched_terms?.length ? h.matched_terms : terms;
              return (
                <li key={`${h.document_id}-${h.passage_id ?? i}`} className="py-3">
                  <button className="text-left text-[14px] font-medium leading-snug text-accent hover:underline" onClick={() => d && onOpen(d, h.passage_id, mt)} disabled={!d}>{h.title}</button>
                  <p className="mt-0.5 text-xs text-muted">{d?.authors?.length ? `${d.authors.join("; ")} · ` : ""}{h.year ?? d?.year ?? "year n/a"}{d?.doi && <> · DOI <Mono>{d.doi}</Mono></>}{d?.document_type ? ` · ${d.document_type}` : ""}</p>
                  {snippet && <p className="mt-1.5 line-clamp-4 whitespace-pre-wrap font-serif text-[15px] leading-relaxed"><Highlight text={snippet} terms={mt} /></p>}
                  <p className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted"><span>Citation: document {h.document_id}{h.passage_id ? `, passage ${h.passage_id}` : ""}, page {h.page_number ?? "-"}</span>{h.section && <span>{h.section}</span>}<span>score {h.score.toFixed(2)}</span>{mt.slice(0, 6).map((t) => <Badge key={t}>{t}</Badge>)}</p>
                </li>);
            })}</ol>)}
        </>)}</Q></div>}
    </Card>
  );
}

type SortKey = "title" | "year" | "pages" | "created_at";
function DocTable({ docs, onOpen }: { docs: DocumentRec[]; onOpen: (d: DocumentRec) => void }) {
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: "created_at", dir: -1 }); const [filter, setFilter] = useState("");
  const rows = useMemo(() => {
    const f = filter.toLowerCase();
    const val = (d: DocumentRec) => (sort.key === "title" ? (d.title || d.filename).toLowerCase() : sort.key === "year" ? d.year ?? 0 : sort.key === "pages" ? d.pages : d.created_at);
    return docs.filter((d) => `${d.title} ${d.filename} ${d.authors?.join(" ")} ${d.doi ?? ""}`.toLowerCase().includes(f)).sort((a, b) => (val(a) < val(b) ? -1 : val(a) > val(b) ? 1 : 0) * sort.dir);
  }, [docs, sort, filter]);
  const th = (key: SortKey, label: string) => (
    <th scope="col" aria-sort={sort.key === key ? (sort.dir === 1 ? "ascending" : "descending") : "none"} className="px-3 py-1.5 font-medium">
      <button className="inline-flex items-center gap-1 uppercase tracking-wide hover:text-ink" onClick={() => setSort((s) => ({ key, dir: s.key === key ? (s.dir === 1 ? -1 : 1) : 1 }))}>{label}<span aria-hidden>{sort.key === key ? (sort.dir === 1 ? "↑" : "↓") : ""}</span></button></th>);
  return (
    <div>
      <div className="mb-2 max-w-sm"><label htmlFor="d-filter" className="sr-only">Filter documents</label><Input id="d-filter" placeholder="Filter by title, author, DOI or file name" value={filter} onChange={(e) => setFilter(e.target.value)} /></div>
      {rows.length === 0 ? <EmptyState title="No documents match" /> : (
        <div className="overflow-x-auto rounded-lg border border-line bg-surface"><table className="w-full text-left text-sm"><caption className="sr-only">Library documents</caption>
          <thead className="bg-surface-2 text-[11px] text-muted"><tr>{th("title", "Document")}<th scope="col" className="px-3 py-1.5 font-medium uppercase tracking-wide">Authors</th>{th("year", "Year")}{th("pages", "Pages")}<th scope="col" className="px-3 py-1.5 font-medium uppercase tracking-wide">Index</th>{th("created_at", "Added")}</tr></thead>
          <tbody className="divide-y divide-line">{rows.map((d) => (
            <tr key={d.id} className="cursor-pointer hover:bg-surface-2/60" onClick={() => onOpen(d)}>
              <td className="max-w-md px-3 py-2"><button className="text-left font-medium text-accent hover:underline" onClick={(e) => { e.stopPropagation(); onOpen(d); }}>{d.title || d.filename}</button>{d.doi && <p className="text-xs text-muted">DOI <Mono>{d.doi}</Mono></p>}{d.warnings?.length > 0 && <Badge tone="warn">{d.warnings.length} warning{d.warnings.length === 1 ? "" : "s"}</Badge>}</td>
              <td className="max-w-[14rem] truncate px-3 py-2 text-muted" title={d.authors?.join("; ")}>{d.authors?.join("; ") || "—"}</td><td className="px-3 py-2 tabular-nums">{d.year ?? "—"}</td><td className="px-3 py-2 tabular-nums">{d.pages}</td>
              <td className="px-3 py-2"><StatusBadge status={d.extraction_status === "ok" || d.extraction_status === "extracted" ? "completed" : d.extraction_status} /></td><td className="whitespace-nowrap px-3 py-2 text-muted">{fmtDate(d.created_at)}</td>
            </tr>))}</tbody></table></div>)}
    </div>
  );
}

function Body({ projectId }: { projectId: string }) {
  const [params] = useSearchParams(); const qc = useQueryClient(); const toast = useToast();
  const [upload, setUpload] = useState(false); const [view, setView] = useState<{ doc: DocumentRec; focus?: string; terms: string[] } | null>(null); const [del, setDel] = useState<DocumentRec | null>(null);
  const docs = useQuery({ queryKey: ["documents", projectId], queryFn: ({ signal }) => api.get<DocumentRec[]>(`/projects/${projectId}/documents`, { signal }) });
  const stats = useQuery({ queryKey: ["corpus", projectId], queryFn: ({ signal }) => api.get<CorpusStats>(`/projects/${projectId}/corpus`, { signal }) });
  // deep link from Evidence / Reports: /library?doc=<id>&passage=<id> opens the preview at that passage
  const deepDoc = params.get("doc"); const deepPassage = params.get("passage") ?? undefined;
  useEffect(() => { const d = docs.data?.find((x) => x.id === deepDoc); if (d) setView({ doc: d, focus: deepPassage, terms: [] }); }, [docs.data, deepDoc, deepPassage]);
  const refresh = () => { void qc.invalidateQueries({ queryKey: ["documents", projectId] }); void qc.invalidateQueries({ queryKey: ["corpus", projectId] }); void qc.invalidateQueries({ queryKey: ["overview"] }); };
  const rm = useMutation({ mutationFn: (id: string) => api.del(`/projects/${projectId}/documents/${id}`), onSuccess: () => { setDel(null); setView(null); refresh(); toast("ok", "Document deleted."); } });
  const re = useMutation({ mutationFn: (id: string) => api.post(`/projects/${projectId}/documents/${id}/reprocess`), onSuccess: () => { refresh(); void qc.invalidateQueries({ queryKey: ["passages", projectId] }); toast("ok", "Document reprocessed."); } });
  return (
    <div>
      <PageHeader title="Literature Library" subtitle="Documents you uploaded to this project. Only material you have the right to use belongs here." actions={<Button variant="primary" onClick={() => setUpload(true)}><Upload className="h-4 w-4" aria-hidden />Upload document</Button>} />
      <Q q={stats} label="corpus">{(s) => <p className="mb-3 flex flex-wrap gap-x-5 gap-y-1 text-sm text-muted"><span><b className="text-ink tabular-nums">{s.documents}</b> {s.documents === 1 ? "document" : "documents"}</span><span><b className="text-ink tabular-nums">{s.passages}</b> passages</span><span><b className="text-ink tabular-nums">{s.indexed_documents}</b> indexed</span><span><b className="text-ink tabular-nums">{s.indexed_passages}</b> indexed passages</span></p>}</Q>
      <Q q={docs} label="documents">{(d) => (
        <div className="space-y-4">
          <SearchPanel projectId={projectId} initial={params.get("q") ?? ""} key={params.get("q") ?? ""} docs={d} onOpen={(doc, focus, terms) => setView({ doc, focus, terms: terms ?? [] })} />
          <div><h2 className="mb-2 text-[13px] font-semibold">Documents</h2>
            {d.length === 0 ? <EmptyState title="No documents yet" hint="Upload PDF, DOCX, TXT or Markdown files you are allowed to use. Research runs need at least one." action={<Button variant="primary" onClick={() => setUpload(true)}>Upload document</Button>} /> : <DocTable docs={d} onOpen={(doc) => setView({ doc, terms: [] })} />}
            {re.error && <p role="alert" className="mt-2 text-sm text-bad">{re.error.message}</p>}</div>
        </div>)}</Q>
      <Modal open={upload} onOpenChange={setUpload} title="Upload a document"><UploadForm projectId={projectId} onDone={() => setUpload(false)} /></Modal>
      {view && <DocPreview projectId={projectId} doc={view.doc} focus={view.focus} terms={view.terms} onClose={() => setView(null)} onReprocess={() => re.mutate(view.doc.id)} onDelete={() => setDel(view.doc)} />}
      <ConfirmDialog open={!!del} onOpenChange={(o) => !o && setDel(null)} title="Delete this document?" description={`"${del?.title || del?.filename}" and its passages will be removed from the library. Existing reports keep their stored citations.`} confirmLabel="Delete document" busy={rm.isPending} onConfirm={() => del && rm.mutate(del.id)} />
      {rm.error && <p role="alert" className="text-sm text-bad">{rm.error.message}</p>}
    </div>
  );
}

export default function Library() { return <RequireProject>{(p) => <Body projectId={p.id} />}</RequireProject>; }
