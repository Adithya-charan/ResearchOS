import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Pencil, Plus, Trash2 } from "lucide-react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { Link, useNavigate, useParams } from "react-router-dom";
import { z } from "zod";
import { Q } from "../components/common";
import { Button, Card, CardTitle, ConfirmDialog, EmptyState, Field, Input, KV, Modal, PageHeader, StatusBadge, Tabs, Textarea } from "../components/ui";
import { api } from "../lib/api";
import { fmtDate } from "../lib/format";
import type { CorpusStats, Project, ReportSummary } from "../lib/types";
import { qk, useProject, useRuns } from "../project";
import { useAuth } from "../auth";

export const projectSchema = z.object({
  title: z.string().trim().min(1, "Enter a title").max(200, "At most 200 characters"),
  question: z.string().trim().min(5, "Describe the research question (at least 5 characters)").max(2000, "At most 2000 characters"),
  description: z.string().max(5000, "At most 5000 characters").optional(),
});
type PForm = z.infer<typeof projectSchema>;

function ProjectForm({ initial, onDone, projectId }: { initial?: Project; onDone: (p: Project) => void; projectId?: string }) {
  const { user } = useAuth();
  const qc = useQueryClient();
  const f = useForm<PForm>({ resolver: zodResolver(projectSchema), defaultValues: { title: initial?.title ?? "", question: initial?.question ?? "", description: initial?.description ?? "" } });
  const m = useMutation({
    mutationFn: (v: PForm) => (projectId ? api.patch<Project>(`/projects/${projectId}`, v) : api.post<Project>("/projects", { ...v, description: v.description ?? "" })),
    onSuccess: (p) => { void qc.invalidateQueries({ queryKey: qk.projects(user?.id) }); void qc.invalidateQueries({ queryKey: ["overview"] }); onDone(p); },
  });
  const e = f.formState.errors;
  return (
    <form onSubmit={f.handleSubmit((v) => m.mutate(v))} className="space-y-3" noValidate>
      <Field label="Title" htmlFor="p-title" error={e.title?.message}><Input id="p-title" aria-invalid={!!e.title} {...f.register("title")} /></Field>
      <Field label="Research question" htmlFor="p-question" error={e.question?.message} hint="Runs use this question unless you enter another."><Textarea id="p-question" aria-invalid={!!e.question} {...f.register("question")} /></Field>
      <Field label="Description (optional)" htmlFor="p-desc" error={e.description?.message}><Textarea id="p-desc" {...f.register("description")} /></Field>
      {m.error && <p role="alert" className="text-sm text-bad">{m.error.message}</p>}
      <div className="flex justify-end"><Button type="submit" variant="primary" busy={m.isPending}>{projectId ? "Save changes" : "Create project"}</Button></div>
    </form>
  );
}

function ProjectDetail({ id }: { id: string }) {
  const nav = useNavigate(); const qc = useQueryClient(); const { user } = useAuth(); const { setProjectId } = useProject();
  const [tab, setTab] = useState<"summary" | "runs" | "corpus" | "reports">("summary");
  const [edit, setEdit] = useState(false); const [del, setDel] = useState(false);
  const q = useQuery({ queryKey: ["project", id], queryFn: ({ signal }) => api.get<Project>(`/projects/${id}`, { signal }) });
  const runs = useRuns(id);
  const corpus = useQuery({ queryKey: ["corpus", id], queryFn: ({ signal }) => api.get<CorpusStats>(`/projects/${id}/corpus`, { signal }), enabled: tab === "corpus" || tab === "summary" });
  const reports = useQuery({ queryKey: ["reports", id], queryFn: ({ signal }) => api.get<ReportSummary[]>(`/projects/${id}/reports`, { signal }), enabled: tab === "reports" });
  const rm = useMutation({
    mutationFn: () => api.del(`/projects/${id}`),
    onSuccess: () => { void qc.invalidateQueries({ queryKey: qk.projects(user?.id) }); qc.removeQueries({ queryKey: ["project", id] }); void qc.invalidateQueries({ queryKey: ["overview"] }); nav("/projects"); },
  });
  return (
    <Q q={q} label="project">{(p) => (
      <div>
        <PageHeader title={p.title} subtitle={p.question} actions={<>
          <Button onClick={() => { setProjectId(p.id); nav("/workspace"); }} variant="primary">Open workspace</Button>
          <Button onClick={() => setEdit(true)}><Pencil className="h-4 w-4" aria-hidden />Edit</Button>
          <Button variant="danger" onClick={() => setDel(true)}><Trash2 className="h-4 w-4" aria-hidden />Delete</Button></>} />
        <Tabs label="Project sections" value={tab} onChange={setTab} tabs={[{ id: "summary", label: "Summary" }, { id: "runs", label: "Research history" }, { id: "corpus", label: "Corpus" }, { id: "reports", label: "Reports" }]} />
        {tab === "summary" && <Card><KV items={[["Description", p.description || "-"], ["Created", fmtDate(p.created_at)], ["Updated", fmtDate(p.updated_at)], ["Runs", runs.data?.length ?? "…"], ["Library documents", corpus.data?.documents ?? "…"]]} /></Card>}
        {tab === "runs" && <Q q={runs} label="runs">{(r) => r.length === 0 ? <EmptyState title="No research runs yet" action={<Link to="/workspace"><Button variant="primary">Start research</Button></Link>} /> : (
          <ul className="divide-y divide-line rounded-xl border border-line bg-surface">{r.map((x) => <li key={x.id} className="flex items-center justify-between gap-3 p-3"><div className="min-w-0"><Link className="block truncate font-medium text-accent hover:underline" to={`/workspace/${x.id}`}>{x.question}</Link><p className="text-xs text-muted">{fmtDate(x.created_at)}</p></div><StatusBadge status={x.state.status} /></li>)}</ul>)}</Q>}
        {tab === "corpus" && <Q q={corpus} label="corpus">{(c) => <Card><CardTitle right={<Link to="/library" className="text-xs text-accent underline">Open library</Link>}>Corpus</CardTitle><KV items={[["Documents", c.documents], ["Passages", c.passages], ["Indexed documents", c.indexed_documents], ["Indexed passages", c.indexed_passages]]} /></Card>}</Q>}
        {tab === "reports" && <Q q={reports} label="reports">{(r) => r.length === 0 ? <EmptyState title="No reports yet" hint="Reports are generated from a completed research run." /> : (
          <ul className="divide-y divide-line rounded-xl border border-line bg-surface">{r.map((x) => <li key={x.id} className="flex items-center justify-between gap-3 p-3"><Link className="font-medium text-accent hover:underline" to={`/reports/${x.id}`}>{x.title}</Link><span className="text-xs text-muted">v{x.version} · {fmtDate(x.updated_at)}</span></li>)}</ul>)}</Q>}
        <Modal open={edit} onOpenChange={setEdit} title="Edit project"><ProjectForm initial={p} projectId={p.id} onDone={() => { setEdit(false); void q.refetch(); }} /></Modal>
        <ConfirmDialog open={del} onOpenChange={setDel} title="Delete this project?" busy={rm.isPending} confirmLabel="Delete project"
          description="This permanently deletes the project with its documents, runs, claims and reports. This cannot be undone." onConfirm={() => rm.mutate()} />
        {rm.error && <p role="alert" className="mt-2 text-sm text-bad">{rm.error.message}</p>}
      </div>
    )}</Q>
  );
}

export default function Projects() {
  const { projectId } = useParams();
  const { projects, loading, error, refetch, setProjectId } = useProject();
  const [open, setOpen] = useState(false); const [filter, setFilter] = useState(""); const nav = useNavigate();
  if (projectId) return <ProjectDetail id={projectId} />;
  const shown = projects.filter((p) => (p.title + " " + p.question).toLowerCase().includes(filter.toLowerCase()));
  return (
    <div>
      <PageHeader title="Projects" subtitle="A project holds one research question, its library, runs, claims and reports. Projects are private to your account."
        actions={<Button variant="primary" onClick={() => setOpen(true)}><Plus className="h-4 w-4" aria-hidden />New project</Button>} />
      <div className="mb-4 max-w-sm"><label htmlFor="proj-filter" className="sr-only">Search projects</label><Input id="proj-filter" placeholder="Search projects" value={filter} onChange={(e) => setFilter(e.target.value)} /></div>
      {loading ? <p role="status">Loading projects…</p> : error ? <p role="alert" className="text-bad">{(error as Error).message} <Button size="sm" onClick={refetch}>Retry</Button></p> : shown.length === 0 ? (
        <EmptyState title={projects.length ? "No projects match your search" : "No projects yet"} hint={projects.length ? undefined : "Create your first project to start a research run."}
          action={projects.length ? undefined : <Button variant="primary" onClick={() => setOpen(true)}>Create project</Button>} />
      ) : (
        <ul className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">{shown.map((p) => (
          <li key={p.id}><Card className="h-full"><Link to={`/projects/${p.id}`} className="text-base font-semibold text-accent hover:underline">{p.title}</Link>
            <p className="mt-1 line-clamp-3 text-sm text-muted">{p.question}</p><p className="mt-2 text-xs text-muted">Updated {fmtDate(p.updated_at)}</p>
            <div className="mt-3"><Button size="sm" onClick={() => { setProjectId(p.id); nav("/workspace"); }}>Open workspace</Button></div></Card></li>))}</ul>)}
      <Modal open={open} onOpenChange={setOpen} title="New project"><ProjectForm onDone={(p) => { setOpen(false); setProjectId(p.id); nav(`/projects/${p.id}`); }} /></Modal>
    </div>
  );
}
