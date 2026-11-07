import { zodResolver } from "@hookform/resolvers/zod";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { useForm } from "react-hook-form";
import { Link, useNavigate, useParams } from "react-router-dom";
import { z } from "zod";
import { Q, RequireProject } from "../components/common";
import { Badge, Button, Card, CardTitle, ClaimBadge, EmptyState, Field, Input, KV, Mono, Notice, PageHeader, Section, StatusBadge, Table, Textarea, useToast } from "../components/ui";
import { api } from "../lib/api";
import { agentName, fmtDate, fmtDuration, titleCase } from "../lib/format";
import type { Approval, ClaimList, EngineEvent, ReportSummary, Run, StructuredReport, Task } from "../lib/types";
import { useAgentOutput } from "../lib/useAgentOutput";
import { useRunStream } from "../lib/useRunStream";
import { qk, useRuns } from "../project";

export const runSchema = z.object({
  question: z.string().trim().min(3, "Enter at least 3 characters").max(2000, "At most 2000 characters"),
  max_runtime_s: z.coerce.number().min(10, "At least 10 seconds").max(3600, "At most 3600 seconds"),
  max_tasks: z.coerce.number().int().min(1, "At least 1").max(200, "At most 200"),
});
type RunForm = z.input<typeof runSchema>;

function StartRun({ projectId, defaultQuestion }: { projectId: string; defaultQuestion: string }) {
  const nav = useNavigate(); const qc = useQueryClient();
  const f = useForm<RunForm, unknown, z.output<typeof runSchema>>({ resolver: zodResolver(runSchema), defaultValues: { question: defaultQuestion, max_runtime_s: 300, max_tasks: 40 } });
  const m = useMutation({
    mutationFn: (v: z.output<typeof runSchema>) => api.post<Run>(`/projects/${projectId}/runs`, { question: v.question, start: true, limits: { max_runtime_s: v.max_runtime_s, max_tasks: v.max_tasks } }),
    onSuccess: (r) => { void qc.invalidateQueries({ queryKey: qk.runs(projectId) }); void qc.invalidateQueries({ queryKey: ["overview"] }); nav(`/workspace/${r.id}`); },
  });
  const e = f.formState.errors;
  const noCorpus = m.error && /corpus|document|searchable/i.test(m.error.message);
  return (
    <Card>
      <CardTitle>Start a research run</CardTitle>
      <form onSubmit={f.handleSubmit((v) => m.mutate(v))} className="space-y-3" noValidate>
        <Field label="Research question" htmlFor="r-question" error={e.question?.message}><Textarea id="r-question" aria-invalid={!!e.question} {...f.register("question")} /></Field>
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label="Maximum runtime (seconds)" htmlFor="r-rt" error={e.max_runtime_s?.message} hint="The server may lower this ceiling."><Input id="r-rt" type="number" aria-invalid={!!e.max_runtime_s} {...f.register("max_runtime_s")} /></Field>
          <Field label="Maximum tasks" htmlFor="r-mt" error={e.max_tasks?.message}><Input id="r-mt" type="number" aria-invalid={!!e.max_tasks} {...f.register("max_tasks")} /></Field>
        </div>
        <Notice>Agents read only this project&apos;s uploaded library. Findings come from deterministic, lexical retrieval and rule-based agents; they are leads to verify, not conclusions.</Notice>
        {m.error && <p role="alert" className="text-sm text-bad">{m.error.message}{noCorpus && <> <Link to="/library" className="underline">Upload documents in the library</Link>.</>}</p>}
        <Button type="submit" variant="primary" busy={m.isPending}>Start research</Button>
      </form>
    </Card>
  );
}

function ApprovalRow({ a, runId }: { a: Approval; runId: string }) {
  const qc = useQueryClient(); const toast = useToast(); const [comment, setComment] = useState("");
  const decide = useMutation({
    mutationFn: (ok: boolean) => api.post(`/approvals/${a.kind}/${a.id}/${ok ? "approve" : "reject"}`, { comment }),
    onSuccess: (_d, ok) => { toast("ok", ok ? "Approved." : "Rejected."); for (const k of [["approvals"], ["tasks", runId], ["run", runId], ["overview"]]) void qc.invalidateQueries({ queryKey: k }); },
  });
  const cid = `ap-${a.kind}-${a.id}`;
  return (
    <li className="space-y-2 py-3">
      <div className="flex flex-wrap items-center justify-between gap-2"><p className="text-sm font-medium">{a.title}</p><Badge tone="warn">{titleCase(a.kind)}</Badge></div>
      {a.review_hint && <p className="text-xs text-muted">{a.review_hint}</p>}
      <Field label="Comment (required to reject)" htmlFor={cid}><Input id={cid} value={comment} onChange={(e) => setComment(e.target.value)} /></Field>
      {decide.error && <p role="alert" className="text-sm text-bad">{decide.error.message}</p>}
      <div className="flex gap-2"><Button variant="primary" size="sm" busy={decide.isPending} onClick={() => decide.mutate(true)}>Approve</Button>
        <Button variant="danger" size="sm" busy={decide.isPending} disabled={!comment.trim()} onClick={() => decide.mutate(false)}>Reject</Button></div>
    </li>
  );
}

/** Progress is derived from the real task list: one step per agent, in the order its first task was planned. */
export function stagesOf(tasks: Task[]): { agent: string; status: string; done: number; total: number }[] {
  const order: string[] = []; const by: Record<string, Task[]> = {};
  for (const t of [...tasks].sort((a, b) => a.seq - b.seq)) { if (!by[t.agent_type]) { by[t.agent_type] = []; order.push(t.agent_type); } by[t.agent_type].push(t); }
  const rank = ["failed", "timed_out", "awaiting_approval", "running", "blocked", "queued", "pending", "cancelled", "succeeded"];
  return order.map((agent) => {
    const ts = by[agent]; const done = ts.filter((t) => t.status === "succeeded").length;
    const status = rank.find((r) => ts.some((t) => t.status === r)) ?? "pending";
    return { agent, status, done, total: ts.length };
  });
}

function Progress({ tasks }: { tasks: Task[] }) {
  const stages = useMemo(() => stagesOf(tasks), [tasks]);
  if (!stages.length) return <p className="text-sm text-muted">No tasks yet. The supervisor plans them when the run starts.</p>;
  const done = tasks.filter((t) => t.status === "succeeded").length;
  return (
    <div>
      <div className="mb-2 flex items-center justify-between text-xs text-muted"><span>{done} of {tasks.length} tasks finished</span><span>{Math.round((done / tasks.length) * 100)}%</span></div>
      <div className="h-1.5 overflow-hidden rounded bg-surface-2" role="progressbar" aria-valuemin={0} aria-valuemax={tasks.length} aria-valuenow={done} aria-label="Run progress"><div className="h-full bg-accent" style={{ width: `${(done / tasks.length) * 100}%` }} /></div>
      <ol className="mt-3 flex flex-wrap gap-1.5">{stages.map((s) => (
        <li key={s.agent} className="flex items-center gap-1.5 rounded-md border border-line px-2 py-1 text-xs"><span className="text-ink">{agentName(s.agent)}</span><StatusBadge status={s.status} /><span className="text-muted">{s.done}/{s.total}</span></li>))}</ol>
    </div>
  );
}

function RunView({ runId, projectId }: { runId: string; projectId: string }) {
  const qc = useQueryClient(); const toast = useToast(); const nav = useNavigate();
  const run = useQuery({ queryKey: ["run", runId], queryFn: ({ signal }) => api.get<Run>(`/runs/${runId}`, { signal }) });
  const tasks = useQuery({ queryKey: ["tasks", runId], queryFn: ({ signal }) => api.get<Task[]>(`/runs/${runId}/tasks`, { signal }) });
  const approvals = useQuery({ queryKey: ["approvals", "pending"], queryFn: ({ signal }) => api.get<Approval[]>("/approvals", { signal, query: { status: "pending" } }) });
  const claims = useQuery({ queryKey: ["claims", projectId, "run", runId], queryFn: ({ signal }) => api.get<ClaimList>(`/projects/${projectId}/claims`, { signal, query: { run_id: runId, limit: 100 } }) });
  const reports = useQuery({ queryKey: ["reports", projectId], queryFn: ({ signal }) => api.get<ReportSummary[]>(`/projects/${projectId}/reports`, { signal }) });
  const gaps = useAgentOutput(projectId, "gap", runId); const hyps = useAgentOutput(projectId, "hypothesis", runId);
  const myReport = (reports.data ?? []).find((r) => r.run_id === runId);
  const preview = useQuery({ queryKey: ["report-structured", myReport?.id], enabled: !!myReport, queryFn: ({ signal }) => api.get<StructuredReport>(`/reports/${myReport!.id}/structured`, { signal }) });
  const active = run.data?.state.active ?? false;
  const stream = useRunStream(runId, active);
  const ctl = useMutation({
    mutationFn: (verb: "pause" | "resume" | "cancel" | "recover") => api.post<Run>(`/runs/${runId}/${verb}`),
    onSuccess: (_d, verb) => { toast("ok", `Run ${verb === "recover" ? "recovery requested" : verb === "cancel" ? "cancelled" : verb + "d"}.`); void qc.invalidateQueries({ queryKey: ["run", runId] }); void qc.invalidateQueries({ queryKey: ["tasks", runId] }); },
  });
  const gen = useMutation({
    mutationFn: () => api.post<{ id: string }>(`/runs/${runId}/reports`),
    onSuccess: () => { toast("ok", "Report generated."); void qc.invalidateQueries({ queryKey: ["reports"] }); void qc.invalidateQueries({ queryKey: ["overview"] }); },
  });
  const mine = (approvals.data ?? []).filter((a) => a.run_id === runId);
  const gapList = ((gaps.detail.data?.output?.data.gaps as { id: string; statement: string }[] | undefined) ?? []);
  const hypList = ((hyps.detail.data?.output?.data.hypotheses as { id: string; statement: string; falsification_criteria?: string }[] | undefined) ?? []);
  return (
    <Q q={run} label="run">{(r) => {
      const s = r.state; const t = tasks.data ?? [];
      const failed = t.filter((x) => ["failed", "timed_out"].includes(x.status));
      const warnings = [...(s.job?.error ? [`Job error: ${s.job.error}`] : []), ...(s.stop_reason && s.stop_reason !== "completed" ? [`Stopped: ${s.stop_reason}`] : []), ...failed.map((x) => `${agentName(x.agent_type)} ${x.status.replace("_", " ")}${x.error ? `: ${x.error}` : ""}`)];
      return (
        <div>
          <PageHeader title="Research run" subtitle={r.question} actions={<>
            <Link to={`/agents/${runId}`}><Button>Agent command center</Button></Link>
            {s.active && s.control !== "paused" && <Button onClick={() => ctl.mutate("pause")} busy={ctl.isPending}>Pause</Button>}
            {s.active && s.control === "paused" && <Button onClick={() => ctl.mutate("resume")} busy={ctl.isPending}>Resume</Button>}
            {s.active && <Button variant="danger" onClick={() => ctl.mutate("cancel")} busy={ctl.isPending}>Cancel run</Button>}
            {!s.active && s.status !== "completed" && <Button onClick={() => ctl.mutate("recover")} busy={ctl.isPending}>Recover</Button>}
          </>} />
          {ctl.error && <p role="alert" className="mb-3 text-sm text-bad">{ctl.error.message}</p>}
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_20rem]">
            <div className="min-w-0 space-y-3">
              <Card><CardTitle>Progress</CardTitle><Q q={tasks} label="tasks">{(l) => <Progress tasks={l} />}</Q></Card>
              <Section title="Research question and scope" defaultOpen><KV items={[["Question", r.question], ["Workflow", r.workflow], ["Started", fmtDate(r.created_at)], ["Time limit reached", s.stop_reason === "max_runtime" ? "yes" : "no"]]} />
                <p className="mt-2 text-xs text-muted">Scope is this project&apos;s uploaded library only; nothing outside it is searched.</p></Section>
              <Section title="Literature and evidence" count={claims.data?.total} defaultOpen>
                <Q q={claims} label="claims">{(c) => c.claims.length === 0 ? <p className="text-sm text-muted">No claims extracted for this run yet. Extract them on the <Link className="text-accent underline" to="/evidence">Evidence Verification</Link> page.</p> : (
                  <ul className="divide-y divide-line">{c.claims.slice(0, 8).map((x) => <li key={x.id} className="flex items-start justify-between gap-3 py-2 text-sm"><span className="min-w-0">{x.text}<span className="block text-xs text-muted">{x.claim_key ?? x.claim_type} · {x.evidence_counts.supporting} supporting · {x.evidence_counts.contradicting} contradicting</span></span><ClaimBadge status={x.status} /></li>)}</ul>)}</Q>
                {(claims.data?.total ?? 0) > 8 && <p className="mt-2 text-xs"><Link className="text-accent underline" to="/evidence">See all {claims.data?.total} claims</Link></p>}
              </Section>
              <Section title="Research gaps" count={gapList.length || undefined}>{gapList.length === 0 ? <p className="text-sm text-muted">{gaps.task ? (gaps.task.status === "succeeded" ? "No gaps identified." : `Gap task is ${gaps.task.status}.`) : "The gap task has not run."}</p> : (
                <ul className="space-y-2 text-sm">{gapList.slice(0, 5).map((g) => <li key={g.id}><Mono>{g.id}</Mono> {g.statement}</li>)}</ul>)}<p className="mt-2 text-xs"><Link className="text-accent underline" to="/gaps">Open Research Gaps</Link></p></Section>
              <Section title="Hypotheses" count={hypList.length || undefined}>{hypList.length === 0 ? <p className="text-sm text-muted">{hyps.task ? (hyps.task.status === "succeeded" ? "No hypotheses drafted." : `Hypothesis task is ${hyps.task.status}.`) : "The hypothesis task has not run."}</p> : (
                <ul className="space-y-2 text-sm">{hypList.slice(0, 5).map((h) => <li key={h.id}><Mono>{h.id}</Mono> {h.statement}{h.falsification_criteria && <span className="block text-xs text-muted">Falsified if: {h.falsification_criteria}</span>}</li>)}</ul>)}<p className="mt-2 text-xs text-muted">Untested proposals, not findings.</p></Section>
              <Section title="Report preview" right={myReport ? <Badge>{myReport.status.replace(/_/g, " ")}</Badge> : undefined}>
                {!myReport ? <div className="text-sm text-muted">No report for this run yet. {!s.active && s.status === "completed" ? <Button variant="primary" size="sm" busy={gen.isPending} onClick={() => gen.mutate()}>Generate report</Button> : "Reports can be generated once the run completes."}{gen.error && <span role="alert" className="block text-bad">{gen.error.message}</span>}</div> : (
                  <Q q={preview} label="report preview">{(p) => <div><p className="text-sm font-medium">{p.structured.title}</p><ul className="mt-1 space-y-0.5 text-xs text-muted">{p.structured.sections.slice(0, 6).map((sec) => <li key={sec.id}>{sec.number}. {sec.title} — {sec.blocks.length} block{sec.blocks.length === 1 ? "" : "s"}</li>)}</ul><p className="mt-2 text-xs"><Link className="text-accent underline" to={`/reports/${myReport.id}`}>Read the full report</Link></p></div>}</Q>)}
              </Section>
              <Section title="Tasks" count={t.length}>{t.length === 0 ? <EmptyState title="No tasks yet" /> : (
                <Table caption="Run tasks" head={["#", "Agent", "Task", "Status"]}>{t.map((x) => <tr key={x.id}><td>{x.seq}</td><td>{agentName(x.agent_type)}</td><td>{x.title}{x.error && <p className="text-xs text-bad">{x.error}</p>}</td><td><StatusBadge status={x.status} /></td></tr>)}</Table>)}</Section>
              <Section title="Recent live events" count={stream.events.length || undefined}>
                {stream.events.length === 0 ? <p className="text-sm text-muted">{active ? "Waiting for events…" : "No live events (run not active). Stored events are on the Activity page."}</p> : (
                  <ul className="max-h-56 space-y-1 overflow-auto text-xs">{stream.events.slice().reverse().map((e: EngineEvent) => <li key={e.id} className="flex gap-2"><span className="text-muted">{fmtDate(e.created_at)}</span><span className="font-medium">{e.type}</span></li>)}</ul>)}</Section>
            </div>

            <aside className="min-w-0 space-y-3 lg:sticky lg:top-0 lg:self-start" aria-label="Run status">
              <Card><CardTitle>Run status</CardTitle>
                <KV items={[["Status", <StatusBadge key="s" status={s.status} />], ["Lifecycle", s.lifecycle ?? "-"], ["Control", s.control ?? "-"], ["Elapsed", fmtDuration(s.elapsed_s)], ["Job", s.job ? `${s.job.status} (${s.job.attempts}/${s.job.max_attempts})` : "-"]]} />
                <p className="mt-3 text-xs text-muted" role="status">{active ? `Live stream: ${stream.status}${stream.info ? ` (${stream.info})` : ""}` : "Not active; no live stream."}</p></Card>
              <Card><CardTitle>Tasks by status</CardTitle>{s.task_counts && Object.keys(s.task_counts).length ? <ul className="space-y-1">{Object.entries(s.task_counts).map(([k, v]) => <li key={k} className="flex items-center justify-between"><StatusBadge status={k} /><b className="tabular-nums">{v}</b></li>)}</ul> : <p className="text-sm text-muted">No tasks yet.</p>}</Card>
              <Card><CardTitle>Pending approvals</CardTitle>{mine.length === 0 ? <p className="text-sm text-muted">Nothing is waiting for you.</p> : <ul className="divide-y divide-line">{mine.map((a) => <ApprovalRow key={`${a.kind}-${a.id}`} a={a} runId={runId} />)}</ul>}</Card>
              <Card><CardTitle>Warnings</CardTitle>{warnings.length === 0 ? <p className="text-sm text-muted">No warnings recorded for this run.</p> : <ul className="list-disc space-y-1 pl-4 text-sm text-warn">{warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>}</Card>
              <div className="flex flex-wrap gap-2"><Button size="sm" onClick={() => nav("/workspace")}>All runs</Button>{!s.active && s.status === "completed" && !myReport && <Button size="sm" variant="primary" busy={gen.isPending} onClick={() => gen.mutate()}>Generate report</Button>}</div>
            </aside>
          </div>
        </div>
      );
    }}</Q>
  );
}

export default function Workspace() {
  const { runId } = useParams();
  return <RequireProject>{(p) => <WorkspaceBody projectId={p.id} question={p.question} runId={runId} />}</RequireProject>;
}

function WorkspaceBody({ projectId, question, runId }: { projectId: string; question: string; runId?: string }) {
  const runs = useRuns(projectId);
  if (runId) return <RunView runId={runId} projectId={projectId} />;
  return (
    <div>
      <PageHeader title="Research Workspace" subtitle="Start a research run on this project and follow it live." />
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_22rem]">
        <StartRun projectId={projectId} defaultQuestion={question} />
        <Card><CardTitle>Run history</CardTitle>
          <Q q={runs} label="runs">{(r) => r.length === 0 ? <EmptyState title="No research runs yet" /> : (
            <ul className="divide-y divide-line">{r.map((x) => <li key={x.id} className="flex items-center justify-between gap-3 py-2"><div className="min-w-0"><Link className="block truncate font-medium hover:text-accent" to={`/workspace/${x.id}`}>{x.question}</Link><p className="text-xs text-muted">{fmtDate(x.created_at)}</p></div><StatusBadge status={x.state.status} /></li>)}</ul>)}</Q></Card>
      </div>
    </div>
  );
}
