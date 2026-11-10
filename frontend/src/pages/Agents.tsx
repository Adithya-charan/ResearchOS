import { useQuery } from "@tanstack/react-query";
import { Background, Controls, Handle, MarkerType, Position, ReactFlow, type Edge, type Node, type NodeProps } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Q, RequireProject } from "../components/common";
import { ReliabilityPanel } from "../components/ReliabilityPanel";
import { Badge, Card, CardTitle, EmptyState, KV, Mono, Notice, PageHeader, Select, StatusBadge, Table } from "../components/ui";
import { api } from "../lib/api";
import { agentName, fmtDate, fmtDuration } from "../lib/format";
import type { AgentsResponse, Run, Task, TaskDetail } from "../lib/types";
import { useRunStream } from "../lib/useRunStream";
import { useRuns } from "../project";

const STATE_RULE: Record<string, string> = {
  succeeded: "var(--ok)", completed: "var(--ok)", running: "var(--accent)", failed: "var(--bad)", timed_out: "var(--bad)", cancelled: "var(--line-strong)",
  blocked: "var(--warn)", awaiting_approval: "var(--warn)", pending: "var(--line-strong)", queued: "var(--line-strong)",
};

function TaskNode({ data }: NodeProps) {
  const d = data as { task: Task; selected: boolean };
  const t = d.task;
  return (
    <div style={{ borderLeft: `3px solid ${STATE_RULE[t.status] ?? "var(--line-strong)"}` }}
      className={`h-[58px] w-52 rounded-md border border-line bg-surface px-2.5 py-1.5 text-xs ${d.selected ? "outline outline-2 outline-accent" : ""}`}>
      <Handle type="target" position={Position.Top} style={{ opacity: 0 }} />
      <p className="truncate font-medium text-ink">{agentName(t.agent_type)}</p>
      <p className="truncate text-muted" title={t.title}>{t.title}</p>
      <p className="text-[10.5px] text-muted">{t.status.replace(/_/g, " ")}</p>
      <Handle type="source" position={Position.Bottom} style={{ opacity: 0 }} />
    </div>
  );
}
const nodeTypes = { task: TaskNode };

/** Layered layout by dependency depth: purely presentational, derived from the real depends_on lists. */
export function layout(tasks: Task[]): { depth: Record<string, number> } {
  const byId = new Map(tasks.map((t) => [t.id, t]));
  const depth: Record<string, number> = {};
  const visit = (id: string, stack: Set<string>): number => {
    if (id in depth) return depth[id];
    if (stack.has(id)) return 0;
    stack.add(id);
    const t = byId.get(id);
    const d = t ? Math.max(-1, ...t.depends_on.filter((x) => byId.has(x)).map((x) => visit(x, stack))) + 1 : 0;
    stack.delete(id);
    return (depth[id] = d);
  };
  tasks.forEach((t) => visit(t.id, new Set()));
  return { depth };
}

function Graph({ tasks, selected, onSelect }: { tasks: Task[]; selected: string | null; onSelect: (id: string) => void }) {
  const { nodes, edges } = useMemo(() => {
    const { depth } = layout(tasks);
    const rows: Record<number, number> = {};
    const nodes: Node[] = tasks.map((t) => {
      const d = depth[t.id] ?? 0; const r = (rows[d] = (rows[d] ?? -1) + 1);
      return { id: t.id, type: "task", position: { x: r * 224, y: d * 86 }, data: { task: t, selected: t.id === selected } };
    });
    const ids = new Set(tasks.map((t) => t.id));
    const edges: Edge[] = tasks.flatMap((t) => t.depends_on.filter((x) => ids.has(x)).map((x) => ({ id: `${x}->${t.id}`, source: x, target: t.id, type: "smoothstep", markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14 }, style: { stroke: "var(--line-strong)", strokeWidth: 1.25 } })));
    return { nodes, edges };
  }, [tasks, selected]);
  return (
    <div className="h-[760px] rounded-lg border border-line bg-surface" role="region" aria-label={`Task dependency graph with ${tasks.length} tasks. The same data is listed in the table below.`}>
      <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView fitViewOptions={{ padding: 0.12 }} minZoom={0.3} nodesDraggable={false} nodesConnectable={false} edgesFocusable={false} onNodeClick={(_, n) => onSelect(n.id)} proOptions={{ hideAttribution: true }}>
        <Background /><Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}

function TaskPanel({ runId, taskId, all }: { runId: string; taskId: string; all: Task[] }) {
  const q = useQuery({ queryKey: ["task", runId, taskId], queryFn: ({ signal }) => api.get<TaskDetail>(`/runs/${runId}/tasks/${taskId}`, { signal }) });
  return (
    <Q q={q} label="task detail">{(t) => (
      <div className="space-y-3 text-sm">
        <KV items={[["Agent", agentName(t.agent_type)], ["Status", <StatusBadge key="s" status={t.status} />], ["Task", t.title], ["Depends on", t.depends_on.length ? t.depends_on.map((id) => { const d = all.find((x) => x.id === id); return d ? `#${d.seq} ${agentName(d.agent_type)}` : id; }).join(", ") : "none"], ["Duration", t.started_at && t.completed_at ? fmtDuration((Date.parse(t.completed_at) - Date.parse(t.started_at)) / 1000) : "-"], ["Retries", t.retry_count], ["Started", fmtDate(t.started_at)], ["Finished", fmtDate(t.completed_at)], ["Requires approval", t.requires_approval ? "yes" : "no"]]} />
        {t.error && <p role="alert" className="text-bad">{t.error}</p>}
        {t.output && <div><h3 className="font-medium">Output summary</h3><p className="text-muted">{t.output.summary}</p>
          {t.output.findings?.length > 0 && <ul className="mt-1 list-disc pl-5">{t.output.findings.map((f, i) => <li key={i}>{f}</li>)}</ul>}
          {t.output.uncertainties?.length > 0 && <><h3 className="mt-2 font-medium">Uncertainties</h3><ul className="list-disc pl-5 text-muted">{t.output.uncertainties.map((f, i) => <li key={i}>{f}</li>)}</ul></>}
          <h3 className="mt-2 font-medium">Evidence references ({t.output.evidence?.length ?? 0})</h3>
          {t.output.evidence?.length > 0 ? <ul className="mt-1 space-y-0.5 text-xs">{t.output.evidence.slice(0, 12).map((e, i) => <li key={i}><Mono>{e.passage_id}</Mono> <span className="text-muted">page {e.page_number ?? "-"}</span></li>)}</ul> : <p className="text-xs text-muted">None recorded for this task.</p>}</div>}
        {t.executions && t.executions.length > 0 && <div><h3 className="mb-1 font-medium">Executions</h3>
          <Table caption="Task executions" head={["Attempt", "Status", "Duration", "Error"]}>{t.executions.map((e) => <tr key={e.id}><td className="px-3 py-1.5">{e.attempt}</td><td className="px-3 py-1.5"><StatusBadge status={e.status} /></td><td className="px-3 py-1.5">{fmtDuration(e.duration_s)}</td><td className="px-3 py-1.5 text-bad">{e.error ?? ""}</td></tr>)}</Table></div>}
        <details><summary className="cursor-pointer font-medium">Input (raw)</summary><pre tabIndex={0} className="mt-1 max-h-48 overflow-auto rounded bg-surface-2 p-2 text-xs">{JSON.stringify(t.input ?? {}, null, 2)}</pre></details>
      </div>
    )}</Q>
  );
}

function Command({ runId }: { runId: string }) {
  const run = useQuery({ queryKey: ["run", runId], queryFn: ({ signal }) => api.get<Run>(`/runs/${runId}`, { signal }) });
  const tasks = useQuery({ queryKey: ["tasks", runId], queryFn: ({ signal }) => api.get<Task[]>(`/runs/${runId}/tasks`, { signal }) });
  const agents = useQuery({ queryKey: ["agents"], queryFn: ({ signal }) => api.get<AgentsResponse>("/agents", { signal }), staleTime: 300_000 });
  const [sel, setSel] = useState<string | null>(null);
  const active = run.data?.state.active ?? false;
  const stream = useRunStream(runId, active);
  return (
    <div className="space-y-4">
      <Card><div className="flex flex-wrap items-center gap-3 text-sm">
        {run.data && <><StatusBadge status={run.data.state.status} /><span className="min-w-0 flex-1 truncate" title={run.data.question}>{run.data.question}</span></>}
        <Badge tone={stream.status === "open" ? "ok" : "neutral"}>{active ? `Live: ${stream.status}` : "Not streaming (run not active)"}</Badge>
      </div></Card>
      <Q q={tasks} label="tasks">{(t) => t.length === 0 ? <EmptyState title="No tasks yet" hint="The task graph appears once the supervisor has planned the run." /> : (
        <div className="grid gap-4 xl:grid-cols-[1fr_22rem]">
          <div className="space-y-4">
            <Card><CardTitle>Task dependency graph</CardTitle><Graph tasks={t} selected={sel} onSelect={setSel} /></Card>
            <Card><CardTitle>Tasks by agent</CardTitle>
              <Table caption="Tasks by agent" head={["#", "Agent", "Task", "Status", "Depends on"]}>{t.map((x) => (
                <tr key={x.id} className={sel === x.id ? "bg-accent/5" : ""}><td className="px-3 py-2">{x.seq}</td><td className="px-3 py-2">{agentName(x.agent_type)}</td>
                  <td className="px-3 py-2"><button className="text-left text-accent underline" onClick={() => setSel(x.id)}>{x.title}</button></td><td className="px-3 py-2"><StatusBadge status={x.status} /></td>
                  <td className="px-3 py-2 text-xs text-muted">{x.depends_on.length || "-"}</td></tr>))}</Table></Card>
          </div>
          <Card className="h-fit xl:sticky xl:top-0"><CardTitle>{sel ? "Task detail" : "Select a task"}</CardTitle>{sel ? <TaskPanel runId={runId} taskId={sel} key={sel} all={t} /> : <p className="text-sm text-muted">Click a node or a task title to inspect its input, output, executions and errors.</p>}</Card>
        </div>)}</Q>
      <ReliabilityPanel runId={runId} active={active} />
      <Card><CardTitle>Agent roster</CardTitle>
        <Q q={agents} label="agents">{(a) => (
          <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">{a.agents.map((x) => (
            <div key={x.agent_type} className="rounded-lg border border-line p-3 text-sm"><p className="font-medium">{agentName(x.agent_type)}</p><p className="text-xs text-muted">{x.spec.role}</p>
              <p className="mt-1 text-xs text-muted">Timeout {x.spec.timeout_s ?? "-"} s · retries {x.spec.max_retries ?? "-"}</p></div>))}</div>)}</Q></Card>
      <Notice>Agents are deterministic, rule-based workers built on lexical retrieval over your library. Statuses and timings on this page come from the run's stored records and live stream.</Notice>
    </div>
  );
}

export default function Agents() {
  const { runId } = useParams(); const nav = useNavigate();
  return (
    <RequireProject>{(p) => <Inner projectId={p.id} runId={runId} nav={nav} />}</RequireProject>
  );
}

function Inner({ projectId, runId, nav }: { projectId: string; runId?: string; nav: (to: string) => void }) {
  const runs = useRuns(projectId);
  const chosen = runId ?? runs.data?.[0]?.id;
  return (
    <div className="space-y-4">
      <PageHeader title="Agent Command Center" subtitle="Inspect the agent task graph of a research run, with live updates while it is running." />
      <Q q={runs} label="runs">{(r) => r.length === 0 ? <EmptyState title="No research runs yet" hint="Start a run from the Research Workspace to see its agents here." /> : (
        <>
          <div className="max-w-md"><label htmlFor="run-pick" className="mb-1 block text-sm font-medium">Run</label>
            <Select id="run-pick" value={chosen} onChange={(e) => nav(`/agents/${e.target.value}`)}>{r.map((x) => <option key={x.id} value={x.id}>{x.question.slice(0, 70)} · {x.state.status}</option>)}</Select></div>
          {chosen && <Command runId={chosen} key={chosen} />}
        </>)}</Q>
    </div>
  );
}
