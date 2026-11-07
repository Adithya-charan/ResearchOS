import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router-dom";
import { AgentOutputView } from "../components/AgentOutputView";
import { Q, RequireProject } from "../components/common";
import { Badge, Button, Card, CardTitle, EmptyState, KV, Modal, Notice, PageHeader, StatusBadge, Table, Tabs } from "../components/ui";
import { api } from "../lib/api";
import { fmtDate, titleCase } from "../lib/format";
import type { ExperimentFull, ExperimentSummary } from "../lib/types";
import { useAgentOutput } from "../lib/useAgentOutput";

interface Plan { id: string; hypothesis_id?: string; hypothesis?: string; status: string; label?: string; needs_human_input?: string[]; independent_variable?: { name: string; levels?: string }; falsification_criteria?: string; datasets?: { document: string; quote: string }[] }

/** Lifecycle shown to the user, mapped from the real record: a plan is only a proposal until a human approves a concrete spec. */
export const stageOf = (e: Pick<ExperimentSummary, "status" | "approval">): string => {
  if (["running", "completed", "failed", "cancelled", "timed_out"].includes(e.status)) return e.status;
  if (e.approval === "approved") return "approved";
  return e.approval === "pending" ? "awaiting_approval" : "draft";
};

function Detail({ id, projectId, onClose }: { id: string; projectId: string; onClose: () => void }) {
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ["experiment", id], queryFn: ({ signal }) => api.get<ExperimentFull>(`/experiments/${id}`, { signal }), refetchInterval: (s) => (s.state.data?.status === "running" ? 3000 : false) });
  const logs = useQuery({ queryKey: ["experiment-logs", id], queryFn: ({ signal }) => api.get<unknown>(`/experiments/${id}/logs`, { signal }), enabled: !!q.data && ["completed", "failed", "running", "timed_out"].includes(q.data.status) });
  const refresh = () => { void qc.invalidateQueries({ queryKey: ["experiment", id] }); void qc.invalidateQueries({ queryKey: ["experiments", projectId] }); void qc.invalidateQueries({ queryKey: ["approvals"] }); void qc.invalidateQueries({ queryKey: ["overview"] }); };
  const ask = useMutation({ mutationFn: () => api.post(`/experiments/${id}/request-approval`), onSuccess: refresh });
  const run = useMutation({ mutationFn: () => api.post(`/experiments/${id}/execute`), onSuccess: refresh });
  return (
    <Modal open onOpenChange={(o) => !o && onClose()} title="Experiment">
      <Q q={q} label="experiment">{(e) => (
        <div className="space-y-3 text-sm">
          <KV items={[["Name", e.name], ["Stage", <StatusBadge key="s" status={stageOf(e)} />], ["Approval", e.approval], ["Spec hash", <code key="h" className="break-all text-xs">{e.spec_sha256}</code>], ["Created", fmtDate(e.created_at)], ["Review", e.review?.status ?? "-"]]} />
          {e.error && <p role="alert" className="text-bad">{e.error}</p>}
          {e.integrity && !e.integrity.ok && <Notice tone="warn">Integrity problems: {e.integrity.problems.join("; ")}</Notice>}
          {e.limitations?.length > 0 && <Notice>{e.limitations.join(" ")}</Notice>}
          <details><summary className="cursor-pointer font-medium">Specification</summary><pre tabIndex={0} className="mt-1 max-h-56 overflow-auto rounded bg-surface-2 p-2 text-xs">{JSON.stringify(e.spec, null, 2)}</pre></details>
          {e.manifest ? <details open><summary className="cursor-pointer font-medium">Recorded manifest</summary><pre tabIndex={0} className="mt-1 max-h-56 overflow-auto rounded bg-surface-2 p-2 text-xs">{JSON.stringify(e.manifest, null, 2)}</pre></details> : <p className="text-muted">No results: this experiment has not produced a recorded run.</p>}
          {logs.data !== undefined && <details><summary className="cursor-pointer font-medium">Logs</summary><pre tabIndex={0} className="mt-1 max-h-56 overflow-auto rounded bg-surface-2 p-2 text-xs">{typeof logs.data === "string" ? logs.data : JSON.stringify(logs.data, null, 2)}</pre></details>}
          {(ask.error || run.error) && <p role="alert" className="text-bad">{(ask.error ?? run.error)?.message}</p>}
          <div className="flex flex-wrap gap-2">
            {e.approval !== "approved" && e.approval !== "pending" && e.status === "draft" && <Button busy={ask.isPending} onClick={() => ask.mutate()}>Request approval</Button>}
            {e.approval === "pending" && <Link to="/activity"><Button>Decide in Activity &amp; approvals</Button></Link>}
            {e.approval === "approved" && e.status !== "running" && !e.manifest && <Button variant="primary" busy={run.isPending} onClick={() => run.mutate()}>Execute in sandbox</Button>}
          </div>
        </div>)}</Q>
    </Modal>
  );
}

function Body({ projectId }: { projectId: string }) {
  const [tab, setTab] = useState<"experiments" | "plans">("experiments"); const [open, setOpen] = useState<string | null>(null);
  const list = useQuery({ queryKey: ["experiments", projectId], queryFn: ({ signal }) => api.get<ExperimentSummary[]>(`/projects/${projectId}/experiments`, { signal }) });
  const plans = useAgentOutput(projectId, "experiment_design");
  return (
    <div className="space-y-4">
      <PageHeader title="Experiments" subtitle="Proposed plans are designs only. An experiment runs only after a human approves its exact specification, inside the sandbox." />
      <Tabs label="Experiment views" value={tab} onChange={setTab} tabs={[{ id: "experiments", label: "Experiments" }, { id: "plans", label: "Proposed plans" }]} />
      {tab === "experiments" && <Q q={list} label="experiments">{(l) => l.length === 0 ? <EmptyState title="No experiments yet" hint="Nothing has been run. Proposed plans from research runs are listed under “Proposed plans”; no results exist for them." /> : (
        <Table caption="Experiments" head={["Name", "Stage", "Updated", ""]}>{l.map((e) => <tr key={e.id}><td className="px-3 py-2 font-medium">{e.name}</td><td className="px-3 py-2"><StatusBadge status={stageOf(e)} /></td><td className="px-3 py-2">{fmtDate(e.updated_at)}</td><td className="px-3 py-2"><Button size="sm" onClick={() => setOpen(e.id)}>Details</Button></td></tr>)}</Table>)}</Q>}
      {tab === "plans" && <AgentOutputView res={plans} what="experiment plans">{(d) => {
        const ps = (d.plans as Plan[] | undefined) ?? [];
        return ps.length === 0 ? <EmptyState title="No plans proposed" /> : (
          <ul className="grid gap-3 lg:grid-cols-2">{ps.map((p) => (
            <li key={p.id}><Card className="h-full space-y-2"><div className="flex flex-wrap gap-2"><Badge tone="violet">{p.id}</Badge><Badge tone="warn">Proposed · not run</Badge>{p.hypothesis_id && <Badge>for {p.hypothesis_id}</Badge>}</div>
              {p.hypothesis && <p className="text-sm font-medium">{p.hypothesis}</p>}
              {p.independent_variable && <p className="text-sm">Vary: <b>{p.independent_variable.name}</b>{p.independent_variable.levels ? ` (${p.independent_variable.levels})` : ""}</p>}
              {p.falsification_criteria && <p className="text-sm text-muted">{p.falsification_criteria}</p>}
              {p.needs_human_input && p.needs_human_input.length > 0 && <p className="text-xs text-warn">Needs your input: {p.needs_human_input.map(titleCase).join(", ")}</p>}
              {p.label && <p className="text-xs text-muted">{p.label}</p>}</Card></li>))}</ul>);
      }}</AgentOutputView>}
      {open && <Detail id={open} projectId={projectId} onClose={() => setOpen(null)} />}
      <Card><CardTitle>Not available here</CardTitle><p className="text-sm text-muted">Writing and submitting experiment code is done through the API or CLI (<code>scripts/run_experiment.py</code>). This page shows records, approval state and recorded outputs only; it never displays an invented result.</p></Card>
    </div>
  );
}
export default function Experiments() { return <RequireProject>{(p) => <Body projectId={p.id} />}</RequireProject>; }
