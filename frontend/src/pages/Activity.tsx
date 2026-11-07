import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Q, RequireProject } from "../components/common";
import { Badge, Button, Card, CardTitle, ClaimBadge, EmptyState, Select, StatusBadge, Table, Tabs } from "../components/ui";
import { PageHeader } from "../components/ui";
import { api } from "../lib/api";
import { fmtDate } from "../lib/format";
import type { Approval, AuditResponse, ClaimVerification, EngineEvent } from "../lib/types";
import { useRuns } from "../project";

const PAGE = 25;

function Audit() {
  const [action, setAction] = useState(""); const [offset, setOffset] = useState(0);
  const q = useQuery({ queryKey: ["audit", action, offset], queryFn: ({ signal }) => api.get<AuditResponse>("/audit", { signal, query: { action, limit: PAGE, offset } }) });
  return (
    <Q q={q} label="audit log">{(a) => (
      <div className="space-y-3">
        <div className="max-w-xs"><label htmlFor="au-action" className="mb-1 block text-sm font-medium">Action</label>
          <Select id="au-action" value={action} onChange={(e) => { setAction(e.target.value); setOffset(0); }}><option value="">All actions</option>{a.actions.map((x) => <option key={x}>{x}</option>)}</Select></div>
        {a.entries.length === 0 ? <EmptyState title="No audit entries" /> : (
          <Table caption="Your audit entries" head={["Time", "Action", "Target", "Detail"]}>{a.entries.map((e) => (
            <tr key={e.id}><td className="whitespace-nowrap px-3 py-2">{fmtDate(e.at)}</td><td className="px-3 py-2"><Badge>{e.action}</Badge></td><td className="px-3 py-2 font-mono text-xs">{e.target || "-"}</td><td className="max-w-xs truncate px-3 py-2 text-xs text-muted" title={JSON.stringify(e.detail)}>{Object.keys(e.detail).length ? JSON.stringify(e.detail) : ""}</td></tr>))}</Table>)}
        <div className="flex items-center justify-between text-sm"><span className="text-muted">{a.total === 0 ? 0 : offset + 1}–{Math.min(offset + PAGE, a.total)} of {a.total}. Only your own account's entries are shown.</span>
          <span className="flex gap-2"><Button size="sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>Previous</Button><Button size="sm" disabled={offset + PAGE >= a.total} onClick={() => setOffset(offset + PAGE)}>Next</Button></span></div>
      </div>)}</Q>
  );
}

function Approvals() {
  const [all, setAll] = useState(false);
  const q = useQuery({ queryKey: ["approvals", all ? "all" : "pending"], queryFn: ({ signal }) => api.get<Approval[]>("/approvals", { signal, query: { status: all ? "all" : "pending" } }) });
  return (
    <div className="space-y-3"><label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={all} onChange={(e) => setAll(e.target.checked)} />Include decided approvals</label>
      <Q q={q} label="approvals">{(l) => l.length === 0 ? <EmptyState title={all ? "No approvals" : "Nothing awaiting approval"} /> : (
        <Table caption="Approvals" head={["Requested", "Kind", "Item", "Decision", "Comment"]}>{l.map((a) => <tr key={`${a.kind}-${a.id}`}><td className="px-3 py-2">{fmtDate(a.requested_at)}</td><td className="px-3 py-2">{a.kind}</td><td className="px-3 py-2">{a.title}</td><td className="px-3 py-2"><StatusBadge status={a.decision ?? "pending"} /></td><td className="px-3 py-2 text-xs text-muted">{a.comment ?? ""}</td></tr>)}</Table>)}</Q>
      <p className="text-xs text-muted">Approve or reject a task from its run in the Research Workspace; experiments from the Experiments page.</p></div>
  );
}

function RunEvents({ projectId }: { projectId: string }) {
  const runs = useRuns(projectId); const [pick, setPick] = useState("");
  const id = pick || runs.data?.[0]?.id;
  const ev = useQuery({ queryKey: ["events", id], enabled: !!id, queryFn: ({ signal }) => api.get<EngineEvent[]>(`/runs/${id}/events`, { signal, query: { limit: 200 } }) });
  if (runs.data && runs.data.length === 0) return <EmptyState title="No runs yet" />;
  return (
    <div className="space-y-3"><div className="max-w-md"><label htmlFor="ev-run" className="mb-1 block text-sm font-medium">Run</label><Select id="ev-run" value={id ?? ""} onChange={(e) => setPick(e.target.value)}>{(runs.data ?? []).map((r) => <option key={r.id} value={r.id}>{r.question.slice(0, 70)}</option>)}</Select></div>
      <Q q={ev} label="events">{(l) => l.length === 0 ? <EmptyState title="No events" /> : (
        <Table caption="Run events" head={["#", "Time", "Type", "Task"]}>{l.map((e) => <tr key={e.id}><td className="px-3 py-2">{e.id}</td><td className="px-3 py-2">{fmtDate(e.created_at)}</td><td className="px-3 py-2">{e.type}</td><td className="px-3 py-2 font-mono text-xs">{e.task_id ?? "-"}</td></tr>)}</Table>)}</Q></div>
  );
}

function Verifs({ projectId }: { projectId: string }) {
  const q = useQuery({ queryKey: ["verifications", projectId], queryFn: ({ signal }) => api.get<ClaimVerification[]>(`/projects/${projectId}/verifications`, { signal }) });
  return <Q q={q} label="verifications">{(l) => l.length === 0 ? <EmptyState title="No verification rounds yet" /> : (
    <Table caption="Verification rounds" head={["Time", "Claim", "Round", "Status", "By"]}>{l.slice(0, 200).map((v) => <tr key={v.id}><td className="px-3 py-2">{fmtDate(v.created_at)}</td><td className="px-3 py-2 font-mono text-xs">{v.claim_id}</td><td className="px-3 py-2">{v.round}</td><td className="px-3 py-2"><ClaimBadge status={v.status} /></td><td className="px-3 py-2">{v.requested_by}</td></tr>)}</Table>)}</Q>;
}

export default function Activity() {
  const [tab, setTab] = useState<"audit" | "approvals" | "events" | "verifications">("audit");
  return (
    <div>
      <PageHeader title="Activity & Audit" subtitle="Your account's recorded actions, approvals, run events and verification rounds." />
      <Tabs label="Activity views" value={tab} onChange={setTab} tabs={[{ id: "audit", label: "Audit log" }, { id: "approvals", label: "Approvals" }, { id: "events", label: "Run events" }, { id: "verifications", label: "Verifications" }]} />
      <Card><CardTitle>{{ audit: "Audit log", approvals: "Approvals", events: "Run events", verifications: "Verification rounds" }[tab]}</CardTitle>
        {tab === "audit" && <Audit />}
        {tab === "approvals" && <Approvals />}
        {tab === "events" && <RequireProject>{(p) => <RunEvents projectId={p.id} />}</RequireProject>}
        {tab === "verifications" && <RequireProject>{(p) => <Verifs projectId={p.id} />}</RequireProject>}
      </Card>
    </div>
  );
}
