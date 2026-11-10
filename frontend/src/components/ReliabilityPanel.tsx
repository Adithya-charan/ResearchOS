import { useQuery } from "@tanstack/react-query";
import { Q } from "./common";
import { Badge, Card, CardTitle, Notice, Table } from "./ui";
import { api } from "../lib/api";
import { agentName, fmtDuration } from "../lib/format";
import type { RunReliability } from "../lib/types";

/** What happened to each agent in a run: attempts, retries, timeouts, failures traced to their root cause, and partial results. */
export function ReliabilityPanel({ runId, active }: { runId: string; active: boolean }) {
  const q = useQuery({ queryKey: ["reliability", runId], queryFn: ({ signal }) => api.get<RunReliability>(`/runs/${runId}/reliability`, { signal }), refetchInterval: active ? 3000 : false });
  return (
    <Card><CardTitle right={q.data && (q.data.degraded ? <Badge tone="bad">degraded</Badge> : q.data.finished ? <Badge tone="ok">all agents succeeded</Badge> : <Badge tone="info">in progress</Badge>)}>Execution reliability</CardTitle>
      <Q q={q} label="execution reliability">{(r) => (
        <div className="space-y-3 text-sm">
          <p className={r.degraded ? "text-bad" : "text-muted"}>{r.note}</p>
          <Table caption="Attempts, retries and timeouts per agent" head={["Agent", "Tasks", "Succeeded", "Failed", "Skipped", "Attempts", "Retries", "Timeouts", "Time"]}>
            {Object.entries(r.agents).map(([k, a]) => (
              <tr key={k}><td>{agentName(k)}</td><td>{a.tasks}</td><td>{a.succeeded}</td><td className={a.failed ? "text-bad" : ""}>{a.failed}</td><td className={a.skipped ? "text-warn" : ""}>{a.skipped}</td><td>{a.attempts}</td><td>{a.retries}</td><td>{a.timeouts}</td><td>{fmtDuration(a.duration_s)}</td></tr>))}
          </Table>
          {r.failures.length > 0 && <div><h3 className="mb-1 font-medium">Failures</h3><ul className="space-y-1">{r.failures.map((f) => (
            <li key={f.task_id} className="rounded border border-bad/30 p-2"><span className="font-medium">{agentName(f.agent_type)}</span> · {f.title} <Badge tone="bad">{f.status}</Badge>
              <p className="text-xs text-muted">{f.attempts} attempt(s), {f.retries_used} retr{f.retries_used === 1 ? "y" : "ies"}{f.timed_out ? ", timed out" : ""}</p>
              {f.error && <p className="mt-0.5 text-xs text-bad">{f.error}</p>}</li>))}</ul></div>}
          {r.skipped.length > 0 && <div><h3 className="mb-1 font-medium">Not run because an upstream task failed</h3><ul className="space-y-1">{r.skipped.map((s) => (
            <li key={s.task_id} className="rounded border border-warn/30 p-2"><span className="font-medium">{agentName(s.agent_type)}</span> · {s.title}
              <p className="text-xs text-muted">Root cause: {s.caused_by.length ? s.caused_by.map((c) => `${agentName(c.agent_type)} (${c.status})`).join(", ") : "unknown"}</p></li>))}</ul></div>}
          {r.degraded && r.partial_results.length > 0 && <p className="text-xs text-muted">{r.partial_results.length} task(s) finished and kept their output: {r.partial_results.map((p) => agentName(p.agent_type)).join(", ")}.</p>}
          <Notice>{r.report_available ? "A report can be built from this run." : "No report is available: reports are built only from a completed synthesis task, so a missing agent cannot silently leave a gap in one."}</Notice>
          <details className="text-xs text-muted"><summary className="cursor-pointer">What the engine guarantees</summary><ul className="mt-1 list-disc pl-5">{r.guarantees.map((g) => <li key={g}>{g}</li>)}</ul></details>
        </div>)}</Q>
    </Card>
  );
}
