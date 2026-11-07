import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import type { useAgentOutput } from "../lib/useAgentOutput";
import { EmptyState, ErrorState, Notice, Select, SkeletonBlock, StatusBadge } from "./ui";

type Res = ReturnType<typeof useAgentOutput>;

/** Shared frame for pages that read an agent's output from a run: run picker, loading/empty/error states, then content. */
export function AgentOutputView({ res, what, children }: { res: Res; what: string; children: (data: Record<string, unknown>) => ReactNode }) {
  if (res.runs.isLoading) return <SkeletonBlock label={`Loading ${what}`} />;
  if (res.error) return <ErrorState error={res.error} />;
  if (res.runList.length === 0) return <EmptyState title="No research runs yet" hint={`${what} come from the agents of a research run.`} action={<Link className="text-accent underline" to="/workspace">Start a research run</Link>} />;
  return (
    <div className="space-y-4">
      <div className="max-w-md"><label htmlFor="ao-run" className="mb-1 block text-sm font-medium">Research run</label>
        <Select id="ao-run" value={res.runId ?? ""} onChange={(e) => res.setRunId(e.target.value)}>{res.runList.map((r) => <option key={r.id} value={r.id}>{r.question.slice(0, 70)} · {r.state.status}</option>)}</Select></div>
      {res.tasks.isLoading ? <SkeletonBlock label={`Loading ${what}`} /> : !res.task ? (
        <EmptyState title={`No ${what} for this run`} hint="The agent that produces them has not run, or the run was cancelled before it." />
      ) : res.task.status !== "succeeded" ? (
        <EmptyState title={`The ${what} task is ${res.task.status.replace(/_/g, " ")}`} hint={res.task.error ?? "Output is available after the task succeeds."} icon={<StatusBadge status={res.task.status} />} />
      ) : res.detail.isLoading ? <SkeletonBlock label={`Loading ${what}`} /> : res.detail.data?.output ? <>{children(res.detail.data.output.data)}<Notice>Derived read-only from the run's agent output. Edit or approve nothing here; verification of each statement happens on the Evidence Verification page.</Notice></> : <EmptyState title="No output recorded" />}
    </div>
  );
}
