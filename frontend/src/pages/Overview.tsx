import { useQuery } from "@tanstack/react-query";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Link } from "react-router-dom";
import { Q } from "../components/common";
import { Badge, Button, Card, CardTitle, EmptyState, Mono, PageHeader, StatusBadge } from "../components/ui";
import { api } from "../lib/api";
import { CLAIM_SHORT, fmtDate, fmtDay, fmtNum, titleCase } from "../lib/format";
import { CLAIM_STATUSES, type AuditResponse, type ClaimStatus, type Overview as O, type ReportSummary } from "../lib/types";
import { useAuth } from "../auth";
import { useProject } from "../project";

const CLAIM_VAR: Record<ClaimStatus, string> = { SUPPORTED: "var(--ok)", PARTIALLY_SUPPORTED: "var(--warn)", CONTRADICTED: "var(--bad)", INSUFFICIENT_EVIDENCE: "var(--muted)", UNVERIFIED: "var(--line-strong)" };
const AXIS = { fontSize: 11, fill: "var(--muted)" } as const;

function Metric({ label, value, sub, to }: { label: string; value: number | string; sub?: string; to?: string }) {
  const body = (<><p className="text-[11px] font-medium uppercase tracking-wide text-muted">{label}</p><p className="mt-0.5 text-2xl font-semibold tabular-nums text-ink">{typeof value === "number" ? fmtNum(value) : value}</p>{sub && <p className="text-xs text-muted">{sub}</p>}</>);
  return to ? <Link to={to} className="block px-4 py-3 hover:bg-surface-2/60">{body}</Link> : <div className="px-4 py-3">{body}</div>;
}

/** Axis + gridlines are drawn from real data; the hidden table below each chart gives screen-reader users the same numbers. */
function DataTable({ caption, rows }: { caption: string; rows: [string, number][] }) {
  return <table className="sr-only"><caption>{caption}</caption><tbody>{rows.map(([k, v]) => <tr key={k}><th scope="row">{k}</th><td>{v}</td></tr>)}</tbody></table>;
}

function VerificationBar({ o }: { o: O }) {
  const total = o.claims_total;
  if (total === 0) return <EmptyState title="No claims checked yet" hint="Claims appear after a run's claims are extracted on the Evidence Verification page." />;
  return (
    <div>
      <div className="flex h-2.5 overflow-hidden rounded bg-surface-2" role="img" aria-label={`Claims by status: ${CLAIM_STATUSES.map((s) => `${CLAIM_SHORT[s]} ${o.claims[s]}`).join(", ")}`}>
        {CLAIM_STATUSES.map((s) => o.claims[s] > 0 && <div key={s} style={{ width: `${(o.claims[s] / total) * 100}%`, background: CLAIM_VAR[s] }} title={`${CLAIM_SHORT[s]}: ${o.claims[s]}`} />)}
      </div>
      <ul className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
        {CLAIM_STATUSES.map((s) => <li key={s} className="flex items-center justify-between gap-2"><span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-sm" style={{ background: CLAIM_VAR[s] }} aria-hidden />{CLAIM_SHORT[s]}</span><b className="tabular-nums">{o.claims[s]}</b></li>)}
      </ul>
      <p className="mt-2 text-xs text-muted">Status reflects lexical agreement with the uploaded library, not formal proof.</p>
    </div>
  );
}

export default function Overview() {
  const { user } = useAuth(); const { project } = useProject();
  const q = useQuery({ queryKey: ["overview", user?.id], queryFn: ({ signal }) => api.get<O>("/overview", { signal }) });
  const audit = useQuery({ queryKey: ["audit-recent", user?.id], queryFn: ({ signal }) => api.get<AuditResponse>("/audit", { signal, query: { limit: 8 } }) });
  const reports = useQuery({ queryKey: ["reports", project?.id], enabled: !!project, queryFn: ({ signal }) => api.get<ReportSummary[]>(`/projects/${project!.id}/reports`, { signal }) });
  return (
    <div>
      <PageHeader title="Overview" subtitle="Live totals from your own projects, runs, claims and reports." actions={<Link to="/workspace"><Button variant="primary">New research run</Button></Link>} />
      <Q q={q} label="overview" rows={6}>{(o) => {
        const series = o.runs_per_day.map((d) => ({ ...d, label: fmtDay(d.date) }));
        const taskData = Object.entries(o.task_status).map(([k, v]) => ({ name: titleCase(k), value: v }));
        const active = o.recent_runs.filter((r) => r.active);
        return (
          <div className="space-y-4">
            <div className="grid grid-cols-2 divide-x divide-y divide-line overflow-hidden rounded-lg border border-line bg-surface sm:grid-cols-3 lg:grid-cols-6 lg:divide-y-0">
              <Metric label="Projects" value={o.projects} to="/projects" />
              <Metric label="Research runs" value={o.runs} sub={`${o.active_runs} active · ${o.completed_runs} completed`} to="/workspace" />
              <Metric label="Documents" value={o.documents} to="/library" />
              <Metric label="Claims checked" value={o.claims_total} sub="lexical verification" to="/evidence" />
              <Metric label="Reports" value={o.reports} to="/reports" />
              <Metric label="Pending approvals" value={o.pending_approvals} to="/activity" />
            </div>

            <div className="grid gap-4 lg:grid-cols-3">
              <Card className="lg:col-span-2"><CardTitle right={<Link to="/workspace" className="text-xs text-accent hover:underline">Workspace</Link>}>Research runs</CardTitle>
                {o.recent_runs.length === 0 ? <EmptyState title="No runs yet" hint="Upload documents to the library, then start a run from the Research Workspace." action={<Link to="/library"><Button>Open library</Button></Link>} /> : (
                  <>
                    {active.length > 0 && <div className="mb-3 rounded-lg border border-accent/30 bg-accent-soft px-3 py-2 text-sm" role="status"><b>{active.length}</b> active run{active.length === 1 ? "" : "s"}: {active.map((r) => <Link key={r.id} to={`/agents/${r.id}`} className="ml-2 text-accent underline">{r.question.slice(0, 40)}</Link>)}</div>}
                    <ul className="divide-y divide-line">{o.recent_runs.map((r) => (
                      <li key={r.id} className="flex items-center justify-between gap-3 py-2"><div className="min-w-0"><Link to={`/workspace/${r.id}`} className="block truncate font-medium hover:text-accent">{r.question}</Link><p className="text-xs text-muted">{fmtDate(r.created_at)} · <Mono>{r.id.slice(0, 12)}</Mono></p></div><StatusBadge status={r.status} /></li>))}</ul>
                  </>)}
                <p className="mt-2 text-xs text-muted">{o.note}</p>
              </Card>
              <Card><CardTitle right={<Link to="/evidence" className="text-xs text-accent hover:underline">Evidence</Link>}>Evidence verification</CardTitle><VerificationBar o={o} /><DataTable caption="Claims by verification status" rows={CLAIM_STATUSES.map((s) => [CLAIM_SHORT[s], o.claims[s]])} /></Card>
            </div>

            <div className="grid gap-4 lg:grid-cols-2">
              <Card><CardTitle>Runs started per day, last 30 days</CardTitle>
                {o.runs === 0 ? <EmptyState title="No runs yet" /> : (
                  <div className="h-44"><ResponsiveContainer width="100%" height="100%"><BarChart data={series} margin={{ left: -18, right: 4, top: 4 }}>
                    <CartesianGrid vertical={false} stroke="var(--line)" /><XAxis dataKey="label" tick={AXIS} interval={4} tickLine={false} axisLine={{ stroke: "var(--line-strong)" }} />
                    <YAxis allowDecimals={false} tick={AXIS} tickLine={false} axisLine={false} /><Tooltip cursor={{ fill: "var(--surface-2)" }} contentStyle={{ fontSize: 12, borderRadius: 6, border: "1px solid var(--line-strong)", background: "var(--surface)", color: "var(--ink)" }} formatter={(v: number) => [v, "Runs"]} />
                    <Bar dataKey="runs" fill="var(--accent)" radius={[2, 2, 0, 0]} isAnimationActive={false} />
                  </BarChart></ResponsiveContainer></div>)}
                <DataTable caption="Runs per day" rows={o.runs_per_day.map((d) => [d.date, d.runs])} />
              </Card>
              <Card><CardTitle>Agent task status (latest runs)</CardTitle>
                {taskData.length === 0 ? <EmptyState title="No agent tasks yet" /> : (
                  <div className="h-44"><ResponsiveContainer width="100%" height="100%"><BarChart data={taskData} layout="vertical" margin={{ left: 8, right: 16, top: 4 }}>
                    <CartesianGrid horizontal={false} stroke="var(--line)" /><XAxis type="number" allowDecimals={false} tick={AXIS} tickLine={false} axisLine={false} /><YAxis type="category" dataKey="name" width={96} tick={AXIS} tickLine={false} axisLine={false} />
                    <Tooltip cursor={{ fill: "var(--surface-2)" }} contentStyle={{ fontSize: 12, borderRadius: 6, border: "1px solid var(--line-strong)", background: "var(--surface)", color: "var(--ink)" }} formatter={(v: number) => [v, "Tasks"]} />
                    <Bar dataKey="value" fill="var(--accent)" radius={[0, 2, 2, 0]} barSize={14} isAnimationActive={false} />
                  </BarChart></ResponsiveContainer></div>)}
                <DataTable caption="Tasks by status" rows={taskData.map((d) => [d.name, d.value])} />
              </Card>
            </div>

            <div className="grid gap-4 lg:grid-cols-3">
              <Card><CardTitle right={<Link to="/activity" className="text-xs text-accent hover:underline">All activity</Link>}>Research activity</CardTitle>
                <Q q={audit} label="activity">{(a) => a.entries.length === 0 ? <EmptyState title="No activity yet" /> : (
                  <ol className="relative space-y-2.5 border-l border-line pl-4">{a.entries.map((e) => (
                    <li key={e.id} className="relative"><span className="absolute -left-[21px] top-1.5 h-2 w-2 rounded-full border border-line-strong bg-surface" aria-hidden /><p className="text-sm">{titleCase(e.action)}</p><p className="text-xs text-muted">{fmtDate(e.at)}</p></li>))}</ol>)}</Q></Card>
              <Card><CardTitle right={<Link to="/reports" className="text-xs text-accent hover:underline">All reports</Link>}>Recent reports{project ? ` · ${project.title}` : ""}</CardTitle>
                {!project ? <EmptyState title="No project selected" /> : <Q q={reports} label="reports">{(r) => r.length === 0 ? <EmptyState title="No reports yet" hint="Generate one from a completed run." /> : (
                  <ul className="divide-y divide-line">{r.slice(0, 5).map((x) => <li key={x.id} className="py-2"><Link to={`/reports/${x.id}`} className="line-clamp-2 font-medium hover:text-accent">{x.title}</Link><p className="text-xs text-muted">v{x.version} · {fmtDate(x.updated_at)} · <Badge>{x.status.replace(/_/g, " ")}</Badge></p></li>)}</ul>)}</Q>}</Card>
              <Card><CardTitle right={<Link to="/projects" className="text-xs text-accent hover:underline">All projects</Link>}>Recent projects</CardTitle>
                {o.recent_projects.length === 0 ? <EmptyState title="No projects yet" action={<Link to="/projects"><Button variant="primary">Create a project</Button></Link>} /> : (
                  <ul className="divide-y divide-line">{o.recent_projects.map((p) => <li key={p.id} className="py-2"><Link to={`/projects/${p.id}`} className="font-medium hover:text-accent">{p.title}</Link><p className="truncate text-xs text-muted">{p.question}</p></li>)}</ul>)}</Card>
            </div>
          </div>
        );
      }}</Q>
    </div>
  );
}
