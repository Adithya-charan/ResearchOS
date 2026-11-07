import { useState } from "react";
import { Link } from "react-router-dom";
import { AgentOutputView } from "../components/AgentOutputView";
import { RequireProject } from "../components/common";
import { Badge, Card, EmptyState, PageHeader, Select } from "../components/ui";
import { titleCase } from "../lib/format";
import { useAgentOutput } from "../lib/useAgentOutput";

interface Hyp { id: string; gap_id?: string; statement: string; assumptions?: string[]; falsification_criteria?: string; status: string; label?: string; evidence?: unknown[] }

function Body({ projectId }: { projectId: string }) {
  const res = useAgentOutput(projectId, "hypothesis");
  const [gapFilter, setGapFilter] = useState("");
  return (
    <div className="space-y-4">
      <PageHeader title="Hypotheses" subtitle="Untested proposals drafted by the Hypothesis agent. A hypothesis is not a finding; no experiment has been run for it." />
      <AgentOutputView res={res} what="hypotheses">{(d) => {
        const all = (d.hypotheses as Hyp[] | undefined) ?? [];
        const gaps = Array.from(new Set(all.map((h) => h.gap_id).filter(Boolean))) as string[];
        const shown = all.filter((h) => !gapFilter || h.gap_id === gapFilter);
        return all.length === 0 ? <EmptyState title="No hypotheses drafted" /> : (
          <div className="space-y-3">
            <div className="max-w-xs"><label htmlFor="h-gap" className="mb-1 block text-sm font-medium">Filter by gap</label><Select id="h-gap" value={gapFilter} onChange={(e) => setGapFilter(e.target.value)}><option value="">All gaps</option>{gaps.map((g) => <option key={g}>{g}</option>)}</Select></div>
            <ul className="grid gap-3 lg:grid-cols-2">{shown.map((h) => (
              <li key={h.id}><Card className="h-full space-y-2"><div className="flex flex-wrap items-center gap-2"><Badge tone="violet">{h.id}</Badge>{h.gap_id && <Badge>from gap {h.gap_id}</Badge>}<Badge tone="neutral">{titleCase(h.status)}</Badge></div>
                <p className="text-sm font-medium">{h.statement}</p>
                {h.assumptions && h.assumptions.length > 0 && <div><p className="text-xs font-semibold uppercase text-muted">Assumptions</p><ul className="list-disc pl-5 text-sm">{h.assumptions.map((a, i) => <li key={i}>{a}</li>)}</ul></div>}
                {h.falsification_criteria && <div><p className="text-xs font-semibold uppercase text-muted">Falsification criteria</p><p className="text-sm">{h.falsification_criteria}</p></div>}
                {h.label && <p className="text-xs text-muted">{h.label}</p>}
                <p className="text-xs"><Link className="text-accent underline" to="/evidence">Verify claim "hypothesis:{h.id}"</Link> · <Link className="text-accent underline" to="/experiments">Experiment plans</Link></p></Card></li>))}</ul>
          </div>);
      }}</AgentOutputView>
    </div>
  );
}
export default function Hypotheses() { return <RequireProject>{(p) => <Body projectId={p.id} />}</RequireProject>; }
