import { Link } from "react-router-dom";
import { AgentOutputView } from "../components/AgentOutputView";
import { LibraryGaps } from "../components/LibraryGaps";
import { RequireProject } from "../components/common";
import { Badge, Card, EmptyState, PageHeader } from "../components/ui";
import { titleCase } from "../lib/format";
import { useAgentOutput } from "../lib/useAgentOutput";

interface Gap { id: string; kind: string; statement: string; term?: string; evidence?: { document_id: string; passage_id: string; page_number: number | null }[] }

function Body({ projectId }: { projectId: string }) {
  const res = useAgentOutput(projectId, "gap");
  return (
    <div className="space-y-4">
      <PageHeader title="Research Gaps" subtitle="Candidate gaps the Gap agent found in this project's library for a run. A gap describes the documents searched, not the literature as a whole." />
      <AgentOutputView res={res} what="gaps">{(d) => {
        const gaps = (d.gaps as Gap[] | undefined) ?? [];
        return gaps.length === 0 ? <EmptyState title="No gaps identified" hint="The agent found nothing the library leaves unaddressed for this question." /> : (
          <ul className="grid gap-3 md:grid-cols-2">{gaps.map((g) => (
            <li key={g.id}><Card className="h-full"><div className="mb-1 flex items-center gap-2"><Badge tone="violet">{g.id}</Badge><Badge>{titleCase(g.kind)}</Badge>{g.term && <Badge tone="info">term: {g.term}</Badge>}</div>
              <p className="text-sm">{g.statement}</p>
              <p className="mt-2 text-xs text-muted">Supporting passages: {g.evidence?.length ?? 0}</p>
              <p className="mt-1 text-xs"><Link className="text-accent underline" to="/evidence">Check claim "gap:{g.id}" on Evidence Verification</Link> · <Link className="text-accent underline" to="/hypotheses">Related hypotheses</Link></p></Card></li>))}</ul>);
      }}</AgentOutputView>
      <h2 className="pt-2 text-base font-semibold tracking-tight">Across the whole library</h2>
      <p className="-mt-2 max-w-3xl text-sm text-muted">Stated limitations, conflicting findings and themes found by comparing every document in the library, each linked to the passages it comes from.</p>
      <LibraryGaps projectId={projectId} />
    </div>
  );
}
export default function Gaps() { return <RequireProject>{(p) => <Body projectId={p.id} />}</RequireProject>; }
