"""Knowledge graph built only from a finished report (and optional experiment manifests): every node and edge is a link
already recorded elsewhere - nothing is inferred. Node 'status' keeps the report's labels (verified / unsupported /
needs_review / candidate_gap / hypothesis), so the graph never upgrades a claim."""
from __future__ import annotations


def build_graph(report: dict, experiments: list[dict] | None = None) -> dict:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def node(nid: str, ntype: str, label: str, **extra) -> str:
        nodes.setdefault(nid, {"id": nid, "type": ntype, "label": label[:160], **extra})
        return nid

    def edge(a: str, b: str, kind: str) -> None:
        if {"source": a, "target": b, "type": kind} not in edges:
            edges.append({"source": a, "target": b, "type": kind})

    titles = report.get("titles", {})
    q = node("question", "question", report["question"])

    def doc(c: dict) -> str:
        d = c["document_id"]
        return node(d, "document", titles.get(d, d))

    for kind, items in report["findings"].items():
        for f in items:
            cid = node(f["claim_id"], "claim", f["text"], status="verified", claim_kind=kind, page=f["citation"]["page_number"],
                       passage_id=f["citation"]["passage_id"])
            edge(cid, doc(f["citation"]), "cited_from")
            edge(cid, q, "addresses")
    for n in report["needs_review"]:
        cid = node(n["claim_id"], "claim", n["text"], status="needs_review")
        if n["citation"]:
            edge(cid, doc(n["citation"]), "cited_from")
    for u in report["unsupported_claims"]:
        node(u["claim_id"], "claim", u["text"], status="unsupported")
    for t in report["tensions"]:
        if t["a"] in nodes and t["b"] in nodes:
            edge(t["a"], t["b"], "possible_tension")
    for g in report["candidate_gaps"]:
        gid = node(g["id"], "gap", g["statement"], status="candidate_gap")
        for ref in g["evidence"]:
            edge(gid, doc(ref), "evidenced_by")
        edge(gid, q, "gap_in")
    for h in report["hypotheses"]:
        hid = node(h["id"], "hypothesis", h["statement"], status="hypothesis")
        edge(hid, h["gap_id"], "derived_from")
    for m in experiments or []:
        eid = node(m["id"], "experiment", m["name"], status=m["status"], review=m.get("review", "pending"))
        if m.get("hypothesis_id") and m["hypothesis_id"] in nodes:
            edge(eid, m["hypothesis_id"], "tests")
    return {"question": report["question"], "nodes": list(nodes.values()), "edges": edges,
            "note": "Links come from the report and manifests; 'possible_tension' is a word-overlap flag, not a proven contradiction."}


def to_mermaid(graph: dict) -> str:
    ids = {n["id"]: f"n{i}" for i, n in enumerate(graph["nodes"])}
    lines = ["graph LR"]
    for n in graph["nodes"]:
        label = n["label"].replace('"', "'")[:60]
        lines.append(f'  {ids[n["id"]]}["{n["type"]}: {label}"]')
    for e in graph["edges"]:
        if e["source"] in ids and e["target"] in ids:
            lines.append(f'  {ids[e["source"]]} -->|{e["type"]}| {ids[e["target"]]}')
    return "\n".join(lines) + "\n"
