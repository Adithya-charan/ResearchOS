"""P11.7 evaluation: offline metrics (retrieval, evidence, citations, gaps) plus an end-to-end run through the real API for report
completeness and execution reliability. Writes artifacts/evaluation/p11_quality.json. Deterministic; no network and no model."""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from agent_engine.reporting.structured import SECTION_TITLES  # noqa: E402
from api.app import create_app  # noqa: E402
from api.config import Settings  # noqa: E402
from research.p11_evaluation import DATASET, run_offline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "evaluation" / "p11_quality.json"
PW = "correct horse battery 42"
TOPICS = [
    ("batteries", ["bat1", "bat2", "bat3", "bat4", "bat5"], "How does a carbon coating or an electrolyte additive change silicon anode swelling and capacity retention?"),
    ("soil", ["soil1", "soil2", "soil3", "soil4"], "What effect does biochar have on maize yield and soil nitrogen?"),
    ("federated", ["fl1", "fl2", "fl3", "fl4"], "How can federated learning reduce communication cost while limiting accuracy loss?"),
]


def end_to_end() -> dict:
    corpus = {d["id"]: d for d in json.loads((DATASET / "corpus.json").read_text("utf-8"))["docs"]}
    runs = []
    with tempfile.TemporaryDirectory(prefix="p11eval-", ignore_cleanup_errors=True) as td:  # Windows keeps SQLite files open until exit
        app = create_app(Settings(database_url="sqlite:///:memory:", engine_db=":memory:", data_dir=Path(td) / "data", corpus_isolate=False))
        with TestClient(app, base_url="http://testserver") as c:
            r = c.post("/auth/register", json={"email": "eval@example.org", "password": PW})
            h = {"X-CSRF-Token": r.json()["csrf_token"]}
            for name, docs, question in TOPICS:
                pid = c.post("/projects", json={"title": name, "question": question}, headers=h).json()["id"]
                for d in docs:
                    up = c.post(f"/projects/{pid}/documents", headers=h, data={"title": corpus[d]["title"], "license": "CC0-1.0", "source": "invented"},
                                files={"file": (f"{d}.md", corpus[d]["text"].encode(), "text/markdown")})
                    assert up.status_code == 201, up.text
                t0 = time.time()
                rid = c.post(f"/projects/{pid}/runs", json={"question": question}, headers=h).json()["id"]
                state = {}
                while time.time() - t0 < 180:
                    state = c.get(f"/runs/{rid}", headers=h).json()["state"]
                    if state["status"] in ("completed", "failed", "cancelled") and not state["active"]:
                        break
                    time.sleep(0.2)
                row = {"topic": name, "documents": len(docs), "run_status": state.get("status"), "seconds": round(time.time() - t0, 2)}
                rel = c.get(f"/runs/{rid}/reliability").json()
                row["reliability"] = {k: rel[k] for k in ("degraded", "tasks", "failures", "retried_tasks")}
                row["attempts"] = sum(a["attempts"] for a in rel["agents"].values())
                if state.get("status") == "completed":
                    rep = c.post(f"/runs/{rid}/reports", headers=h)
                    view = rep.json()
                    st = c.get(f"/reports/{view['id']}/structured").json()
                    s = st["structured"]
                    blocks = [b for sec in s["sections"] for b in sec["blocks"]]
                    ci = st["citation_integrity"]
                    row["report"] = {
                        "sections": len(s["sections"]), "all_18_titles_in_order": [x["title"] for x in s["sections"]] == list(SECTION_TITLES),
                        "empty_sections": [x["title"] for x in s["sections"] if not x["blocks"]],
                        "blocks": len(blocks), "blocks_labelled": sum(bool(b.get("epistemic")) for b in blocks),
                        "evidence_coverage": s["quality"]["evidence_coverage"], "statement_mix": s["quality"]["statement_mix"],
                        "citation_integrity": {k: ci[k] for k in ("citations_checked", "citations_valid", "integrity_rate", "claim_blocks",
                                                                  "claim_blocks_with_a_valid_citation", "ok")} | {"unsupported_claims": len(ci["unsupported_claims"])},
                    }
                runs.append(row)
    done = [r for r in runs if "report" in r]
    return {
        "runs": runs, "runs_started": len(runs), "runs_completed": sum(r["run_status"] == "completed" for r in runs),
        "execution_reliability": {"completion_rate": round(sum(r["run_status"] == "completed" for r in runs) / len(runs), 4),
                                  "runs_with_retries": sum(bool(r["reliability"]["retried_tasks"]) for r in runs),
                                  "runs_with_failures": sum(bool(r["reliability"]["failures"]) for r in runs),
                                  "note": f"{len(runs)} runs only, one per topic; this shows the scripted pipeline completes on clean input, not a failure-rate estimate."},
        "report_completeness": {"reports": len(done), "all_18_sections_in_order": sum(r["report"]["all_18_titles_in_order"] for r in done),
                                "with_empty_sections": sum(bool(r["report"]["empty_sections"]) for r in done),
                                "blocks_labelled": f"{sum(r['report']['blocks_labelled'] for r in done)}/{sum(r['report']['blocks'] for r in done)}",
                                "reports_with_citations": sum(r["report"]["citation_integrity"]["citations_checked"] > 0 for r in done),
                                "reports_without_citations": [r["topic"] for r in done if r["report"]["citation_integrity"]["citations_checked"] == 0],
                                "mean_citation_integrity_where_cited": (
                                    round(sum(rates) / len(rates), 4) if (rates := [r["report"]["citation_integrity"]["integrity_rate"] for r in done
                                                                                    if r["report"]["citation_integrity"]["integrity_rate"] is not None]) else None),
                                "note": "A report with no claims (the run found insufficient evidence) has no citation rate; it is listed, not counted as perfect or as zero."},
    }


def main() -> int:
    result = {"generated_by": "scripts/eval_p11.py", **run_offline(), "end_to_end": end_to_end()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    e, ci, g, e2e = result["evidence"], result["citations"]["citation_correctness"], result["gaps"], result["end_to_end"]
    print(f"evidence: accuracy {e['status_accuracy']} | support precision {e['evidence_support_precision']} | unsupported-claim rate {e['unsupported_claim_rate']}"
          f" | contradiction {e['contradiction_detection']}")
    print(f"citations: valid accepted {ci['valid_citations_accepted']} | defects flagged {ci['defective_citations_flagged']}")
    print(f"gaps: conflicts {g['conflict_detection']} | limitation recall {g['stated_limitation_recall']} | traceability problems {len(g['traceability_problems'])}")
    rc = e2e["report_completeness"]
    print(f"end-to-end: {e2e['runs_completed']}/{e2e['runs_started']} runs completed | 18 sections {rc['all_18_sections_in_order']}/{rc['reports']} | "
          f"labelled {rc['blocks_labelled']} | cited reports {rc['reports_with_citations']}/{rc['reports']} integrity {rc['mean_citation_integrity_where_cited']}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
