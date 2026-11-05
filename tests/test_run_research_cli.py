"""The runnable demo: scripts/run_research.py (kept small: it indexes only a few documents)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_research  # noqa: E402

CORPUS = ROOT / "datasets" / "raw" / "synthetic_v1"
pytestmark = pytest.mark.skipif(not CORPUS.is_dir(), reason="synthetic corpus not generated")


def args(tmp_path, *extra):
    return ["--corpus", str(CORPUS), "--max-docs", "250", "--db", str(tmp_path / "runs.db"), *extra]


def test_end_to_end_run_prints_evidence_uncertainty_and_limitations(tmp_path, capsys):
    out_json = tmp_path / "summary.json"
    code = run_research.main(args(tmp_path, "--json", str(out_json)))
    out = capsys.readouterr().out

    assert code == 0
    assert "Indexed 250 documents" in out and "run_finished" in out and "task_succeeded" in out  # live event stream
    assert "synthetic template corpus" in out  # the corpus' nature is never hidden
    assert "Evidence check:" in out and "Uncertainties:" in out and "Limitations:" in out
    assert "not proof" in out

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["status"] == "completed" and data["summary"]["limitations"]
    for ref in data["summary"]["evidence"]:
        assert ref["document_id"].startswith("syn-") and ref["passage_id"].startswith(ref["document_id"] + ":p")


def test_a_run_can_be_reopened_from_the_database(tmp_path, capsys):
    assert run_research.main(args(tmp_path)) == 0
    first = capsys.readouterr().out
    run_id = first.split("run_id: ")[1].split()[0]

    assert run_research.main(args(tmp_path, "--resume", run_id)) == 0
    again = capsys.readouterr().out
    assert f"Resuming run {run_id}" in again and "Run summary  [completed]" in again


def test_nonsense_questions_get_an_honest_negative_result(tmp_path, capsys):
    code = run_research.main(args(tmp_path, "--question", "quantum entanglement between neutron stars"))
    out = capsys.readouterr().out
    assert code == 0
    assert "Evidence check: INSUFFICIENT" in out
    assert "No close match was identified in the indexed corpus searched for this run" in out
    assert "This is not proof that the idea has never been published." in out


def test_bad_inputs_are_reported_not_crashed_on(tmp_path, capsys):
    assert run_research.main(["--corpus", str(tmp_path / "missing"), "--db", str(tmp_path / "x.db")]) == 2
    assert run_research.main(args(tmp_path, "--resume", "run_does_not_exist")) == 2
    assert "error:" in capsys.readouterr().err


def test_full_workflow_writes_a_report_and_it_can_be_exported_later(tmp_path, capsys):
    import export_report
    rep = tmp_path / "rep"
    assert run_research.main(args(tmp_path, "--workflow", "full", "--report-dir", str(rep))) == 0
    out = capsys.readouterr().out
    assert "Report: " in out and "human review: pending" in out
    md = (rep / "report.md").read_text(encoding="utf-8")
    assert "## Uncertainty" in md and ("not proof" in md or "not established" in md) and (rep / "report.json").is_file()

    run_id = out.split("run_id: ")[1].split()[0]
    again = tmp_path / "again"
    assert export_report.main(["--db", str(tmp_path / "runs.db"), "--run", run_id, "--out", str(again)]) == 0
    assert (again / "report.md").read_text(encoding="utf-8") == md
    assert export_report.main(["--db", str(tmp_path / "runs.db"), "--run", run_id, "--out", str(again), "--graph"]) == 0
    graph = json.loads((again / "graph.json").read_text(encoding="utf-8"))
    assert graph["nodes"] and graph["edges"] and (again / "graph.mmd").read_text(encoding="utf-8").startswith("graph LR")
    assert export_report.main(["--db", str(tmp_path / "runs.db"), "--run", "nope", "--out", str(again)]) == 2


def test_export_of_a_basic_run_explains_there_is_no_report(tmp_path, capsys):
    import export_report
    assert run_research.main(args(tmp_path)) == 0
    run_id = capsys.readouterr().out.split("run_id: ")[1].split()[0]
    assert export_report.main(["--db", str(tmp_path / "runs.db"), "--run", run_id, "--out", str(tmp_path / "o")]) == 1
    assert "no report" in capsys.readouterr().err
