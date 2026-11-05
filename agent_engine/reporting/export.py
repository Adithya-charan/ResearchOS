"""Report export: the report is always read back from the persisted synthesis output, never from memory of the run."""
from __future__ import annotations

import json
from pathlib import Path

from agent_engine.contracts import AgentOutput
from agent_engine.reporting.render import render_markdown
from agent_engine.state.store import Store


def report_from_store(store: Store, run_id: str) -> dict | None:
    """The report from the latest succeeded synthesis task of the run, or None if the workflow did not reach synthesis."""
    for t in reversed(store.list_tasks(run_id)):
        if t["agent_type"] == "synthesis" and t["status"] == "succeeded" and t["output"]:
            return AgentOutput.model_validate(t["output"]).data.get("report")
    return None


def export_report(report: dict, directory: str | Path, stem: str = "report") -> dict[str, Path]:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    paths = {"json": d / f"{stem}.json", "markdown": d / f"{stem}.md"}
    paths["json"].write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["markdown"].write_text(render_markdown(report), encoding="utf-8")
    return paths
