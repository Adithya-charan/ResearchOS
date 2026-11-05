"""Extraction entry point. Parsing untrusted files is the riskiest thing the platform does, so by default it happens in a
separate Python process with a wall-clock timeout: a hang or a native crash in a parser then fails one file, not the app.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from research.corpus.extract_docx import extract_docx
from research.corpus.extract_pdf import DEFAULT_MAX_PAGES, extract_pdf
from research.corpus.extract_text import extract_html, extract_md, extract_txt
from research.corpus.models import ExtractedDocument, ExtractionError
from research.corpus.validate import DEFAULT_MAX_BYTES, detect_file_type

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_WORKER_OUTPUT = 400 * 1024 * 1024


def extract_in_process(path: Path, file_type: str, max_pages: int = DEFAULT_MAX_PAGES) -> ExtractedDocument:
    if file_type == "pdf":
        return extract_pdf(path, max_pages)
    if file_type == "docx":
        return extract_docx(path)
    if file_type == "md":
        return extract_md(path)
    if file_type == "html":
        return extract_html(path)
    if file_type == "txt":
        return extract_txt(path)
    raise ExtractionError("unsupported_type", f"no extractor for '{file_type}'")


def run_isolated(cmd: list[str], timeout_s: float) -> bytes:
    """Run a worker command, returning its stdout. Raises ExtractionError on timeout, crash or runaway output."""
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout_s, cwd=REPO_ROOT, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as e:
        raise ExtractionError("timeout", f"parser did not finish within {timeout_s:g}s and was stopped") from e
    except OSError as e:
        raise ExtractionError("worker_failed", f"could not start the parser process: {e}") from e
    if len(proc.stdout) > MAX_WORKER_OUTPUT:
        raise ExtractionError("worker_failed", "parser produced an implausible amount of output")
    if proc.returncode != 0 and not proc.stdout:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["no error output"]
        raise ExtractionError("worker_failed", f"parser process crashed (exit code {proc.returncode}): {tail[0][:200]}")
    return proc.stdout


def extract_isolated(path: Path, file_type: str, max_pages: int, timeout_s: float) -> ExtractedDocument:
    cmd = [sys.executable, "-E", "-m", "research.corpus.worker", str(path), file_type, str(max_pages)]
    out = run_isolated(cmd, timeout_s)
    try:
        payload = json.loads(out.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ExtractionError("worker_failed", "parser process returned unreadable output") from e
    try:
        if "error" in payload:
            raise ExtractionError(str(payload["error"]["code"]), str(payload["error"]["message"]))
        return ExtractedDocument.from_dict(payload["document"])
    except ExtractionError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as e:  # the worker is a separate process: trust nothing it says
        raise ExtractionError("worker_failed", "parser process returned a malformed result") from e


def extract_file(path: Path, *, max_bytes: int = DEFAULT_MAX_BYTES, max_pages: int = DEFAULT_MAX_PAGES,
                 isolate: bool = True, timeout_s: float = 90.0) -> ExtractedDocument:
    """Validate, then extract. Every failure is an ExtractionError with a stable `code`."""
    path = Path(path)
    file_type = detect_file_type(path, max_bytes)
    if isolate:
        return extract_isolated(path, file_type, max_pages, timeout_s)
    return extract_in_process(path, file_type, max_pages)
