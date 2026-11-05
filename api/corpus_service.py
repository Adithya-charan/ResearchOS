"""Per-project document libraries. One `Corpus` (research/corpus) per project directory, guarded by a lock, so users'
documents never share storage or an index. Everything an upload contains is untrusted data."""
from __future__ import annotations

import logging
import re
import shutil
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path, PurePosixPath, PureWindowsPath

from api.config import Settings
from research.bm25.index import BM25Index
from research.corpus import Corpus
from research.corpus.models import IngestReport
from research.corpus.validate import EXT_TYPES

log = logging.getLogger("researchos.corpus")
Scanner = Callable[[Path], tuple[bool, str]]  # (clean, message) - plug in ClamAV or similar; None = not configured

_BAD_CHARS = re.compile(r"[^A-Za-z0-9._ \-()]")
# Phrases that address an AI agent instead of describing research. A heuristic flag for humans, never a filter that makes
# anything safe: the real defence is that no agent ever treats document text as an instruction.
_INJECTION = re.compile(
    r"(ignore|disregard|forget)\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules)"
    r"|you\s+are\s+now\s+(a|an|in)\b|system\s+prompt|\bas\s+an\s+ai\s+(language\s+)?model\b"
    r"|do\s+not\s+(tell|reveal|mention)\s+(the\s+)?user|new\s+instructions?\s*:|override\s+(your|the)\s+(rules|instructions)", re.I)


class UploadRejected(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def safe_filename(raw: str | None) -> str:
    """Basename only (both path styles), no control or shell-special characters, bounded length, never hidden or empty."""
    name = PureWindowsPath(PurePosixPath(raw or "").name).name
    name = _BAD_CHARS.sub("_", name).strip(" .")
    if not name or name.startswith("."):
        raise UploadRejected(400, "invalid file name")
    stem, dot, ext = name.rpartition(".")
    return (stem[:100] + dot + ext[:10]) if dot else name[:110]


def injection_hits(text: str) -> list[str]:
    return [m.group(0) for m in _INJECTION.finditer(text)][:3]


class CorpusManager:
    def __init__(self, settings: Settings, scanner: Scanner | None = None):
        self.settings, self.scanner = settings, scanner
        self._corpora: dict[str, Corpus] = {}
        self._locks: dict[str, threading.RLock] = {}
        self._guard = threading.Lock()

    def lock(self, project_id: str) -> threading.RLock:
        with self._guard:
            return self._locks.setdefault(project_id, threading.RLock())

    def root(self, project_id: str) -> Path:
        return self.settings.data_dir / "corpora" / project_id

    def exists(self, project_id: str) -> bool:
        return (self.root(project_id) / "corpus.db").is_file()

    def get(self, project_id: str) -> Corpus:
        with self._guard:
            c = self._corpora.get(project_id)
            if c is None:
                c = Corpus(self.root(project_id), max_bytes=self.settings.max_upload_bytes, isolate=self.settings.corpus_isolate)
                self._corpora[project_id] = c
            return c

    def drop(self, project_id: str) -> None:
        """Delete a project's whole library (used when the project is deleted)."""
        with self.lock(project_id):
            with self._guard:
                c = self._corpora.pop(project_id, None)
            if c is not None:
                c.close()
            shutil.rmtree(self.root(project_id), ignore_errors=True)

    def close(self) -> None:
        with self._guard:
            for c in self._corpora.values():
                c.close()
            self._corpora.clear()

    # ------------------------------------------------------------------ upload
    def stage(self, filename: str | None, stream_chunks, tmp_dir: Path) -> Path:
        """Write the upload to a scratch file, enforcing the size limit on the bytes actually received."""
        name = safe_filename(filename)
        ext = Path(name).suffix.lower()
        if ext == ".csv":
            raise UploadRejected(415, "CSV files are not supported yet")
        if ext not in EXT_TYPES:
            raise UploadRejected(415, f"unsupported file type '{ext or name}' (allowed: {', '.join(sorted(EXT_TYPES))})")
        dest = tmp_dir / name
        size, limit = 0, self.settings.max_upload_bytes
        with open(dest, "wb") as f:
            for chunk in stream_chunks:
                size += len(chunk)
                if size > limit:
                    raise UploadRejected(413, f"file exceeds the {limit // (1024 * 1024)} MB upload limit")
                f.write(chunk)
        if size == 0:
            raise UploadRejected(400, "file is empty")
        return dest

    def add(self, project_id: str, staged: Path, *, source: str | None, license: str | None,
            allowed_use: list[str]) -> tuple[IngestReport, str]:
        with self.lock(project_id):
            c = self.get(project_id)
            st = c.stats()
            if st["documents"] >= self.settings.max_documents_per_project:
                raise UploadRejected(429, f"document quota reached ({self.settings.max_documents_per_project})")
            stored = sum(d.size_bytes for d in c.list_documents())
            if stored + staged.stat().st_size > self.settings.max_corpus_bytes:
                raise UploadRejected(429, "project storage quota reached")
            scan = "not configured"
            if self.scanner is not None:
                clean, msg = self.scanner(staged)
                if not clean:
                    raise UploadRejected(422, f"file rejected by the malware scanner: {msg}")
                scan = "clean"
            report = c.ingest(staged, source=source, license=license, allowed_use=allowed_use)
            return report, scan

    def reprocess(self, project_id: str, doc_id: str) -> IngestReport | None:
        """Re-extract a document from its stored original (e.g. after an extractor improvement), keeping source and licence."""
        with self.lock(project_id):
            c = self.get(project_id)
            rec = c.get_document(doc_id)
            if rec is None:
                return None
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td) / rec.filename
                shutil.copyfile(c.root / rec.stored_path, tmp)
                return c.ingest(tmp, source=rec.source, license=rec.license, allowed_use=rec.allowed_use, force=True)

    def index_for(self, project_id: str) -> BM25Index | None:
        """A snapshot index of the project's searchable passages, or None when the project has no documents."""
        if not self.exists(project_id):
            return None
        with self.lock(project_id):
            idx = self.get(project_id).build_index()
        return idx if idx.num_passages else None
