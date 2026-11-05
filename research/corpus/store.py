"""SQLite persistence for documents and passages. Every passage row carries what blueprint 11.3 asks for: document id
(and through it the file hash and source/licence), page, section, text, extraction method and quality status."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from research.bm25.index import Passage
from research.corpus.models import DocumentRecord

SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_info (version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY, file_hash TEXT NOT NULL UNIQUE, filename TEXT NOT NULL, file_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL, title TEXT NOT NULL, title_source TEXT NOT NULL, authors_json TEXT NOT NULL,
    year INTEGER, year_source TEXT, doi TEXT, page_count INTEGER NOT NULL,
    extraction_method TEXT NOT NULL, extraction_status TEXT NOT NULL, quality_json TEXT NOT NULL,
    warnings_json TEXT NOT NULL, source TEXT, license TEXT, allowed_use_json TEXT NOT NULL,
    searchable INTEGER NOT NULL, ingested_at TEXT NOT NULL, stored_path TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS passages (
    passage_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL, page_number INTEGER, section TEXT NOT NULL, text TEXT NOT NULL,
    text_sha256 TEXT NOT NULL, quality TEXT NOT NULL, extraction_method TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_passages_doc ON passages(document_id, ordinal);
"""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _doc(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        id=row["id"], file_hash=row["file_hash"], filename=row["filename"], file_type=row["file_type"],
        size_bytes=row["size_bytes"], title=row["title"], title_source=row["title_source"],
        authors=json.loads(row["authors_json"]), year=row["year"], year_source=row["year_source"], doi=row["doi"],
        page_count=row["page_count"], extraction_method=row["extraction_method"],
        extraction_status=row["extraction_status"], quality=json.loads(row["quality_json"]),
        warnings=json.loads(row["warnings_json"]), source=row["source"], license=row["license"],
        allowed_use=json.loads(row["allowed_use_json"]), ingested_at=row["ingested_at"], stored_path=row["stored_path"])


def _passage(row: sqlite3.Row) -> Passage:
    return Passage(row["passage_id"], row["document_id"], row["text"], row["page_number"], row["section"], row["quality"])


class CorpusStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._depth = 0
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA busy_timeout = 5000")
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.executescript(SCHEMA)
            if self._conn.execute("SELECT COUNT(*) FROM schema_info").fetchone()[0] == 0:
                self._conn.execute("INSERT INTO schema_info VALUES (?)", (SCHEMA_VERSION,))

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            if self._depth:
                yield
                return
            self._conn.execute("BEGIN IMMEDIATE")
            self._depth = 1
            try:
                yield
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")
            finally:
                self._depth = 0

    def _all(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    # ------------------------------------------------------------------ writes
    def save_document(self, rec: DocumentRecord, passages: Iterable[Passage], method: str, replace: bool = False) -> None:
        """Insert a document and all its passages atomically (replacing an earlier version if asked)."""
        with self.transaction():
            if replace:
                self._conn.execute("DELETE FROM documents WHERE id=?", (rec.id,))
            self._conn.execute(
                "INSERT INTO documents (id, file_hash, filename, file_type, size_bytes, title, title_source, authors_json,"
                " year, year_source, doi, page_count, extraction_method, extraction_status, quality_json, warnings_json,"
                " source, license, allowed_use_json, searchable, ingested_at, stored_path)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rec.id, rec.file_hash, rec.filename, rec.file_type, rec.size_bytes, rec.title, rec.title_source,
                 json.dumps(rec.authors), rec.year, rec.year_source, rec.doi, rec.page_count, rec.extraction_method,
                 rec.extraction_status, json.dumps(rec.quality, sort_keys=True), json.dumps(rec.warnings), rec.source,
                 rec.license, json.dumps(rec.allowed_use), int(rec.searchable), rec.ingested_at, rec.stored_path))
            for i, p in enumerate(passages):
                self._conn.execute(
                    "INSERT INTO passages (passage_id, document_id, ordinal, page_number, section, text, text_sha256,"
                    " quality, extraction_method) VALUES (?,?,?,?,?,?,?,?,?)",
                    (p.passage_id, p.document_id, i, p.page_number, p.section, p.text, sha256_text(p.text), p.quality, method))

    def delete_document(self, doc_id: str) -> DocumentRecord | None:
        with self.transaction():
            rec = self.get_document(doc_id)
            if rec is not None:
                self._conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))  # passages cascade
            return rec

    # ------------------------------------------------------------------ reads
    def get_document(self, doc_id: str) -> DocumentRecord | None:
        row = self._one("SELECT * FROM documents WHERE id=?", (doc_id,))
        return _doc(row) if row else None

    def find_by_hash(self, file_hash: str) -> DocumentRecord | None:
        row = self._one("SELECT * FROM documents WHERE file_hash=?", (file_hash,))
        return _doc(row) if row else None

    def list_documents(self) -> list[DocumentRecord]:
        return [_doc(r) for r in self._all("SELECT * FROM documents ORDER BY ingested_at, id")]

    def document_ids_matching(self, *, file_types=None, year_min=None, year_max=None, licenses=None) -> set[str]:
        """Ids of searchable documents that satisfy the metadata filters (blueprint 10.1: filter by metadata and
        source permissions)."""
        where, params = ["searchable = 1"], []
        if file_types:
            where.append(f"file_type IN ({','.join('?' * len(file_types))})")
            params += sorted(file_types)
        if year_min is not None:
            where.append("year >= ?")
            params.append(year_min)
        if year_max is not None:
            where.append("year <= ?")
            params.append(year_max)
        if licenses:
            where.append(f"license IN ({','.join('?' * len(licenses))})")
            params += sorted(licenses)
        return {r["id"] for r in self._all(f"SELECT id FROM documents WHERE {' AND '.join(where)}", tuple(params))}

    def iter_searchable_passages(self) -> Iterator[Passage]:
        for row in self._all("SELECT p.* FROM passages p JOIN documents d ON d.id = p.document_id "
                             "WHERE d.searchable = 1 ORDER BY d.ingested_at, d.id, p.ordinal"):
            yield _passage(row)

    def get_passage(self, passage_id: str) -> tuple[Passage, int, str, str] | None:
        """-> (passage, ordinal, stored text sha256, extraction method)"""
        row = self._one("SELECT * FROM passages WHERE passage_id=?", (passage_id,))
        return (_passage(row), row["ordinal"], row["text_sha256"], row["extraction_method"]) if row else None

    def document_passages(self, doc_id: str, lo: int | None = None, hi: int | None = None) -> list[Passage]:
        sql, params = "SELECT * FROM passages WHERE document_id=?", [doc_id]
        if lo is not None:
            sql, params = sql + " AND ordinal >= ?", params + [lo]
        if hi is not None:
            sql, params = sql + " AND ordinal <= ?", params + [hi]
        return [_passage(r) for r in self._all(sql + " ORDER BY ordinal", tuple(params))]

    def passage_counts(self) -> dict[str, int]:
        return {r["document_id"]: r["n"] for r in self._all("SELECT document_id, COUNT(*) n FROM passages GROUP BY document_id")}

    def counts(self) -> dict[str, int]:
        d = self._one("SELECT COUNT(*) n, COALESCE(SUM(searchable), 0) s FROM documents")
        p = self._one("SELECT COUNT(*) n FROM passages")
        low = self._one("SELECT COUNT(*) n FROM passages WHERE quality != 'ok'")
        return {"documents": d["n"], "searchable_documents": d["s"], "passages": p["n"], "low_quality_passages": low["n"]}
