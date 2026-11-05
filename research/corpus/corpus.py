"""The corpus: ingest -> persist -> index -> search -> inspect -> verify.

A corpus lives in one folder:

    <root>/corpus.db                      documents + passages (SQLite)
    <root>/files/<ab>/<sha256>.<ext>      a private, content-addressed copy of every ingested original

Ingested files are copied first and everything is parsed from that copy, so the stored hash always matches the bytes that
were read. The BM25 index is rebuilt from the stored passages when the corpus is opened (about a second per 15,000
passages) and kept up to date as documents are added or removed.
"""
from __future__ import annotations

import datetime
import gc
import hashlib
import math
import os
import uuid
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from research.bm25.analysis import STOPWORDS, raw_tokens, stem
from research.bm25.index import BM25Index, Passage
from research.corpus.extract import extract_file
from research.corpus.extract_pdf import DEFAULT_MAX_PAGES
from research.corpus.metadata import extract_metadata
from research.corpus.models import DocumentRecord, ExtractedDocument, ExtractionError, IngestReport
from research.corpus.quality import assess_text, summarize
from research.corpus.segment import build_passages, detect_headings, squash, strip_margin_furniture
from research.corpus.store import CorpusStore, sha256_text
from research.corpus.validate import DEFAULT_MAX_BYTES, EXT_TYPES, detect_file_type

ALLOWED_USES = frozenset({"search", "quote"})


@dataclass(frozen=True)
class SearchFilters:
    """Metadata filters. Documents whose year is unknown never match a year filter. `allowed_use` is always enforced:
    documents without 'search' permission are not in the index at all."""

    document_ids: frozenset[str] | None = None
    file_types: frozenset[str] | None = None
    year_min: int | None = None
    year_max: int | None = None
    licenses: frozenset[str] | None = None
    exclude_low_quality: bool = False  # drop passages from pages whose text extraction looks unreliable


@dataclass(frozen=True)
class CorpusHit:
    passage: Passage
    score: float
    matched_terms: tuple[str, ...]
    document: DocumentRecord


@dataclass
class Verification:
    ok: bool
    checks: dict[str, bool | None]
    message: str


@dataclass
class Inspection:
    passage: Passage
    document: DocumentRecord
    ordinal: int
    text_sha256: str
    extraction_method: str
    before: list[Passage] = field(default_factory=list)
    after: list[Passage] = field(default_factory=list)
    verification: Verification | None = None


class Corpus:
    def __init__(self, root: str | Path, *, max_bytes: int = DEFAULT_MAX_BYTES, max_pages: int = DEFAULT_MAX_PAGES,
                 isolate: bool = True, timeout_s: float = 90.0, max_passage_words: int = 110):
        self.root = Path(root)
        self.files_dir = self.root / "files"
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes, self.max_pages, self.isolate, self.timeout_s = max_bytes, max_pages, isolate, timeout_s
        self.max_passage_words = max_passage_words
        self.store = CorpusStore(self.root / "corpus.db")
        self.index = BM25Index()
        self._reextracted: OrderedDict[str, ExtractedDocument] = OrderedDict()
        self.rebuild_index()

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> Corpus:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def rebuild_index(self) -> None:
        self.index.clear()
        self.index.add_passages(self.store.iter_searchable_passages())

    # ------------------------------------------------------------------ ingestion
    def ingest(self, path: str | Path, *, source: str | None = None, license: str | None = None,
               allowed_use: tuple[str, ...] | list[str] = ("search", "quote"), metadata: dict | None = None,
               force: bool = False) -> IngestReport:
        """Ingest one file. Never raises for a bad *file*: the report says what happened and why. `source` / `license`
        record where the file came from and what it may be used for; `metadata` (title, authors, year, doi) overrides
        anything extracted and is labelled as coming from the user."""
        path = Path(path)
        report = IngestReport(str(path), "rejected")
        bad_use = set(allowed_use) - ALLOWED_USES
        if bad_use:
            raise ValueError(f"unknown allowed_use {sorted(bad_use)}; choose from {sorted(ALLOWED_USES)}")
        try:
            detect_file_type(path, self.max_bytes)
            file_hash, stored = self._stash(path)
        except ExtractionError as e:
            return self._reject(report, e)
        except OSError as e:
            report.status, report.code, report.message = "failed", "corrupt", f"cannot read file: {e}"
            return report

        doc_id = f"doc-{file_hash[:12]}"
        existing = self.store.find_by_hash(file_hash)
        if existing is not None and not force:
            report.status, report.document_id = "duplicate", existing.id
            report.message = f"already ingested as {existing.id} ('{existing.title}')"
            return report

        try:
            extracted = extract_file(stored, max_bytes=self.max_bytes, max_pages=self.max_pages,
                                     isolate=self.isolate, timeout_s=self.timeout_s)
        except ExtractionError as e:
            if existing is None:
                self._discard(stored)
            return self._reject(report, e)

        warnings = list(extracted.warnings)
        removed = strip_margin_furniture(extracted) if extracted.file_type == "pdf" else 0
        if removed:
            warnings.append(f"removed {removed} repeated header/footer/page-number block(s)")
        detect_headings(extracted)
        qualities = [(p.number, assess_text("\n".join(b.text for b in p.blocks), p.has_images)) for p in extracted.pages]
        summary = summarize(qualities)
        passages = build_passages(doc_id, extracted, dict(qualities), self.max_passage_words)
        # Safety net: a passage that cannot be found on its own page of the extracted text could never be verified later,
        # so it is not stored (extractors and segmentation should make this impossible; this proves it per document).
        page_text = {p.number: squash(p.raw_text) for p in extracted.pages}
        unmatched = {p.passage_id for p in passages if squash(p.text) not in page_text.get(p.page_number, "")}
        if unmatched:
            passages = [p for p in passages if p.passage_id not in unmatched]
            warnings.append(f"{len(unmatched)} passage(s) could not be matched back to their page text and were not indexed")
        report.pages, report.passages, report.quality_status = len(extracted.pages), len(passages), summary["status"]
        report.pages_without_text, report.pages_low_quality = summary["pages_no_text"], summary["pages_low"]
        if not passages:
            if existing is None:
                self._discard(stored)
            blanks = len(summary["pages_blank"])
            return self._reject(report, ExtractionError(
                "no_text", "no readable text found" + (f" ({len(summary['pages_no_text'])} page(s) are images without a text "
                                                       "layer - OCR is not available in this version)" if summary["pages_no_text"]
                                                       else f" ({blanks} blank page(s))" if blanks else "")))
        if summary["pages_no_text"]:
            warnings.append(f"pages {_ranges(summary['pages_no_text'])} have no text layer (scanned?) and were NOT indexed; "
                            "OCR is not available")
        if summary["pages_low"]:
            warnings.append(f"pages {_ranges(summary['pages_low'])} have unreliable extracted text; their passages are "
                            "flagged low quality")
        try:
            meta = extract_metadata(extracted, path.name, metadata)
        except (ValueError, TypeError) as e:
            if existing is None:
                self._discard(stored)
            return self._reject(report, ExtractionError("invalid_metadata", str(e)))

        rec = DocumentRecord(
            id=doc_id, file_hash=file_hash, filename=path.name, file_type=extracted.file_type, size_bytes=stored.stat().st_size,
            title=meta["title"], title_source=meta["title_source"], authors=meta["authors"], year=meta["year"],
            year_source=meta["year_source"], doi=meta["doi"], page_count=len(extracted.pages),
            extraction_method=extracted.method, extraction_status="ok" if summary["status"] == "ok" else "partial",
            quality=summary, warnings=warnings, source=source, license=license, allowed_use=list(allowed_use),
            ingested_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            stored_path=stored.relative_to(self.root).as_posix())
        self.store.save_document(rec, passages, extracted.method, replace=existing is not None)
        if existing is not None:
            self.rebuild_index()
        elif rec.searchable:
            self.index.add_passages(passages)
        report.status, report.document_id, report.warnings = "ingested", doc_id, warnings
        report.message = f"'{rec.title}': {len(passages)} passages from {rec.page_count} page(s)"
        return report

    def ingest_many(self, paths: list[str | Path], **kwargs) -> list[IngestReport]:
        """Ingest files; directories are searched recursively for supported types (others are skipped silently, while
        an unsupported file named explicitly is reported as rejected)."""
        reports = []
        for p in map(Path, paths):
            if p.is_dir():
                files = sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in EXT_TYPES)
            else:
                files = [p]
            reports += [self.ingest(f, **kwargs) for f in files]
        return reports

    def _stash(self, path: Path) -> tuple[str, Path]:
        """Copy to a content-addressed file while hashing; the copy is what gets parsed."""
        tmp = self.files_dir / f".incoming-{uuid.uuid4().hex}"
        h, size = hashlib.sha256(), 0
        try:
            with path.open("rb") as src, tmp.open("wb") as dst:
                for chunk in iter(lambda: src.read(1 << 20), b""):
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise ExtractionError("too_large", f"file is larger than the {self.max_bytes / 1e6:.0f} MB limit")
                    h.update(chunk)
                    dst.write(chunk)
            digest = h.hexdigest()
            final = self.files_dir / digest[:2] / f"{digest}{path.suffix.lower()}"
            final.parent.mkdir(exist_ok=True)
            if final.exists():
                tmp.unlink()
            else:
                os.replace(tmp, final)
            return digest, final
        finally:
            tmp.unlink(missing_ok=True)

    def _discard(self, stored: Path) -> None:
        """Best-effort removal of a stored copy that did not become a document. An orphan is harmless (it is re-used if
        the same file is ingested again), so a file that is still locked - Windows - never turns a rejection into a crash."""
        for attempt in range(2):
            try:
                stored.unlink(missing_ok=True)
                break
            except OSError:
                if attempt == 0:
                    gc.collect()  # release handles held by half-finished parser objects, then retry once
        try:
            stored.parent.rmdir()
        except OSError:
            pass

    @staticmethod
    def _reject(report: IngestReport, e: ExtractionError) -> IngestReport:
        report.status = "failed" if e.code in ("timeout", "worker_failed") else "rejected"
        report.code, report.message = e.code, e.message
        return report

    # ------------------------------------------------------------------ retrieval
    def _matching_ids(self, f: SearchFilters) -> set[str] | None:
        """Ids of the documents allowed by the metadata filters; None means 'no restriction'."""
        if f.document_ids is None and not f.file_types and f.year_min is None and f.year_max is None and not f.licenses:
            return None
        ids = self.store.document_ids_matching(file_types=f.file_types, year_min=f.year_min, year_max=f.year_max,
                                               licenses=f.licenses)
        return ids & set(f.document_ids) if f.document_ids is not None else ids

    def build_index(self, filters: SearchFilters | None = None) -> BM25Index:
        """A separate in-memory index limited to the documents (and, optionally, only the cleanly extracted passages)
        that match `filters` - what an agent run searches when it is scoped to part of the library."""
        f = filters or SearchFilters()
        ids = self._matching_ids(f)
        index = BM25Index()
        index.add_passages(p for p in self.store.iter_searchable_passages()
                           if (ids is None or p.document_id in ids) and not (f.exclude_low_quality and p.quality != "ok"))
        return index

    def search(self, query: str, k: int = 8, filters: SearchFilters | None = None) -> list[CorpusHit]:
        f = filters or SearchFilters()
        ids = self._matching_ids(f)
        if ids is not None and not ids:
            return []
        hits = self.index.search(query, k=k, document_ids=ids,
                                 predicate=(lambda p: p.quality == "ok") if f.exclude_low_quality else None)
        docs: dict[str, DocumentRecord] = {}
        out = []
        for h in hits:
            if h.passage.document_id not in docs:
                docs[h.passage.document_id] = self.store.get_document(h.passage.document_id)  # type: ignore[assignment]
            out.append(CorpusHit(h.passage, h.score, h.matched_terms, docs[h.passage.document_id]))
        return out

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        return self.store.get_document(doc_id)

    def list_documents(self) -> list[DocumentRecord]:
        return self.store.list_documents()

    def stats(self) -> dict:
        return {**self.store.counts(), "indexed_documents": self.index.num_documents, "indexed_passages": self.index.num_passages}

    def inspect(self, passage_id: str, context: int = 1, verify: bool = True) -> Inspection | None:
        """The passage, its document, its neighbours, and (optionally) a fresh check against the stored original."""
        got = self.store.get_passage(passage_id)
        if got is None:
            return None
        passage, ordinal, sha, method = got
        doc = self.store.get_document(passage.document_id)
        assert doc is not None
        return Inspection(
            passage, doc, ordinal, sha, method,
            before=self.store.document_passages(doc.id, ordinal - context, ordinal - 1) if context else [],
            after=self.store.document_passages(doc.id, ordinal + 1, ordinal + context) if context else [],
            verification=self.verify_passage(passage_id) if verify else None)

    # ------------------------------------------------------------------ verification
    def verify_passage(self, passage_id: str) -> Verification:
        """Deterministic citation check (blueprint 13.1): the stored text is intact, the stored original is still the
        file that was ingested, the cited page exists, and the passage's text is on that page when the original is
        re-read."""
        checks: dict[str, bool | None] = {k: None for k in (
            "passage_exists", "stored_text_intact", "source_file_present", "source_file_unchanged", "page_exists",
            "text_found_on_page")}
        got = self.store.get_passage(passage_id)
        checks["passage_exists"] = got is not None
        if got is None:
            return Verification(False, checks, f"no passage '{passage_id}' in this corpus")
        passage, _, sha, _ = got
        doc = self.store.get_document(passage.document_id)
        checks["stored_text_intact"] = sha256_text(passage.text) == sha
        stored = self.root / doc.stored_path  # type: ignore[union-attr]
        checks["source_file_present"] = stored.is_file()
        if checks["source_file_present"]:
            checks["source_file_unchanged"] = _file_sha256(stored) == doc.file_hash  # type: ignore[union-attr]
        if checks["source_file_unchanged"]:
            try:
                extracted = self._reextract(doc, stored)  # type: ignore[arg-type]
            except ExtractionError as e:
                return Verification(False, checks, f"the stored original can no longer be read: {e.message}")
            page = next((p for p in extracted.pages if p.number == passage.page_number), None)
            checks["page_exists"] = page is not None
            if page is not None:
                checks["text_found_on_page"] = squash(passage.text) in squash(page.raw_text)
        problems = {
            "stored_text_intact": "the stored passage text no longer matches its checksum (database altered)",
            "source_file_present": "the stored original file is missing",
            "source_file_unchanged": "the stored original file has been modified since it was ingested",
            "page_exists": f"page {passage.page_number} no longer exists in the original",
            "text_found_on_page": f"the passage text was not found on page {passage.page_number} of the original",
        }
        failed = [msg for key, msg in problems.items() if checks[key] is False]
        unknown = [k for k, v in checks.items() if v is None]
        if failed:
            return Verification(False, checks, "; ".join(failed))
        if unknown:
            return Verification(False, checks, "could not complete checks: " + ", ".join(unknown))
        return Verification(True, checks, "passage text found on the cited page of the unmodified original")

    def _reextract(self, doc: DocumentRecord, stored: Path) -> ExtractedDocument:
        cached = self._reextracted.get(doc.file_hash)
        if cached is not None:
            self._reextracted.move_to_end(doc.file_hash)
            return cached
        extracted = extract_file(stored, max_bytes=self.max_bytes, max_pages=self.max_pages, isolate=self.isolate,
                                 timeout_s=self.timeout_s)
        self._reextracted[doc.file_hash] = extracted
        while len(self._reextracted) > 2:
            self._reextracted.popitem(last=False)
        return extracted

    # ------------------------------------------------------------------ maintenance
    def remove_document(self, doc_id: str) -> bool:
        """Permanently delete a document, its passages and the stored copy of the original. Past runs that cited it will
        fail verification - which is the point."""
        rec = self.store.delete_document(doc_id)
        if rec is None:
            return False
        self._discard(self.root / rec.stored_path)
        self._reextracted.pop(rec.file_hash, None)
        self.rebuild_index()
        return True

    # ------------------------------------------------------------------ comparison
    def _profile(self, doc_id: str) -> tuple[Counter, dict[str, str]]:
        tf: Counter = Counter()
        forms: dict[str, Counter] = {}
        for p in self.store.document_passages(doc_id):
            for tok in raw_tokens(p.text):
                if tok in STOPWORDS or len(tok) < 3 or tok.isdigit():
                    continue
                s = stem(tok)
                tf[s] += 1
                forms.setdefault(s, Counter())[tok] += 1
        return tf, {s: c.most_common(1)[0][0] for s, c in forms.items()}

    def compare(self, doc_a: str, doc_b: str, query: str | None = None, top_terms: int = 12, top_passages: int = 3) -> dict:
        """Deterministic side-by-side of two documents (the data behind a 'paper comparison' view): metadata, vocabulary
        overlap, the terms that characterise each, and - given a query - each document's best passages for it."""
        a, b = self.store.get_document(doc_a), self.store.get_document(doc_b)
        if a is None or b is None:
            raise KeyError(f"unknown document: {doc_a if a is None else doc_b}")
        (tf_a, form_a), (tf_b, form_b) = self._profile(a.id), self._profile(b.id)
        form = {**form_b, **form_a}

        def weight(term: str) -> float:
            return self.index.idf(term)

        shared = sorted(set(tf_a) & set(tf_b), key=lambda t: (-min(tf_a[t], tf_b[t]) * weight(t), t))
        only_a = sorted(set(tf_a) - set(tf_b), key=lambda t: (-tf_a[t] * weight(t), t))
        only_b = sorted(set(tf_b) - set(tf_a), key=lambda t: (-tf_b[t] * weight(t), t))
        va = {t: c * weight(t) for t, c in tf_a.items()}
        vb = {t: c * weight(t) for t, c in tf_b.items()}
        dot = sum(va[t] * vb[t] for t in set(va) & set(vb))
        norm = math.sqrt(sum(x * x for x in va.values())) * math.sqrt(sum(x * x for x in vb.values()))
        union = set(tf_a) | set(tf_b)
        out = {
            "a": _brief(a), "b": _brief(b),
            "similarity": {"tfidf_cosine": round(dot / norm, 4) if norm else 0.0,
                           "vocabulary_jaccard": round(len(set(tf_a) & set(tf_b)) / len(union), 4) if union else 0.0},
            "shared_terms": [form[t] for t in shared[:top_terms]],
            "only_in_a": [form_a[t] for t in only_a[:top_terms]],
            "only_in_b": [form_b[t] for t in only_b[:top_terms]],
            "note": "lexical comparison only: overlap of words, not of meaning or claims",
        }
        if query:
            out["best_passages"] = {
                d.id: [{"passage_id": h.passage.passage_id, "page": h.passage.page_number, "score": round(h.score, 3),
                        "text": h.passage.text[:300]} for h in self.index.search(query, k=top_passages, document_ids={d.id})]
                for d in (a, b)}
        return out


def _brief(d: DocumentRecord) -> dict:
    return {"id": d.id, "title": d.title, "authors": d.authors, "year": d.year, "year_source": d.year_source,
            "pages": d.page_count, "quality": d.quality["status"], "license": d.license, "source": d.source}


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ranges(pages: list[int | None]) -> str:
    nums = sorted(p for p in pages if p is not None)
    if not nums:
        return "(all)"
    out, start, prev = [], nums[0], nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = n
    out.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ", ".join(out)
