"""Manage and query a local document corpus (PDF, DOCX, TXT, Markdown, HTML).

    python scripts/manage_corpus.py ingest papers/ --license CC-BY-4.0 --source "my reading list"
    python scripts/manage_corpus.py list
    python scripts/manage_corpus.py search "ionic conductivity of sulfide electrolytes" --year-min 2018
    python scripts/manage_corpus.py inspect doc-3f8f7f9deb7d:p001      # the passage in context + a fresh check against the file
    python scripts/manage_corpus.py verify doc-3f8f7f9deb7d:p001
    python scripts/manage_corpus.py compare doc-aaaaaaaaaaaa doc-bbbbbbbbbbbb --query "grain boundaries"
    python scripts/manage_corpus.py remove doc-3f8f7f9deb7d --yes

Everything is local. Scanned (image-only) pages are reported and skipped: OCR is not part of this version. Metadata that was
read from a file is labelled with where it came from; nothing is looked up online.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from research.corpus import Corpus, CorpusHit, DocumentRecord, SearchFilters

DEFAULT_ROOT = "artifacts/corpus"
MARK = {True: "ok", False: "FAILED", None: "-"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def one_line(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def describe_quality(doc: DocumentRecord) -> str:
    q = doc.quality
    if q["status"] == "ok":
        return "ok"
    bits = []
    if q["pages_low"]:
        bits.append(f"{len(q['pages_low'])} unreliable")
    if q["pages_no_text"]:
        bits.append(f"{len(q['pages_no_text'])} without text layer")
    return f"{q['status']} ({', '.join(bits)})" if bits else q["status"]


def year_text(doc: DocumentRecord) -> str:
    return f"{doc.year} [{doc.year_source}]" if doc.year else "unknown"


# ------------------------------------------------------------------ commands
def cmd_ingest(c: Corpus, a) -> int:
    overrides = {k: v for k, v in (("title", a.title), ("authors", [s.strip() for s in a.authors.split(";")] if a.authors else None),
                                   ("year", a.year), ("doi", a.doi)) if v}
    if overrides and (len(a.paths) != 1 or Path(a.paths[0]).is_dir()):
        print("error: --title/--authors/--year/--doi describe one document; pass exactly one file", file=sys.stderr)
        return 2
    allowed = tuple(u.strip() for u in a.allowed_use.split(",") if u.strip())
    try:
        reports = c.ingest_many(a.paths, source=a.source, license=a.license, allowed_use=allowed, metadata=overrides or None,
                                force=a.force)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    counts = {"ingested": 0, "duplicate": 0, "rejected": 0, "failed": 0}
    for r in reports:
        counts[r.status] += 1
        name = Path(r.path).name
        if r.status == "ingested":
            print(f"[ingested]  {name} -> {r.document_id}  {r.message}")
        elif r.status == "duplicate":
            print(f"[duplicate] {name}: {r.message}")
        else:
            print(f"[{r.status}:{r.code}] {name}: {r.message}")
        for w in r.warnings:
            print(f"            ! {w}")
    if "search" not in allowed:
        print("note: allowed use does not include 'search', so these documents are stored but not searchable")
    if not reports:
        print("nothing to ingest (no supported files found: .pdf .docx .txt .md .html)")
    print(f"\n{counts['ingested']} ingested, {counts['duplicate']} duplicate, {counts['rejected']} rejected, "
          f"{counts['failed']} failed.   Corpus now: {c.stats()['documents']} document(s), {c.stats()['passages']} passage(s)")
    return 1 if counts["rejected"] or counts["failed"] else 0


def cmd_list(c: Corpus, a) -> int:
    docs = c.list_documents()
    counts = c.store.passage_counts()
    if a.json:
        print(json.dumps([{**d.to_dict(), "passages": counts.get(d.id, 0)} for d in docs], indent=2))
        return 0
    if not docs:
        print("The corpus is empty. Add documents with: ingest <file-or-folder>")
        return 0
    for d in docs:
        print(f"{d.id}  {one_line(d.title, 60)}")
        print(f"    {d.file_type}, {d.page_count} page(s), {counts.get(d.id, 0)} passage(s) | year {year_text(d)} | "
              f"quality {describe_quality(d)} | license {d.license or 'unspecified'}"
              + ("" if d.searchable else " | NOT SEARCHABLE"))
    return 0


def hit_dict(h: CorpusHit) -> dict:
    return {"passage_id": h.passage.passage_id, "document_id": h.passage.document_id, "title": h.document.title,
            "year": h.document.year, "page": h.passage.page_number, "section": h.passage.section, "score": round(h.score, 4),
            "quality": h.passage.quality, "matched_terms": list(h.matched_terms), "text": h.passage.text}


def cmd_search(c: Corpus, a) -> int:
    filters = SearchFilters(year_min=a.year_min, year_max=a.year_max,
                            licenses=frozenset(a.license) if a.license else None,
                            file_types=frozenset(a.type) if a.type else None,
                            document_ids=frozenset(a.document) if a.document else None,
                            exclude_low_quality=a.exclude_low_quality)
    hits = c.search(a.query, k=a.k, filters=filters)
    if a.json:
        print(json.dumps([hit_dict(h) for h in hits], indent=2))
        return 0
    if not hits:
        print(f"No passage matched. Searched {c.index.num_documents} document(s) / {c.index.num_passages} passage(s) with "
              "lexical BM25 (exact words, stemmed): relevant text phrased differently would not be found.")
        return 0
    for i, h in enumerate(hits, 1):
        p, d = h.passage, h.document
        where = f"p.{p.page_number}" if p.page_number else "text"
        flag = "  [unreliable text extraction]" if p.quality != "ok" else ""
        print(f"{i}. [{h.score:.2f}] {one_line(d.title, 70)} ({d.year or 'year unknown'}), {where}"
              + (f", {one_line(p.section, 40)}" if p.section else "") + flag)
        print(f"   {p.passage_id}")
        print(f"   \"{one_line(p.text, 260)}\"")
        print(f"   matched: {', '.join(h.matched_terms)}")
    print(f"\n{len(hits)} passage(s). Lexical BM25 only; inspect a passage with: inspect <passage_id>")
    return 0


def print_verification(v) -> None:
    for name, ok in v.checks.items():
        print(f"    {MARK[ok]:>6}  {name.replace('_', ' ')}")
    print(f"    => {'VERIFIED' if v.ok else 'NOT VERIFIED'}: {v.message}")


def cmd_inspect(c: Corpus, a) -> int:
    ins = c.inspect(a.passage_id, context=a.context)
    if ins is None:
        print(f"error: no passage '{a.passage_id}' in this corpus", file=sys.stderr)
        return 2
    d, p = ins.document, ins.passage
    if a.json:
        print(json.dumps({"document": d.to_dict(), "passage": hit_dict(CorpusHit(p, 0.0, (), d)), "ordinal": ins.ordinal,
                          "text_sha256": ins.text_sha256, "extraction_method": ins.extraction_method,
                          "before": [x.text for x in ins.before], "after": [x.text for x in ins.after],
                          "verification": {"ok": ins.verification.ok, "checks": ins.verification.checks,
                                           "message": ins.verification.message}}, indent=2))
        return 0
    print(f"Document {d.id}: {d.title}   (title from: {d.title_source})")
    print(f"    authors: {'; '.join(d.authors) or 'unknown'} | year: {year_text(d)} | doi: {d.doi or 'none found'}")
    print(f"    {d.filename} ({d.file_type}, {d.page_count} page(s)) sha256 {d.file_hash[:16]}… | extracted with {d.extraction_method}")
    print(f"    extraction quality: {describe_quality(d)} | source: {d.source or 'unspecified'} | license: {d.license or 'unspecified'}"
          f" | allowed use: {', '.join(d.allowed_use) or 'none'}")
    for w in d.warnings:
        print(f"    ! {w}")
    print(f"\nPassage {p.passage_id}  (page {p.page_number if p.page_number else 'n/a'}, section "
          f"{p.section or 'n/a'}, quality {p.quality})")
    for x in ins.before:
        print(f"  ... {one_line(x.text, 200)}")
    print(f"\n  {p.text}\n")
    for x in ins.after:
        print(f"  ... {one_line(x.text, 200)}")
    print("\nVerification against the stored original:")
    print_verification(ins.verification)
    return 0 if ins.verification.ok else 1


def cmd_verify(c: Corpus, a) -> int:
    ids = list(a.passage_ids)
    for doc_id in a.document or []:
        if c.get_document(doc_id) is None:
            print(f"error: no document '{doc_id}'", file=sys.stderr)
            return 2
        ids += [p.passage_id for p in c.store.document_passages(doc_id)]
    if not ids:
        print("error: give passage ids and/or --document DOC_ID", file=sys.stderr)
        return 2
    failed = 0
    for pid in ids:
        v = c.verify_passage(pid)
        failed += not v.ok
        if not v.ok or len(ids) <= 5:
            print(f"{pid}: {'VERIFIED' if v.ok else 'NOT VERIFIED'} - {v.message}")
    print(f"\n{len(ids) - failed} of {len(ids)} passage(s) verified against their stored originals.")
    return 1 if failed else 0


def cmd_remove(c: Corpus, a) -> int:
    d = c.get_document(a.document_id)
    if d is None:
        print(f"error: no document '{a.document_id}'", file=sys.stderr)
        return 2
    if not a.yes:
        print(f"This would permanently delete '{d.title}' ({d.id}), its passages and the stored copy of {d.filename}.\n"
              "Runs that cited it will no longer verify. Re-run with --yes to confirm.", file=sys.stderr)
        return 2
    c.remove_document(d.id)
    print(f"removed {d.id} '{d.title}'")
    return 0


def cmd_compare(c: Corpus, a) -> int:
    try:
        r = c.compare(a.doc_a, a.doc_b, query=a.query)
    except KeyError as e:
        print(f"error: {e.args[0]}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(r, indent=2))
        return 0
    for side in ("a", "b"):
        d = r[side]
        print(f"{side.upper()}: {d['id']}  {d['title']}")
        print(f"    authors: {'; '.join(d['authors']) or 'unknown'} | year {d['year'] or 'unknown'}"
              + (f" [{d['year_source']}]" if d["year"] else "") + f" | {d['pages']} page(s) | extraction {d['quality']}"
              + f" | license {d['license'] or 'unspecified'}")
    s = r["similarity"]
    print(f"\nvocabulary overlap (Jaccard): {s['vocabulary_jaccard']:.3f}    TF-IDF cosine: {s['tfidf_cosine']:.3f}")
    print(f"shared terms:  {', '.join(r['shared_terms']) or '(none)'}")
    print(f"only in A:     {', '.join(r['only_in_a']) or '(none)'}")
    print(f"only in B:     {', '.join(r['only_in_b']) or '(none)'}")
    for doc_id, passages in r.get("best_passages", {}).items():
        print(f"\nBest passages in {doc_id} for \"{a.query}\":")
        for p in passages or []:
            print(f"  [{p['score']}] p.{p['page']}  {p['passage_id']}  \"{one_line(p['text'], 200)}\"")
        if not passages:
            print("  (no matching passage)")
    print(f"\nnote: {r['note']}")
    return 0


def cmd_stats(c: Corpus, a) -> int:
    s = c.stats()
    if a.json:
        print(json.dumps(s, indent=2))
        return 0
    print(f"{s['documents']} document(s) ({s['searchable_documents']} searchable), {s['passages']} passage(s) "
          f"({s['low_quality_passages']} from pages with unreliable extraction); index holds {s['indexed_documents']} "
          f"document(s) / {s['indexed_passages']} passage(s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT, help=f"corpus folder (default {DEFAULT_ROOT})")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="add files or folders (recursively) to the corpus")
    p.add_argument("paths", nargs="+")
    p.add_argument("--source", help="where the documents came from (recorded with each one)")
    p.add_argument("--license", help="licence / terms of use, e.g. CC-BY-4.0 (recorded and filterable)")
    p.add_argument("--allowed-use", default="search,quote", help="comma list of: search, quote (default both; "
                                                                "documents without 'search' are stored but never indexed)")
    p.add_argument("--title", help="override the title (one file only)")
    p.add_argument("--authors", help="override the authors, separated by ';' (one file only)")
    p.add_argument("--year", type=int, help="override the publication year (one file only)")
    p.add_argument("--doi", help="override the DOI (one file only)")
    p.add_argument("--force", action="store_true", help="re-ingest a file that is already in the corpus")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("list", help="list documents")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("search", help="BM25 search over the corpus, with metadata filters")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=8)
    p.add_argument("--year-min", type=int)
    p.add_argument("--year-max", type=int)
    p.add_argument("--license", action="append", help="only documents with this licence (repeatable)")
    p.add_argument("--type", action="append", choices=["pdf", "docx", "txt", "md", "html"], help="only this file type (repeatable)")
    p.add_argument("--document", action="append", help="only this document id (repeatable)")
    p.add_argument("--exclude-low-quality", action="store_true", help="skip passages from pages with unreliable extraction")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("inspect", help="show a passage in context with its document details and a fresh verification")
    p.add_argument("passage_id")
    p.add_argument("--context", type=int, default=1, help="neighbouring passages to show on each side")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("verify", help="check that passages are really on the cited page of the unmodified original")
    p.add_argument("passage_ids", nargs="*")
    p.add_argument("--document", action="append", help="verify every passage of this document (repeatable)")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("remove", help="permanently delete a document and the stored copy of its file")
    p.add_argument("document_id")
    p.add_argument("--yes", action="store_true", help="confirm the deletion")
    p.set_defaults(fn=cmd_remove)

    p = sub.add_parser("compare", help="side-by-side lexical comparison of two documents")
    p.add_argument("doc_a")
    p.add_argument("doc_b")
    p.add_argument("--query", help="also show each document's best passages for this query")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("stats", help="corpus size and extraction-quality counts")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_stats)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = Path(args.root)
    root = root if root.is_absolute() else repo_root() / root
    if args.command != "ingest" and not (root / "corpus.db").is_file():  # don't silently create an empty corpus on a typo
        print(f"error: no corpus at {root} (create one with: ingest <file-or-folder>)", file=sys.stderr)
        return 2
    with Corpus(root) as corpus:
        return args.fn(corpus, args)


if __name__ == "__main__":
    sys.exit(main())
