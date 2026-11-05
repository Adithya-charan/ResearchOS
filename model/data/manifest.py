"""Corpus manifest: every training document must declare its origin, licence and allowed use."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from model.common import atomic_write_text, canonical_json, sha256_bytes, sha256_file

SPLIT_SALT = "researchos-split-v1"
REQUIRED_USE = "training"


@dataclass
class DocRecord:
    doc_id: str
    path: str  # relative to the manifest's data root
    sha256: str
    source: str  # URL or local origin
    license: str
    acquired: str  # ISO date
    allowed_use: list[str] = field(default_factory=lambda: [REQUIRED_USE])
    bytes: int = 0
    notes: str = ""


def load_manifest(path: str | Path) -> list[DocRecord]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(DocRecord(**json.loads(line)))
    return records


def write_manifest(path: str | Path, records: list[DocRecord]) -> None:
    text = "".join(json.dumps(asdict(r), sort_keys=True, ensure_ascii=False) + "\n" for r in records)
    atomic_write_text(path, text)


def manifest_sha256(records: list[DocRecord]) -> str:
    rows = sorted((asdict(r) for r in records), key=lambda r: r["doc_id"])
    return sha256_bytes(canonical_json(rows).encode("utf-8"))


def verify_manifest(records: list[DocRecord], root: str | Path) -> list[str]:
    """Return a list of problems (empty = manifest is sound). Checks metadata and file hashes."""
    root = Path(root)
    problems: list[str] = []
    seen: set[str] = set()
    for r in records:
        if r.doc_id in seen:
            problems.append(f"{r.doc_id}: duplicate doc_id")
        seen.add(r.doc_id)
        for name in ("source", "license", "acquired"):
            if not getattr(r, name).strip():
                problems.append(f"{r.doc_id}: missing {name}")
        f = root / r.path
        if not f.is_file():
            problems.append(f"{r.doc_id}: file not found: {r.path}")
        elif sha256_file(f) != r.sha256:
            problems.append(f"{r.doc_id}: sha256 mismatch for {r.path}")
    return problems


def split_of(doc_id: str, val_frac: float, test_frac: float, salt: str = SPLIT_SALT) -> str:
    """Deterministic document-level split. A document's split depends only on its id,
    so chunks of one document can never land on both sides of the train/val boundary."""
    h = hashlib.sha256(f"{salt}:{doc_id}".encode("utf-8")).digest()
    u = int.from_bytes(h[:8], "big") / 2**64
    if u < test_frac:
        return "test"
    if u < test_frac + val_frac:
        return "val"
    return "train"
