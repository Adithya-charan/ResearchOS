"""File type and size validation before anything parses a file (blueprint 11.2: "file type and size validation, safe
parsing"). The extension must be on the allow-list AND the content must look like that type."""
from __future__ import annotations

import zipfile
from pathlib import Path

from research.corpus.models import ExtractionError

EXT_TYPES = {".pdf": "pdf", ".docx": "docx", ".txt": "txt", ".md": "md", ".markdown": "md", ".html": "html", ".htm": "html"}
DEFAULT_MAX_BYTES = 50 * 1024 * 1024
MAX_UNZIPPED = 200 * 1024 * 1024  # a .docx may not expand beyond this
MAX_ZIP_RATIO = 200  # ...nor compress better than this (zip bombs)
MAX_ZIP_MEMBERS = 5000


def detect_file_type(path: Path, max_bytes: int = DEFAULT_MAX_BYTES) -> str:
    path = Path(path)
    if not path.is_file():
        raise ExtractionError("corrupt", f"not a readable file: {path}")
    size = path.stat().st_size
    if size == 0:
        raise ExtractionError("empty", "file is empty")
    if size > max_bytes:
        raise ExtractionError("too_large", f"file is {size / 1e6:.1f} MB; the limit is {max_bytes / 1e6:.0f} MB")
    ftype = EXT_TYPES.get(path.suffix.lower())
    if ftype is None:
        raise ExtractionError("unsupported_type", f"'{path.suffix or path.name}' is not supported "
                                                  f"(allowed: {', '.join(sorted(EXT_TYPES))})")
    with path.open("rb") as f:
        head = f.read(8192)
    if ftype == "pdf":
        if b"%PDF-" not in head[:1024]:
            raise ExtractionError("type_mismatch", "file has a .pdf extension but is not a PDF")
    elif ftype == "docx":
        _check_docx_container(path)
    else:
        if b"\x00" in head:
            raise ExtractionError("type_mismatch", f"file has a .{path.suffix.lstrip('.')} extension but contains binary data")
        if head.startswith(b"%PDF-") or head.startswith(b"PK\x03\x04"):
            raise ExtractionError("type_mismatch", f"file has a .{path.suffix.lstrip('.')} extension but is really a PDF/zip")
    return ftype


def _check_docx_container(path: Path) -> None:
    if not zipfile.is_zipfile(path):
        raise ExtractionError("type_mismatch", "file has a .docx extension but is not a zip container")
    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            if "word/document.xml" not in {i.filename for i in infos}:
                raise ExtractionError("type_mismatch", "zip container has no word/document.xml: not a Word document")
            if len(infos) > MAX_ZIP_MEMBERS:
                raise ExtractionError("unsafe", f"container has {len(infos)} members (limit {MAX_ZIP_MEMBERS})")
            total = sum(i.file_size for i in infos)
            packed = max(1, sum(i.compress_size for i in infos))
            if total > MAX_UNZIPPED or total / packed > MAX_ZIP_RATIO:
                raise ExtractionError("unsafe", f"container expands to {total / 1e6:.0f} MB (ratio {total / packed:.0f}x): "
                                                "refusing as a possible zip bomb")
    except zipfile.BadZipFile as e:
        raise ExtractionError("corrupt", f"damaged zip container: {e}") from e
