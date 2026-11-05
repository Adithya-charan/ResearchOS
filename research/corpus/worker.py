"""Child process that parses one untrusted file and prints the result as one line of ASCII JSON.

    python -m research.corpus.worker <path> <file_type> <max_pages>

Handled failures are reported as {"error": {...}} with exit code 0; anything else (a crash, a hang) is detected by the parent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from research.corpus.extract import extract_in_process
from research.corpus.models import ExtractionError


def main(argv: list[str]) -> int:
    path, file_type, max_pages = Path(argv[0]), argv[1], int(argv[2])
    try:
        payload = {"document": extract_in_process(path, file_type, max_pages).to_dict()}
    except ExtractionError as e:
        payload = {"error": {"code": e.code, "message": e.message}}
    except Exception as e:  # a parser bug: report it as a corrupt file rather than dying with a traceback
        payload = {"error": {"code": "corrupt", "message": f"{type(e).__name__}: {str(e)[:200]}"}}
    sys.stdout.buffer.write(json.dumps(payload, ensure_ascii=True).encode("ascii"))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
