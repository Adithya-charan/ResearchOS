"""Start the RESEARCHOS API locally:  python scripts/serve_api.py [--host 127.0.0.1] [--port 8000]
Configuration comes from RESEARCHOS_* environment variables (see api/config.py); defaults use local SQLite files."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    import uvicorn
    uvicorn.run("api.app:create_app", factory=True, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
