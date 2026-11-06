"""Start the RESEARCHOS API:  python scripts/serve_api.py [--host 127.0.0.1] [--port 8000]
Configuration comes from RESEARCHOS_* environment variables (see api/config.py); defaults use local SQLite files.

Behind a reverse proxy, set RESEARCHOS_TRUSTED_PROXIES to the proxy's address(es) (comma-separated) so the client IP used for
rate limiting and audit comes from X-Forwarded-For. It defaults to 127.0.0.1 only: forwarded headers from anyone else are ignored."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def server_options(host: str, port: int, env: dict[str, str] | None = None) -> dict:
    env = os.environ if env is None else env
    return {"factory": True, "host": host, "port": port, "log_level": env.get("RESEARCHOS_LOG_LEVEL", "info"),
            "proxy_headers": True, "forwarded_allow_ips": env.get("RESEARCHOS_TRUSTED_PROXIES", "127.0.0.1"),
            "timeout_graceful_shutdown": 20}  # lets training children checkpoint and exit before the process is killed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    import uvicorn
    uvicorn.run("api.app:create_app", **server_options(args.host, args.port))
    return 0


if __name__ == "__main__":
    sys.exit(main())
