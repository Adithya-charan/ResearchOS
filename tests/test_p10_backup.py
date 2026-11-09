"""P10: the documented PostgreSQL backup/restore procedure works on a throwaway database (scripts/backup_restore_drill.py)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests import pg_support

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.postgres
@pytest.mark.skipif(not pg_support.available(), reason="needs a PostgreSQL server")
def test_backup_and_restore_drill_passes_on_a_scratch_database():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "backup_restore_drill.py")], capture_output=True, text=True, timeout=300,
                       env={**__import__("os").environ, "RESEARCHOS_TEST_POSTGRES_URL": pg_support.base_url()})
    assert r.returncode == 0, r.stdout + r.stderr[-1500:]
    result = json.loads(r.stdout[r.stdout.index("{"):])
    assert result["all_passed"] and result["row_counts_equal"] and result["search_after_restore"] and result["login_after_restore"]
