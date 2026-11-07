import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.architecture import ModelConfig  # noqa: E402


def tiny_config(**overrides) -> ModelConfig:
    base = dict(vocab_size=64, max_seq_len=32, d_model=32, n_layers=2, n_heads=4, d_ff=64, dropout=0.0)
    base.update(overrides)
    return ModelConfig(**base)


# Architecture variants every structural test runs against.
VARIANTS = {
    "baseline": {},
    "untied": {"tie_embeddings": False},
    "rope_rms_swiglu": {"positional_encoding": "rope", "normalization": "rmsnorm", "mlp": "swiglu", "bias": False},
    "sdpa": {"attention_impl": "sdpa"},
}


@pytest.fixture(params=list(VARIANTS), ids=list(VARIANTS))
def variant_config(request) -> ModelConfig:
    return tiny_config(**VARIANTS[request.param])


@pytest.fixture(autouse=True)
def _cpu_threads():
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    yield


# --------------------------------------------------------------------------------------------------------------------------------
# Run the existing database-touching tests on PostgreSQL:   RESEARCHOS_TEST_ON_POSTGRES=1 python -m pytest tests/test_api.py ...
# Every `Store(...)` and `make_engine(...)` the tests create is redirected to a schema on a real PostgreSQL server (see
# tests/pg_support.py). ":memory:" gets a fresh schema each time; a file path gets one schema per path, so tests that "restart" by
# opening the same path twice still see their data. Tests that inspect SQLite files directly are SQLite-specific and are not run this way.
import hashlib  # noqa: E402
import os  # noqa: E402

if os.environ.get("RESEARCHOS_TEST_ON_POSTGRES"):
    from agent_engine.state import dburl  # noqa: E402
    from agent_engine.state import store as _store_mod  # noqa: E402
    from tests import pg_support  # noqa: E402

    _schemas: dict[str, str] = {}

    def _pg_target(location: str, prefix: str) -> str:
        if dburl.is_postgres(location):
            return location
        if ":memory:" in location:
            return pg_support.new_schema_url(prefix=prefix)
        key = f"{prefix}:{location}"
        if key not in _schemas:
            digest = hashlib.sha1(key.encode()).hexdigest()[:12]
            _schemas[key] = dburl.with_schema(pg_support.base_url(), f"{prefix}_{digest}")
        return _schemas[key]

    _orig_store_init = _store_mod.Store.__init__

    def _store_init(self, path=":memory:"):
        _orig_store_init(self, _pg_target(str(path), "e"))

    _store_mod.Store.__init__ = _store_init

    import api.app as _api_app  # noqa: E402
    import api.db as _api_db  # noqa: E402

    _orig_make_engine = _api_db.make_engine

    def _make_engine(url, *a, **kw):
        return _orig_make_engine(_pg_target(url, "a"), *a, **kw)

    _api_db.make_engine = _make_engine
    _api_app.make_engine = _make_engine
