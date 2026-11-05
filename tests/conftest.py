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
