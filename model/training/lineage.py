"""Provenance: prove every checkpoint descends from a random initialisation, never from pretrained weights."""
from __future__ import annotations

import copy

import torch.nn as nn

from model.architecture.transformer import weights_fingerprint


class LineageError(ValueError):
    pass


def new_lineage(model: nn.Module, init_seed: int) -> dict:
    """Call right after building the model, BEFORE any training step."""
    return {
        "root": "random_init",
        "init_seed": init_seed,
        "init_weights_sha256": weights_fingerprint(model),
        "parent_checkpoint_sha256": None,
        "ancestry": [],
    }


def child_lineage(parent_meta: dict) -> dict:
    """Lineage for a run that continues training from `parent_meta`'s checkpoint."""
    assert_random_root(parent_meta)
    lin = copy.deepcopy(parent_meta["lineage"])
    lin["parent_checkpoint_sha256"] = parent_meta["model_sha256"]
    lin["ancestry"] = list(lin.get("ancestry", [])) + [parent_meta["model_sha256"]]
    return lin


def assert_random_root(meta: dict) -> None:
    lin = meta.get("lineage")
    if not lin or lin.get("root") != "random_init" or not lin.get("init_weights_sha256"):
        raise LineageError("checkpoint has no verified random-initialisation lineage")
