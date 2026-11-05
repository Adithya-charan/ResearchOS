"""Load a run configuration (model + training + data) from YAML."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from model.architecture.config import ModelConfig
from model.common import repo_root
from model.training.trainer import TrainConfig


@dataclass
class DataConfig:
    processed_dir: str = "datasets/processed/synthetic_v1"
    tokenizer_dir: str = "artifacts/tokenizer"
    checkpoint_root: str = "artifacts/checkpoints"


@dataclass
class RunConfig:
    model: ModelConfig
    train: TrainConfig
    data: DataConfig
    raw: dict

    def resolve(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else repo_root() / p


def load_run_config(path: str | Path, vocab_size_from_tokenizer: int | None = None) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    unknown = set(raw) - {"model", "training", "data", "description"}
    if unknown:
        raise ValueError(f"unknown top-level config sections: {sorted(unknown)}")
    model_dict = dict(raw["model"])
    if model_dict.get("vocab_size") in (None, "auto"):
        if vocab_size_from_tokenizer is None:
            raise ValueError("model.vocab_size is 'auto' but no tokenizer vocabulary size was supplied")
        model_dict["vocab_size"] = vocab_size_from_tokenizer
    data_dict = dict(raw.get("data", {}))
    unknown_data = set(data_dict) - set(DataConfig.__dataclass_fields__)
    if unknown_data:
        raise ValueError(f"unknown data config keys: {sorted(unknown_data)}")
    return RunConfig(
        model=ModelConfig.from_dict(model_dict),
        train=TrainConfig.from_dict(raw["training"]),
        data=DataConfig(**data_dict),
        raw=raw,
    )
