"""Model hyper-parameters. The baseline matches the blueprint's starter configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

NORMS = ("layer_norm", "rmsnorm")
POSITIONS = ("learned", "rope")
MLPS = ("gelu", "swiglu")
ATTENTION_IMPLS = ("manual", "sdpa")


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    max_seq_len: int = 256
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    normalization: str = "layer_norm"
    positional_encoding: str = "learned"
    mlp: str = "gelu"
    tie_embeddings: bool = True
    bias: bool = True
    attention_impl: str = "manual"  # "manual" = explicit softmax(QK^T)V; "sdpa" = torch fused kernel
    rope_theta: float = 10000.0
    init_std: float = 0.02

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.positional_encoding == "rope" and (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("RoPE needs an even head dimension")
        for name, value, allowed in (
            ("normalization", self.normalization, NORMS),
            ("positional_encoding", self.positional_encoding, POSITIONS),
            ("mlp", self.mlp, MLPS),
            ("attention_impl", self.attention_impl, ATTENTION_IMPLS),
        ):
            if value not in allowed:
                raise ValueError(f"{name}={value!r} not in {allowed}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        for name in ("vocab_size", "max_seq_len", "d_model", "n_layers", "n_heads", "d_ff"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown model config keys: {sorted(unknown)}")
        return cls(**data)

    def estimate_params(self) -> int:
        """Exact parameter count, computed by building the model on the meta device (no memory)."""
        import torch

        from model.architecture.transformer import Transformer

        with torch.device("meta"):
            m = Transformer(self)
        return sum(p.numel() for p in m.parameters())
