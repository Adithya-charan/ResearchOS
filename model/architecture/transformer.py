"""Decoder-only causal Transformer, written from scratch on top of torch tensor ops.

The attention math (scaled dot-product, causal mask, softmax, head split/merge),
the blocks, the embeddings, the loss and the KV-cache logic are all implemented
here. Only primitive ops (Linear, Embedding, LayerNorm, softmax, matmul) come
from PyTorch.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model.architecture.config import ModelConfig

# One (key, value) pair per layer, each shaped [batch, heads, seq, head_dim].
KVCache = tuple[tuple[torch.Tensor, torch.Tensor], ...]
IGNORE_INDEX = -100


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: KVCache | None = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_norm(cfg: ModelConfig) -> nn.Module:
    return nn.LayerNorm(cfg.d_model) if cfg.normalization == "layer_norm" else RMSNorm(cfg.d_model)


def build_rope_tables(head_dim: int, max_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)  # [L, head_dim/2]
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs (x[..., :h], x[..., h:]) by position-dependent angles. x: [B, H, T, D]."""
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos.to(x.dtype)[None, None]
    sin = sin.to(x.dtype)[None, None]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.impl = cfg.attention_impl
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.resid_drop = nn.Dropout(cfg.dropout)
        # Lower-triangular mask, True = may attend. Not saved in checkpoints.
        mask = torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool).tril()
        self.register_buffer("causal", mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        past: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if rope is not None:
            q = apply_rope(q, *rope)
            k = apply_rope(k, *rope)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        present = (k, v) if use_cache else None
        S = k.size(2)  # total keys = cached + new; queries are the last T positions

        if self.impl == "sdpa":
            p = self.dropout if self.training else 0.0
            if T == S:
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)
            elif T == 1:
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
            else:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=self.causal[S - T : S, :S], dropout_p=p)
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, T, S]
            if T > 1:  # a single query sees every cached key, so no mask is needed
                scores = scores.masked_fill(~self.causal[S - T : S, :S], float("-inf"))
            weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
            weights = F.dropout(weights, p=self.dropout, training=self.training)
            y = weights @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y)), present


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.kind = cfg.mlp
        if cfg.mlp == "swiglu":
            self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
            self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        else:
            self.fc = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "swiglu":
            h = F.silu(self.gate(x)) * self.up(x)
        else:
            h = F.gelu(self.fc(x))
        return self.drop(self.down(h))


class Block(nn.Module):
    """Pre-norm residual block."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = build_norm(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = build_norm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x, rope=None, past=None, use_cache=False):
        a, present = self.attn(self.norm1(x), rope, past, use_cache)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, present


def lm_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Mean next-token cross-entropy over non-ignored positions (padding is excluded)."""
    flat_t = targets.reshape(-1)
    total = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(), flat_t, ignore_index=ignore_index, reduction="sum"
    )
    count = (flat_t != ignore_index).sum().clamp(min=1)
    return total / count


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model) if cfg.positional_encoding == "learned" else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f = build_norm(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        if cfg.positional_encoding == "rope":
            cos, sin = build_rope_tables(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        self.gradient_checkpointing = False
        self.apply(self._init_weights)
        # GPT-2 style: scale residual-branch output projections by 1/sqrt(2 * n_layers).
        scaled_std = cfg.init_std / math.sqrt(2 * cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.down.weight"):
                nn.init.normal_(p, mean=0.0, std=scaled_std)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_key_values: KVCache | None = None,
        use_cache: bool = False,
        logits_to_keep: int | None = None,
        ignore_index: int = IGNORE_INDEX,
    ) -> ModelOutput:
        B, T = input_ids.shape
        start = 0 if past_key_values is None else past_key_values[0][0].size(2)
        if start + T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {start + T} exceeds max_seq_len={self.cfg.max_seq_len}")
        x = self.tok_emb(input_ids)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(start, start + T, device=input_ids.device))
        x = self.drop(x)
        rope = None
        if self.cfg.positional_encoding == "rope":
            rope = (self.rope_cos[start : start + T], self.rope_sin[start : start + T])

        present_all: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[i]
            if self.gradient_checkpointing and self.training and not use_cache:
                x, _ = checkpoint(block, x, rope, None, False, use_reentrant=False)
            else:
                x, present = block(x, rope, past, use_cache)
                if use_cache:
                    present_all.append(present)
        x = self.norm_f(x)
        if logits_to_keep is not None:
            if targets is not None:
                raise ValueError("logits_to_keep cannot be combined with targets")
            x = x[:, -logits_to_keep:]
        logits = self.lm_head(x)
        loss = lm_loss(logits, targets, ignore_index) if targets is not None else None
        return ModelOutput(logits, loss, tuple(present_all) if use_cache else None)

    def num_parameters(self, include_embeddings: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())  # tied weights are counted once
        if not include_embeddings:
            n -= self.tok_emb.weight.numel()
            if self.pos_emb is not None:
                n -= self.pos_emb.weight.numel()
        return n


def build_model(cfg: ModelConfig, seed: int) -> Transformer:
    """Create a model with weights drawn from random init under `seed` (global RNG untouched)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return Transformer(cfg)


def weights_fingerprint(model: nn.Module) -> str:
    """SHA-256 over all parameter values (sorted by name). Used to prove lineage from random init."""
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
