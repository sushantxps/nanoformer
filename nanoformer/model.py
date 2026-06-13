"""A modern decoder-only transformer, written from scratch.

Architecture follows the LLaMA family of models:
  * RMSNorm (pre-normalization) instead of LayerNorm
  * Rotary positional embeddings (RoPE) instead of learned positions
  * Grouped-query attention (GQA) for a smaller KV cache
  * SwiGLU feed-forward network
  * Weight tying between the token embedding and the output head

The forward pass supports an explicit KV cache so that autoregressive
decoding is O(1) per token instead of O(n).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 4          # < n_heads enables grouped-query attention
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    ffn_multiple_of: int = 64    # round SwiGLU hidden dim to a multiple of this
    dropout: float = 0.0
    tie_weights: bool = True

    def __post_init__(self) -> None:
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, (
            "n_heads must be divisible by n_kv_heads"
        )

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root-mean-square normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for numerical stability, cast back at the end.
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (norm * self.weight.float()).to(dtype)


def precompute_rope(
    head_dim: int, max_seq_len: int, theta: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin tables of shape (max_seq_len, head_dim // 2)."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
    )
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim // 2)
    return freqs.cos(), freqs.sin()


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0
) -> torch.Tensor:
    """Rotate query/key vectors by their absolute position.

    Args:
        x: (batch, n_heads, seq_len, head_dim)
        cos, sin: precomputed tables, (max_seq_len, head_dim // 2)
        offset: absolute position of the first token in ``x`` (used when
            decoding with a KV cache, where seq_len == 1 but the token sits
            at position ``offset``).
    """
    seq_len = x.shape[2]
    cos = cos[offset : offset + seq_len].view(1, 1, seq_len, -1)
    sin = sin[offset : offset + seq_len].view(1, 1, seq_len, -1)
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return rotated.to(x.dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match the number of query heads (GQA)."""
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return (
        x[:, :, None, :, :]
        .expand(b, n_kv, n_rep, s, d)
        .reshape(b, n_kv * n_rep, s, d)
    )


class Attention(nn.Module):
    """Multi-head attention with grouped-query KV heads and a KV cache."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.head_dim

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)
        self.dropout = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        b, s, _ = x.shape
        offset = past_kv[0].shape[2] if past_kv is not None else 0

        q = self.wq(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin, offset)
        k = apply_rope(k, cos, sin, offset)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v)

        k_rep = repeat_kv(k, self.n_rep)
        v_rep = repeat_kv(v, self.n_rep)

        if offset == 0:
            # Prefill: plain causal attention.
            out = F.scaled_dot_product_attention(
                q, k_rep, v_rep,
                is_causal=s > 1,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            # Incremental decode: queries attend to the whole cache. When
            # s > 1 (e.g. verifying a speculative draft) we still need a
            # causal mask over the new block, offset by the cache length.
            if s == 1:
                attn_mask = None
            else:
                total = offset + s
                attn_mask = torch.ones(
                    s, total, dtype=torch.bool, device=x.device
                ).tril(diagonal=offset)
            out = F.scaled_dot_product_attention(
                q, k_rep, v_rep, attn_mask=attn_mask
            )

        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        return self.wo(out), new_kv


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (Shazeer, 2020)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = int(2 * (4 * cfg.dim) / 3)
        hidden = cfg.ffn_multiple_of * (
            (hidden + cfg.ffn_multiple_of - 1) // cfg.ffn_multiple_of
        )
        self.w_gate = nn.Linear(cfg.dim, hidden, bias=False)
        self.w_up = nn.Linear(cfg.dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg.dim, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.ffn = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv = self.attn(self.attn_norm(x), cos, sin, past_kv)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_kv


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

KVCache = list[tuple[torch.Tensor, torch.Tensor]]


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scaled init for residual projections (GPT-2 trick): keeps the
        # variance of the residual stream stable as depth grows.
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings and not self.cfg.tie_weights:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_kv: Optional[KVCache] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[KVCache]]:
        """Run the model.

        Args:
            tokens: (batch, seq_len) token ids.
            targets: optional (batch, seq_len) ids for next-token loss.
            past_kv: optional per-layer KV cache from previous steps.

        Returns:
            (logits, loss, new_kv). ``new_kv`` is only built when ``past_kv``
            is provided (inference); during training it is ``None`` to avoid
            holding activations.
        """
        x = self.tok_emb(tokens)
        use_cache = past_kv is not None
        new_cache: Optional[KVCache] = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            layer_past = past_kv[i] if use_cache and len(past_kv) > i else None
            x, kv = block(x, self.rope_cos, self.rope_sin, layer_past)
            if use_cache:
                new_cache.append(kv)

        x = self.norm(x)

        loss = None
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        else:
            logits = self.lm_head(x)

        return logits, loss, new_cache

    @staticmethod
    def crop_cache(cache: KVCache, length: int) -> KVCache:
        """Trim a KV cache to ``length`` positions (used to roll back
        rejected speculative tokens)."""
        return [(k[:, :, :length, :], v[:, :, :length, :]) for k, v in cache]
