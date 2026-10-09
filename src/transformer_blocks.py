import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def debug_enabled() -> bool:
    return os.getenv("TRANSFORMER_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def debug_print(label: str, tensor: Optional[torch.Tensor] = None) -> None:
    if not debug_enabled():
        return

    if tensor is None:
        print(label)
        return

    with torch.no_grad():
        detached = tensor.detach()
        print(
            f"{label}: shape={tuple(detached.shape)}, "
            f"dtype={detached.dtype}, device={detached.device}, "
            f"mean={detached.float().mean().item():.6f}, "
            f"std={detached.float().std(unbiased=False).item():.6f}"
        )


class RMSNorm(nn.Module):
    """Root Mean Square normalization without mean centering."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        debug_print("[RMSNorm] input", x)
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        debug_print("[RMSNorm] mean square", rms)
        output = x * torch.rsqrt(rms + self.eps) * self.weight
        debug_print("[RMSNorm] output", output)
        return output


def precompute_rope_freqs(
    head_dim: int,
    seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine tables for rotary position embeddings."""
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE.")

    dim_positions = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (dim_positions / head_dim))
    token_positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(token_positions, inv_freq)
    freqs = torch.cat((freqs, freqs), dim=-1)
    cos = freqs.cos()
    sin = freqs.sin()
    debug_print("[RoPE] cos table", cos)
    debug_print("[RoPE] sin table", sin)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dimension: [x1, x2] -> [-x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to a tensor shaped [batch, heads, seq_len, head_dim]."""
    debug_print("[RoPE] input before rotation", x)
    cos = cos.to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(1)
    sin = sin.to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(1)
    output = (x * cos) + (rotate_half(x) * sin)
    debug_print("[RoPE] output after rotation", output)
    return output


class KVCache:
    """Stores key/value tensors across autoregressive decoding steps."""

    def __init__(self):
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    def update(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        debug_print("[KVCache] new k", k)
        debug_print("[KVCache] new v", v)
        if self.k is None:
            self.k = k
            self.v = v
        else:
            self.k = torch.cat((self.k, k), dim=2)
            self.v = torch.cat((self.v, v), dim=2)
        debug_print("[KVCache] cached k", self.k)
        debug_print("[KVCache] cached v", self.v)
        return self.k, self.v

    def clear(self) -> None:
        self.k = None
        self.v = None


class GroupedQueryAttention(nn.Module):
    """Grouped-query attention with optional KV caching."""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads.")

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.num_queries_per_kv = n_heads // n_kv_heads

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        debug_print("[GQA] input", x)
        _, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        debug_print("[GQA] q projection", q)
        debug_print("[GQA] k projection", k)
        debug_print("[GQA] v projection", v)

        q = rearrange(q, "b s (h d) -> b h s d", h=self.n_heads, d=self.head_dim)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.n_kv_heads, d=self.head_dim)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.n_kv_heads, d=self.head_dim)
        debug_print("[GQA] q heads", q)
        debug_print("[GQA] k heads", k)
        debug_print("[GQA] v heads", v)

        q = apply_rope(q, cos[:seq_len], sin[:seq_len])
        k = apply_rope(k, cos[:seq_len], sin[:seq_len])

        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        if self.num_queries_per_kv > 1:
            k = torch.repeat_interleave(k, self.num_queries_per_kv, dim=1)
            v = torch.repeat_interleave(v, self.num_queries_per_kv, dim=1)
            debug_print("[GQA] repeated k for query heads", k)
            debug_print("[GQA] repeated v for query heads", v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        debug_print("[GQA] attention scores", scores)
        if mask is not None:
            scores = scores + mask.to(device=scores.device, dtype=scores.dtype)
            debug_print("[GQA] masked attention scores", scores)

        attn_weights = F.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        debug_print("[GQA] attention weights", attn_weights)
        output = torch.matmul(attn_weights, v)
        debug_print("[GQA] attention output per head", output)
        output = rearrange(output, "b h s d -> b s (h d)")
        debug_print("[GQA] concatenated heads", output)
        output = self.out_proj(output)
        debug_print("[GQA] final output", output)
        return output


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network: SiLU(xW1) * xW3, then projected by W2."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        debug_print("[SwiGLU] input", x)
        gate = F.silu(self.w1(x))
        value = self.w3(x)
        debug_print("[SwiGLU] gate SiLU(W1(x))", gate)
        debug_print("[SwiGLU] value W3(x)", value)
        hidden = gate * value
        debug_print("[SwiGLU] gated hidden", hidden)
        output = self.w2(hidden)
        debug_print("[SwiGLU] output", output)
        return output


class TransformerBlock(nn.Module):
    """Pre-norm decoder block using RMSNorm, GQA, RoPE, KV cache, and SwiGLU."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        hidden_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(dim, eps=eps)
        self.attn = GroupedQueryAttention(dim, n_heads, n_kv_heads, head_dim)
        self.ffn_norm = RMSNorm(dim, eps=eps)
        self.ffn = SwiGLU(dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        debug_print("[Block] input x", x)
        attn_input = self.attn_norm(x)
        attn_output = self.attn(attn_input, cos, sin, kv_cache=kv_cache, mask=mask)
        debug_print("[Block] attention branch output", attn_output)
        x = x + attn_output
        debug_print("[Block] after attention residual", x)

        ffn_input = self.ffn_norm(x)
        ffn_output = self.ffn(ffn_input)
        debug_print("[Block] ffn branch output", ffn_output)
        x = x + ffn_output
        debug_print("[Block] final output", x)
        return x
