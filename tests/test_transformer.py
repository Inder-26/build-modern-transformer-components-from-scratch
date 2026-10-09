
import pytest
import torch
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

for candidate in (str(SRC_DIR), str(PROJECT_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

try:
    from transformer_blocks import (
        RMSNorm,
        precompute_rope_freqs,
        apply_rope,
        SwiGLU,
        TransformerBlock,
        KVCache,
    )
except ModuleNotFoundError:
    from src.transformer_blocks import (
        RMSNorm,
        precompute_rope_freqs,
        apply_rope,
        SwiGLU,
        TransformerBlock,
        KVCache,
    )

# Set random seed for reproducibility
torch.manual_seed(42)


def test_rmsnorm_properties():
    """
    Verifies that RMSNorm scales activations to have an approximate RMS of 1.0.
    """
    dim = 128
    batch_size = 4
    seq_len = 16
    norm = RMSNorm(dim=dim, eps=1e-6)

    # Initialize with random weights
    x = torch.randn(batch_size, seq_len, dim) * 10.0  # Large scale input
    out = norm(x)

    # Calculate RMS of output: RMS = sqrt(mean(x^2))
    rms = torch.sqrt(out.pow(2).mean(dim=-1))

    # Since weight is initialized to 1.0, output RMS should be extremely close to 1.0
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4), (
        "RMSNorm failed to normalize activations to unit RMS."
    )
    assert out.shape == x.shape, "RMSNorm altered the input shape."


def test_rope_relative_distance():
    """
    Verifies that Rotary Position Embeddings preserve relative distance.
    Specifically, the dot product of rotated Query and Key vectors at positions m and n
    must equal the dot product of Query and Key rotated by 0 and (n - m).
    """
    head_dim = 64
    seq_len = 100
    cos, sin = precompute_rope_freqs(head_dim=head_dim, seq_len=seq_len)

    # Create arbitrary Query and Key vectors
    q = torch.randn(1, 1, 1, head_dim)  # Batch=1, Head=1, Seq=1, Dim
    k = torch.randn(1, 1, 1, head_dim)

    # Position m = 15, Position n = 25 (Relative distance = 10)
    m = 15
    n = 25

    # Apply RoPE at absolute positions m and n
    q_m = apply_rope(q, cos[m:m + 1], sin[m:m + 1])
    k_n = apply_rope(k, cos[n:n + 1], sin[n:n + 1])

    # Apply RoPE at relative positions 0 and (n - m)
    q_0 = apply_rope(q, cos[0:1], sin[0:1])
    k_rel = apply_rope(k, cos[n - m:n - m + 1], sin[n - m:n - m + 1])

    # Compute dot products
    dot_product_absolute = torch.sum(q_m * k_n)
    dot_product_relative = torch.sum(q_0 * k_rel)

    assert torch.allclose(dot_product_absolute, dot_product_relative, atol=1e-5), (
        f"RoPE failed relative distance preservation. Abs: {dot_product_absolute.item()}, "
        f"Rel: {dot_product_relative.item()}"
    )


def test_swiglu_shape_and_flow():
    """
    Verifies that SwiGLU correctly transforms shapes and executes without error.
    """
    dim = 64
    hidden_dim = 128
    x = torch.randn(2, 10, dim)
    ffn = SwiGLU(dim=dim, hidden_dim=hidden_dim)
    out = ffn(x)
    assert out.shape == x.shape, (
        f"SwiGLU altered output shape. Expected {x.shape}, got {out.shape}"
    )


def test_kv_cache_equivalence():
    """
    CRITICAL TEST: Verifies that step-by-step decoding with a KV Cache
    produces the exact same numerical outputs as a full-sequence parallel forward pass.
    """
    dim = 64

    n_heads = 4
    n_kv_heads = 2  # Grouped-Query Attention
    head_dim = 16
    hidden_dim = 128
    batch_size = 1
    seq_len = 5

    # Initialize block
    block = TransformerBlock(
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        hidden_dim=hidden_dim,
    )
    block.eval()  # Set to evaluation mode

    # Generate inputs
    x = torch.randn(batch_size, seq_len, dim)

    # Precompute RoPE parameters for the maximum sequence length
    cos, sin = precompute_rope_freqs(head_dim=head_dim, seq_len=seq_len)

    # 1. Parallel Execution (Prefill Phase)
    # Create a causal mask to prevent looking ahead
    mask = torch.full((seq_len, seq_len), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    with torch.no_grad():
        parallel_out = block(x, cos, sin, mask=mask)

    # 2. Autoregressive Execution with KV Cache (Decoding Phase)
    cache = KVCache()
    sequential_outputs = []
    with torch.no_grad():
        for i in range(seq_len):
            # Extract the single token at step i
            token_input = x[:, i:i + 1, :]  # Shape: [batch_size, 1, dim]

            # Slice RoPE parameters for the current step
            # During decoding, we use the cos/sin corresponding to the current absolute position
            cos_step = cos[i:i + 1]
            sin_step = sin[i:i + 1]

            # Forward pass with KV Cache
            step_out = block(token_input, cos_step, sin_step, kv_cache=cache)
            sequential_outputs.append(step_out)

    # Concatenate sequential outputs along the sequence dimension
    sequential_out = torch.cat(sequential_outputs, dim=1)

    # Verify equivalence
    assert torch.allclose(parallel_out, sequential_out, atol=1e-5), (
        "KV Cache output does not match parallel execution output."
    )
    print("\n[SUCCESS] KV Cache Equivalence Verified!")
