import time

import torch

from src.transformer_blocks import KVCache, TransformerBlock, precompute_rope_freqs


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)


def benchmark(fn, repeats: int = 30) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) * 1000 / repeats


def main() -> None:
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 64
    n_heads = 4
    n_kv_heads = 2
    head_dim = 16
    hidden_dim = 128
    batch_size = 1
    seq_len = 32

    block = TransformerBlock(dim, n_heads, n_kv_heads, head_dim, hidden_dim).to(device)
    block.eval()

    x = torch.randn(batch_size, seq_len, dim, device=device)
    cos, sin = precompute_rope_freqs(head_dim, seq_len, device=device)
    mask = causal_mask(seq_len, device)

    with torch.no_grad():
        full_out = block(x, cos, sin, mask=mask)

        cache = KVCache()
        cached_pieces = []
        for pos in range(seq_len):
            token = x[:, pos : pos + 1, :]
            cached_pieces.append(block(token, cos[pos : pos + 1], sin[pos : pos + 1], kv_cache=cache))
        cached_out = torch.cat(cached_pieces, dim=1)

    max_diff = (full_out - cached_out).abs().max().item()
    full_latency = benchmark(lambda: block(x, cos, sin, mask=mask))

    def cached_forward() -> None:
        cache = KVCache()
        for pos in range(seq_len):
            block(x[:, pos : pos + 1, :], cos[pos : pos + 1], sin[pos : pos + 1], kv_cache=cache)

    cached_latency = benchmark(cached_forward)

    print("Modern Transformer Components Demo")
    print(f"Device: {device}")
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(full_out.shape)}")
    print(f"KV cache K shape after decoding: {tuple(cache.k.shape)}")
    print(f"KV cache V shape after decoding: {tuple(cache.v.shape)}")
    print(f"Max cached vs non-cached difference: {max_diff:.8f}")
    print(f"Full-sequence latency: {full_latency:.3f} ms")
    print(f"Step-by-step cached latency: {cached_latency:.3f} ms")


if __name__ == "__main__":
    main()
