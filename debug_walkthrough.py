import os

import torch

from src.transformer_blocks import KVCache, TransformerBlock, precompute_rope_freqs


def main() -> None:
    os.environ["TRANSFORMER_DEBUG"] = "1"
    torch.manual_seed(7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    seq_len = 3
    dim = 8
    n_heads = 2
    n_kv_heads = 1
    head_dim = 4
    hidden_dim = 16

    block = TransformerBlock(dim, n_heads, n_kv_heads, head_dim, hidden_dim).to(device)
    block.eval()

    x = torch.randn(batch_size, seq_len, dim, device=device)
    cos, sin = precompute_rope_freqs(head_dim, seq_len, device=device)

    print("\n=== Full sequence pass ===")
    with torch.no_grad():
        output = block(x, cos, sin)
    print(f"\nFinal output shape: {tuple(output.shape)}")

    print("\n=== One-token cached pass ===")
    cache = KVCache()
    with torch.no_grad():
        token = x[:, 0:1, :]
        token_output = block(token, cos[0:1], sin[0:1], kv_cache=cache)
    print(f"\nOne-token output shape: {tuple(token_output.shape)}")


if __name__ == "__main__":
    main()
