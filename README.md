# Build Modern Transformer Components from Scratch

This project implements the core components used in modern Transformer decoder blocks for Large Language Models. The implementation is written from scratch with PyTorch and includes RMSNorm, Rotary Position Embeddings, Grouped-Query Attention, KV Cache, SwiGLU, and a complete decoder block.

## Components

### RMSNorm

`RMSNorm` normalizes each token vector using the root mean square of its hidden dimension. Unlike LayerNorm, it does not subtract the mean. This keeps the operation simpler while still stabilizing activations.

Formula:

```text
x_norm = x / sqrt(mean(x^2) + eps)
output = x_norm * weight
```

### Rotary Position Embeddings

RoPE adds position information by rotating query and key vectors. Instead of adding a learned position vector to token embeddings, RoPE changes the geometry of attention so relative positions are naturally represented in the query-key dot product.

Implemented functions:

- `precompute_rope_freqs`
- `rotate_half`
- `apply_rope`

### Grouped-Query Attention

Grouped-Query Attention uses more query heads than key/value heads. This reduces memory and computation during inference while preserving multi-head query capacity.

Example configuration:

```text
n_heads = 4
n_kv_heads = 2
```

Each key/value head is shared by two query heads.

### KV Cache

`KVCache` stores key and value tensors during autoregressive decoding. During generation, the model only computes keys and values for the new token, then reuses cached keys and values from previous tokens.

Cache tensor shape:

```text
[batch_size, n_kv_heads, seq_len, head_dim]
```

### SwiGLU

`SwiGLU` is the feed-forward network used in many modern LLMs. It uses three linear layers and a gating operation.

Formula:

```text
output = W2(SiLU(W1(x)) * W3(x))
```

### Transformer Decoder Block

The final `TransformerBlock` combines:

- RMSNorm before attention
- Grouped-Query Attention with RoPE
- Optional KV Cache
- Residual connection
- RMSNorm before feed-forward
- SwiGLU feed-forward network
- Residual connection

Forward flow:

```text
x = x + attention(RMSNorm(x))
x = x + swiglu(RMSNorm(x))
```

## Project Structure

```text
src/
  transformer_blocks.py
tests/
  test_transformer.py
run_demo.py
pyy.py
requirements.txt
BENCHMARK_REPORT.md
```

## Setup

Create and activate a virtual environment. With `uv`:

```powershell
uv venv --python 3.12
.\.venv\Scripts\activate
```

Install CUDA PyTorch:

```powershell
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install the remaining dependencies:

```powershell
uv pip install einops numpy pytest tiktoken transformers
```

## Verify GPU

```powershell
python .\pyy.py
```

Expected CUDA output:

```text
PyTorch Version: ...+cu118
CUDA available: NVIDIA ...
```

## Run Tests

```powershell
python -m pytest -q
```

Current result:

```text
4 passed in 3.95s
```

## Run Demo and Benchmark

```powershell
python .\run_demo.py
```

The demo verifies:

- input and output tensor shapes
- KV cache key/value shapes
- cached vs non-cached numerical equivalence
- full-sequence and step-by-step decoding latency

## Notes

The step-by-step cached benchmark can be slower than the full-sequence benchmark in this small demo because each token is decoded through a Python loop. In real text generation, KV cache is useful because it avoids recomputing keys and values for all previous tokens at every new step.
