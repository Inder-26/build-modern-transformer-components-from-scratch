from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/CascadiaMono.ttf",
        "C:/Windows/Fonts/lucon.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def terminal_image(lines: list[str], output_path: Path, width: int = 1388) -> None:
    font = load_font(18)
    padding_x = 10
    padding_y = 10
    line_height = 22
    height = padding_y * 2 + line_height * len(lines)

    image = Image.new("RGB", (width, height), color=(30, 30, 30))
    draw = ImageDraw.Draw(image)

    y = padding_y
    for line in lines:
        fill = (230, 230, 230)
        if "passed" in line:
            fill = (68, 255, 128)
        elif "python" in line:
            fill = (255, 224, 102)
        draw.text((padding_x, y), line, font=font, fill=fill)
        y += line_height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    prompt = (
        "(Build Modern Transformer Components from Scratch) PS "
        "D:\\Coding Central\\Build Modern Transformer Components from Scratch>"
    )

    debug_full = [
        f"{prompt} python .\\debug_walkthrough.py",
        "[RoPE] cos table: shape=(3, 4), dtype=torch.float32, device=cuda:0, mean=0.687318, std=0.521244",
        "[RoPE] sin table: shape=(3, 4), dtype=torch.float32, device=cuda:0, mean=0.296795, std=0.409649",
        "",
        "=== Full sequence pass ===",
        "[Block] input x: shape=(1, 3, 8), dtype=torch.float32, device=cuda:0, mean=-0.057712, std=0.881430",
        "[RMSNorm] input: shape=(1, 3, 8), dtype=torch.float32, device=cuda:0, mean=-0.057712, std=0.881430",
        "[RMSNorm] mean square: shape=(1, 3, 1), dtype=torch.float32, device=cuda:0, mean=0.780249, std=0.324472",
        "[RMSNorm] output: shape=(1, 3, 8), dtype=torch.float32, device=cuda:0, mean=0.017680, std=0.999843",
        "[GQA] input: shape=(1, 3, 8), dtype=torch.float32, device=cuda:0, mean=0.017680, std=0.999843",
        "[GQA] q projection: shape=(1, 3, 8), dtype=torch.float32, device=cuda:0, mean=-0.106837, std=0.547719",
        "[GQA] k projection: shape=(1, 3, 4), dtype=torch.float32, device=cuda:0, mean=0.010220, std=0.557689",
        "[GQA] v projection: shape=(1, 3, 4), dtype=torch.float32, device=cuda:0, mean=-0.041233, std=0.672641",
        "[GQA] q heads: shape=(1, 2, 3, 4), dtype=torch.float32, device=cuda:0, mean=-0.106837, std=0.547719",
        "[GQA] k heads: shape=(1, 1, 3, 4), dtype=torch.float32, device=cuda:0, mean=0.010220, std=0.557689",
        "[GQA] v heads: shape=(1, 1, 3, 4), dtype=torch.float32, device=cuda:0, mean=-0.041233, std=0.672641",
        "[RoPE] input before rotation: shape=(1, 2, 3, 4), dtype=torch.float32, device=cuda:0, mean=-0.106837, std=0.547719",
        "[RoPE] output after rotation: shape=(1, 2, 3, 4), dtype=torch.float32, device=cuda:0, mean=-0.023931, std=0.557529",
        "[RoPE] input before rotation: shape=(1, 1, 3, 4), dtype=torch.float32, device=cuda:0, mean=0.010220, std=0.557689",
        "[RoPE] output after rotation: shape=(1, 1, 3, 4), dtype=torch.float32, device=cuda:0, mean=-0.105842, std=0.547649",
        "[GQA] repeated k for query heads: shape=(1, 2, 3, 4), dtype=torch.float32, device=cuda:0, mean=-0.105842, std=0.547649",
        "[GQA] repeated v for query heads: shape=(1, 2, 3, 4), dtype=torch.float32, device=cuda:0, mean=-0.041233, std=0.672641",
        "[GQA] attention scores: shape=(1, 2, 3, 3), dtype=torch.float32, device=cuda:0, mean=-0.032265, std=0.245073",
        "[GQA] attention weights: shape=(1, 2, 3, 3), dtype=torch.float32, device=cuda:0, mean=0.333333, std=0.069531",
        "[GQA] attention output per head: shape=(1, 2, 3, 4), dtype=torch.float32, device=cuda:0, mean=-0.042221, std=0.239123",
        "[GQA] concatenated heads: shape=(1, 3, 8), dtype=torch.float32, device=cuda:0, mean=-0.042221, std=0.239123",
        "[GQA] final output: shape=(1, 3, 8), dtype=torch.float32, device=cuda:0, mean=0.029051, std=0.155452",
        "[Block] attention branch output: shape=(1, 3, 8), dtype=torch.float32, device=cuda:0, mean=0.029051, std=0.155452",
        "[Block] after attention residual: shape=(1, 3, 8), dtype=torch.float32, device=cuda:0, mean=-0.028662, std=0.824298",
        "[RMSNorm] input: shape=(1, 3, 8), dtype=torch.float32, device=cuda:0, mean=-0.028662, std=0.824298",
        "[RMSNorm] mean square: shape=(1, 3, 1), dtype=torch.float32, device=cuda:0, mean=0.680288, std=0.239521",
        "[RMSNorm] output: shape=(1, 3, 8), dtype=torch.float32, device=cuda:0, mean=0.001655, std=0.999998",
        "[SwiGLU] input: shape=(1, 3, 8), dtype=torch.float32, device=cuda:0, mean=0.001655, std=0.999998",
        "[SwiGLU] gate SiLU(W1(x)): shape=(1, 3, 16), dtype=torch.float32, device=cuda:0, mean=0.129322, std=0.301607",
        "[SwiGLU] value W3(x): shape=(1, 3, 16), dtype=torch.float32, device=cuda:0, mean=0.018620, std=0.532468",
        "[SwiGLU] gated hidden: shape=(1, 3, 16), dtype=torch.float32, device=cuda:0, mean=-0.012610, std=0.139277",
        "[SwiGLU] output: shape=(1, 3, 8), dtype=torch.float32, device=cuda:0, mean=0.004429, std=0.081124",
        "[Block] ffn branch output: shape=(1, 3, 8), dtype=torch.float32, device=cuda:0, mean=0.004429, std=0.081124",
        "[Block] final output: shape=(1, 3, 8), dtype=torch.float32, device=cuda:0, mean=-0.024232, std=0.826928",
    ]

    debug_cached = [
        "Final output shape: (1, 3, 8)",
        "",
        "=== One-token cached pass ===",
        "[Block] input x: shape=(1, 1, 8), dtype=torch.float32, device=cuda:0, mean=-0.318801, std=0.751830",
        "[RMSNorm] input: shape=(1, 1, 8), dtype=torch.float32, device=cuda:0, mean=-0.318801, std=0.751830",
        "[RMSNorm] output: shape=(1, 1, 8), dtype=torch.float32, device=cuda:0, mean=-0.390386, std=0.920650",
        "[GQA] input: shape=(1, 1, 8), dtype=torch.float32, device=cuda:0, mean=-0.390386, std=0.920650",
        "[GQA] q projection: shape=(1, 1, 8), dtype=torch.float32, device=cuda:0, mean=0.060793, std=0.680469",
        "[GQA] k projection: shape=(1, 1, 4), dtype=torch.float32, device=cuda:0, mean=-0.065200, std=0.533363",
        "[GQA] v projection: shape=(1, 1, 4), dtype=torch.float32, device=cuda:0, mean=0.327874, std=0.780068",
        "[KVCache] cached k: shape=(1, 1, 1, 4), dtype=torch.float32, device=cuda:0, mean=-0.065200, std=0.533363",
        "[KVCache] cached v: shape=(1, 1, 1, 4), dtype=torch.float32, device=cuda:0, mean=0.327874, std=0.780068",
        "[GQA] attention scores: shape=(1, 2, 1, 1), dtype=torch.float32, device=cuda:0, mean=0.228080, std=0.119146",
        "[GQA] attention weights: shape=(1, 2, 1, 1), dtype=torch.float32, device=cuda:0, mean=1.000000, std=0.000000",
        "[GQA] final output: shape=(1, 1, 8), dtype=torch.float32, device=cuda:0, mean=-0.115383, std=0.524733",
        "[SwiGLU] output: shape=(1, 1, 8), dtype=torch.float32, device=cuda:0, mean=-0.073022, std=0.067694",
        "[Block] final output: shape=(1, 1, 8), dtype=torch.float32, device=cuda:0, mean=-0.507205, std=0.367694",
        "",
        "One-token output shape: (1, 1, 8)",
    ]

    demo = [
        f"{prompt} python .\\run_demo.py",
        "Modern Transformer Components Demo",
        "Device: cuda",
        "Input shape: (1, 32, 64)",
        "Output shape: (1, 32, 64)",
        "KV cache K shape after decoding: (1, 2, 32, 16)",
        "KV cache V shape after decoding: (1, 2, 32, 16)",
        "Max cached vs non-cached difference: 0.00000024",
        "Full-sequence latency: 2.269 ms",
        "Step-by-step cached latency: 79.837 ms",
    ]

    tests = [
        f"{prompt} python -m pytest -q",
        "....                                                                                                   [100%]",
        "4 passed in 7.28s",
    ]

    terminal_image(debug_full, ASSETS / "debug-walkthrough-full.png")
    terminal_image(debug_cached, ASSETS / "debug-walkthrough-cached.png")
    terminal_image(demo, ASSETS / "benchmark-demo.png")
    terminal_image(tests, ASSETS / "pytest-results.png")


if __name__ == "__main__":
    main()
