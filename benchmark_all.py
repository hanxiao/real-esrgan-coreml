"""Benchmark all 5 Real-ESRGAN model variants: CoreML vs MLX.

Usage: uv run python benchmark_all.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
from PIL import Image

REPO = Path(__file__).parent
MLX_REPO = Path(os.path.expanduser("~/.openclaw/workspace/real-esrgan-mlx"))
WEIGHTS_DIR = REPO / "weights"
BENCH_IMG = REPO / "bench_input.png"
PRE_PAD = 10
WARMUP = 2
RUNS = 5
INPUT_SIZE = 512

MODEL_NAMES = ["x4plus", "x2plus", "anime_6B", "animevideo", "general"]
MODEL_SCALES = {
    "x4plus": 4, "x2plus": 2, "anime_6B": 4, "animevideo": 4, "general": 4,
}


def ensure_bench_image():
    """Create a 512x512 test image if it doesn't exist."""
    if not BENCH_IMG.exists():
        print("Creating 512x512 benchmark input...")
        img = Image.fromarray(
            np.random.RandomState(42).randint(0, 256, (512, 512, 3), dtype=np.uint8), "RGB"
        )
        img.save(BENCH_IMG)


def load_image() -> np.ndarray:
    return np.array(Image.open(BENCH_IMG).convert("RGB"), dtype=np.float32) / 255.0


def pad_reflect_np(img: np.ndarray, pad: int) -> np.ndarray:
    if pad == 0:
        return img
    return np.pad(img, ((0, pad), (0, pad), (0, 0)), mode='reflect')


# -- Convert all models --

def convert_all():
    """Convert all 5 models for INPUT_SIZE input."""
    from convert import MODEL_CONFIGS, convert
    padded = INPUT_SIZE + PRE_PAD
    for name in MODEL_NAMES:
        mlpkg = WEIGHTS_DIR / f"RealESRGAN_{name}_{padded}_fp16.mlpackage"
        if mlpkg.exists():
            print(f"[convert] {name}: already exists ({mlpkg.name})")
            continue
        print(f"\n[convert] {name}...")
        convert(model_name=name, input_size=padded, use_fp16=True)


# -- CoreML benchmark --

def bench_coreml(model_name: str) -> float:
    """Benchmark CoreML for a model. Returns median time in seconds."""
    scale = MODEL_SCALES[model_name]
    padded = INPUT_SIZE + PRE_PAD
    mlpkg = WEIGHTS_DIR / f"RealESRGAN_{model_name}_{padded}_fp16.mlpackage"
    if not mlpkg.exists():
        print(f"  {model_name}: model not found, skipping")
        return -1.0

    model = ct.models.MLModel(str(mlpkg), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    spec = model.get_spec()
    out_key = spec.description.output[0].name

    img = load_image()
    img_padded = pad_reflect_np(img, PRE_PAD)
    x = np.transpose(img_padded, (2, 0, 1))[None].astype(np.float32)

    # Warmup
    for _ in range(WARMUP):
        model.predict({"input": x})

    # Timed runs
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        model.predict({"input": x})
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return sorted(times)[len(times) // 2]


# -- MLX benchmark --

def bench_mlx(model_name: str) -> float:
    """Benchmark MLX for a model. Returns median time in seconds."""
    import mlx.core as mx
    sys.path.insert(0, str(MLX_REPO))
    from upscale import load_model, upscale_image

    model, scale = load_model(model_name, dtype=mx.float16)
    img = load_image()

    # Warmup
    for _ in range(WARMUP):
        upscale_image(model, img, scale, dtype=mx.float16)

    # Timed runs
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        upscale_image(model, img, scale, dtype=mx.float16)
        mx.eval(mx.zeros(1))  # sync
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return sorted(times)[len(times) // 2]


# -- Quality check --

def quality_check(model_name: str) -> float:
    """Compare CoreML vs MLX output. Returns max absolute diff in [0,1] range."""
    import mlx.core as mx
    sys.path.insert(0, str(MLX_REPO))
    from upscale import load_model, upscale_image

    scale = MODEL_SCALES[model_name]
    padded = INPUT_SIZE + PRE_PAD
    mlpkg = WEIGHTS_DIR / f"RealESRGAN_{model_name}_{padded}_fp16.mlpackage"
    if not mlpkg.exists():
        return -1.0

    img = load_image()

    # CoreML output
    ct_model = ct.models.MLModel(str(mlpkg), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    spec = ct_model.get_spec()
    out_key = spec.description.output[0].name
    img_padded = pad_reflect_np(img, PRE_PAD)
    x = np.transpose(img_padded, (2, 0, 1))[None].astype(np.float32)
    result = ct_model.predict({"input": x})
    coreml_out = np.transpose(result[out_key][0], (1, 2, 0))
    oh, ow = coreml_out.shape[0], coreml_out.shape[1]
    coreml_out = coreml_out[:oh - PRE_PAD * scale, :ow - PRE_PAD * scale, :]
    coreml_out = np.clip(coreml_out, 0.0, 1.0)

    # MLX output
    mlx_model, mlx_scale = load_model(model_name, dtype=mx.float16)
    mlx_out = upscale_image(mlx_model, img, mlx_scale, dtype=mx.float16)

    # Compare (crop to min shape)
    h = min(coreml_out.shape[0], mlx_out.shape[0])
    w = min(coreml_out.shape[1], mlx_out.shape[1])
    return float(np.max(np.abs(coreml_out[:h, :w] - mlx_out[:h, :w])))


def main():
    ensure_bench_image()

    # Step 1: convert all models
    print("=" * 70)
    print("STEP 1: Converting all models to CoreML")
    print("=" * 70)
    convert_all()

    # Step 2: benchmark
    print("\n" + "=" * 70)
    print("STEP 2: Benchmarking (512x512 input, fp16, CPU_AND_GPU)")
    print("=" * 70)

    results = []
    for name in MODEL_NAMES:
        scale = MODEL_SCALES[name]
        out_size = f"{INPUT_SIZE * scale}x{INPUT_SIZE * scale}"

        print(f"\n--- {name} (scale={scale}, output={out_size}) ---")

        print(f"  CoreML...", end=" ", flush=True)
        t_coreml = bench_coreml(name)
        if t_coreml > 0:
            print(f"{t_coreml:.4f}s")
        else:
            print("SKIP")

        print(f"  MLX...", end=" ", flush=True)
        t_mlx = bench_mlx(name)
        print(f"{t_mlx:.4f}s")

        print(f"  Quality check...", end=" ", flush=True)
        max_diff = quality_check(name)
        print(f"max_diff={max_diff:.6f}")

        speedup = t_mlx / t_coreml if t_coreml > 0 else 0
        results.append((name, scale, out_size, t_coreml, t_mlx, speedup, max_diff))

    # Step 3: summary table
    print("\n" + "=" * 70)
    print("RESULTS (512x512 input, M3 Ultra, fp16)")
    print("=" * 70)
    header = f"{'Model':<12} | {'Scale':>5} | {'Output':>11} | {'CoreML':>8} | {'MLX':>8} | {'Speedup':>7} | {'MaxDiff':>8}"
    print(header)
    print("-" * len(header))
    for name, scale, out_size, t_c, t_m, speedup, diff in results:
        sc = f"{t_c:.3f}s" if t_c > 0 else "N/A"
        sm = f"{t_m:.3f}s"
        sp = f"{speedup:.1f}x" if speedup > 0 else "N/A"
        sd = f"{diff:.6f}" if diff >= 0 else "N/A"
        print(f"{name:<12} | {scale:>5} | {out_size:>11} | {sc:>8} | {sm:>8} | {sp:>7} | {sd:>8}")
    print("=" * 70)

    # Output markdown for README
    print("\n\nMarkdown table for README:\n")
    print("| Model | Scale | Output | CoreML (s) | MLX (s) | Speedup | Max Diff |")
    print("|-------|-------|--------|-----------|---------|---------|----------|")
    for name, scale, out_size, t_c, t_m, speedup, diff in results:
        sc = f"{t_c:.3f}" if t_c > 0 else "N/A"
        sm = f"{t_m:.3f}"
        sp = f"{speedup:.1f}x" if speedup > 0 else "N/A"
        sd = f"{diff:.6f}" if diff >= 0 else "N/A"
        print(f"| {name} | {scale}x | {out_size} | {sc} | {sm} | {sp} | {sd} |")


if __name__ == "__main__":
    main()
