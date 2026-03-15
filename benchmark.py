"""Benchmark CoreML vs MLX backends for Real-ESRGAN x4plus.

Usage: uv run python coreml_benchmark.py
"""

import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
from PIL import Image

REPO = Path(__file__).parent
BENCH_512 = REPO / "bench_input.png"
BENCH_1024 = REPO / "bench_input_1024.png"
REF_PATH = REPO / "reference" / "ref_512_x4plus.npy"
WEIGHTS_DIR = REPO / "weights"
SCALE = 4
PRE_PAD = 10
WARMUP = 2
RUNS = 5


def ensure_1024_input():
    if not BENCH_1024.exists():
        img = Image.open(BENCH_512).resize((1024, 1024), Image.LANCZOS)
        img.save(BENCH_1024)


def load_image(path: Path) -> np.ndarray:
    """Load image as (H, W, 3) float32 [0, 1]."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def pad_reflect_np(img: np.ndarray, pad: int) -> np.ndarray:
    if pad == 0:
        return img
    return np.pad(img, ((0, pad), (0, pad), (0, 0)), mode='reflect')


# -- CoreML benchmarking --

def bench_coreml(input_size: int, compute_unit_str: str, fp16: bool = True):
    """Benchmark CoreML for a given input size and compute unit. Returns (median_time, output_for_quality)."""
    padded = input_size + PRE_PAD
    dtype_label = "fp16" if fp16 else "fp32"
    mlpackage = WEIGHTS_DIR / f"RealESRGAN_x4plus_{padded}_{dtype_label}.mlpackage"

    if not mlpackage.exists():
        print(f"  Model not found: {mlpackage.name}")
        print(f"  Run: uv run python coreml_convert.py --size {padded}" +
              (" --fp32" if not fp16 else ""))
        return None, None

    cu_map = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
    }
    cu = cu_map[compute_unit_str]

    model = ct.models.MLModel(str(mlpackage), compute_units=cu)
    spec = model.get_spec()
    out_key = spec.description.output[0].name

    img = load_image(BENCH_512 if input_size == 512 else BENCH_1024)
    img_padded = pad_reflect_np(img, PRE_PAD)
    x = np.transpose(img_padded, (2, 0, 1))[None].astype(np.float32)

    # Warmup
    for _ in range(WARMUP):
        model.predict({"input": x})

    # Timed runs
    times = []
    output = None
    for _ in range(RUNS):
        t0 = time.perf_counter()
        result = model.predict({"input": x})
        t1 = time.perf_counter()
        times.append(t1 - t0)
        output = result[out_key]

    median_t = sorted(times)[len(times) // 2]

    # Convert output to HWC and crop pre-pad
    if output is not None:
        out_hwc = np.transpose(output[0], (1, 2, 0))
        oh, ow = out_hwc.shape[0], out_hwc.shape[1]
        out_hwc = out_hwc[:oh - PRE_PAD * SCALE, :ow - PRE_PAD * SCALE, :]
        out_hwc = np.clip(out_hwc, 0.0, 1.0).astype(np.float32)
    else:
        out_hwc = None

    return median_t, out_hwc


# -- MLX benchmarking --

def bench_mlx(input_size: int):
    """Benchmark MLX backend. Returns (median_time, output_for_quality)."""
    import mlx.core as mx
    sys.path.insert(0, str(REPO))
    from upscale import load_model, upscale_image

    model, scale = load_model("x4plus", dtype=mx.float16)
    img = load_image(BENCH_512 if input_size == 512 else BENCH_1024)

    # Warmup
    for _ in range(WARMUP):
        upscale_image(model, img, scale, dtype=mx.float16)

    # Timed runs
    times = []
    output = None
    for _ in range(RUNS):
        t0 = time.perf_counter()
        output = upscale_image(model, img, scale, dtype=mx.float16)
        mx.eval(mx.zeros(1))  # sync
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return sorted(times)[len(times) // 2], output


def quality_diff(output: np.ndarray) -> float:
    """Compute max absolute diff vs reference (float32). Returns max_diff or -1 if no ref."""
    if not REF_PATH.exists():
        return -1.0
    ref = np.load(REF_PATH)
    # Ensure same shape
    h = min(output.shape[0], ref.shape[0])
    w = min(output.shape[1], ref.shape[1])
    return float(np.max(np.abs(output[:h, :w] - ref[:h, :w])))


def main():
    ensure_1024_input()

    # Check which CoreML models are available
    available_sizes = []
    for size in [512, 1024]:
        padded = size + PRE_PAD
        p = WEIGHTS_DIR / f"RealESRGAN_x4plus_{padded}_fp16.mlpackage"
        if p.exists():
            available_sizes.append(size)

    if not available_sizes:
        print("No CoreML models found. Convert first:")
        print("  uv run python coreml_convert.py --size 522")
        print("  uv run python coreml_convert.py --size 1034")
        print("(522 = 512 + 10 pre_pad, 1034 = 1024 + 10 pre_pad)")
        sys.exit(1)

    results = []

    # MLX benchmark
    print("=" * 65)
    print("Benchmarking MLX (fp16)...")
    mlx_512, mlx_512_out = bench_mlx(512)
    mlx_512_q = quality_diff(mlx_512_out) if mlx_512_out is not None else -1
    print(f"  512x512: {mlx_512:.4f}s  quality_diff={mlx_512_q:.6f}")

    mlx_1024, _ = bench_mlx(1024)
    print(f"  1024x1024: {mlx_1024:.4f}s")
    results.append(("MLX (fp16)", mlx_512, mlx_1024, mlx_512_q))

    # CoreML benchmarks
    for cu_name in ["ALL", "CPU_AND_GPU"]:
        label = f"CoreML ({cu_name})"
        print(f"\nBenchmarking {label}...")

        t512, out512 = (None, None)
        t1024 = None

        if 512 in available_sizes:
            t512, out512 = bench_coreml(512, cu_name)
            q512 = quality_diff(out512) if out512 is not None else -1
            if t512 is not None:
                print(f"  512x512: {t512:.4f}s  quality_diff={q512:.6f}")
        else:
            q512 = -1

        if 1024 in available_sizes:
            t1024, _ = bench_coreml(1024, cu_name)
            if t1024 is not None:
                print(f"  1024x1024: {t1024:.4f}s")

        results.append((label, t512, t1024, q512))

    # Summary table
    print("\n" + "=" * 65)
    print(f"{'Backend':<20} | {'512x512':>10} | {'1024x1024':>10} | {'Quality':>10}")
    print("-" * 65)
    for name, t512, t1024, q in results:
        s512 = f"{t512:.4f}s" if t512 is not None else "N/A"
        s1024 = f"{t1024:.4f}s" if t1024 is not None else "N/A"
        sq = f"{q:.6f}" if q >= 0 else "N/A"
        print(f"{name:<20} | {s512:>10} | {s1024:>10} | {sq:>10}")
    print("=" * 65)


if __name__ == "__main__":
    main()
