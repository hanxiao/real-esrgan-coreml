"""Benchmark all backends x all models: speed + power consumption.

Runs each model with multiple images, measures average time and power draw.
Requires sudo for powermetrics.

Usage: sudo uv run python benchmark_power.py
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import coremltools as ct
import numpy as np
from PIL import Image

# Import local modules FIRST (before MLX repo)
SCRIPT_DIR = str(Path(__file__).parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

MODELS = ["x4plus", "x2plus", "anime_6B", "animevideo", "general"]
COMPUTE_UNITS = ["CPU_AND_GPU", "ALL"]  # ALL = ANE+GPU+CPU
INPUT_SIZES = [256, 512, 768]  # multiple sizes for averaging
MIN_DURATION = 30.0  # minimum seconds per power measurement
WARMUP = 2

# Import after path setup
from convert import MODEL_CONFIGS


def create_test_images():
    """Create test images of various sizes."""
    imgs = {}
    np.random.seed(42)
    for sz in INPUT_SIZES:
        img = np.random.randint(0, 256, (sz, sz, 3), dtype=np.uint8)
        imgs[sz] = img.astype(np.float32) / 255.0
    return imgs


def get_padded_size(sz, pre_pad=10):
    return sz + pre_pad


def ensure_converted(model_name):
    """Convert model for all input sizes if not already done."""
    from convert import convert as do_convert
    for sz in INPUT_SIZES:
        padded = get_padded_size(sz)
        mlpkg = Path("weights") / f"RealESRGAN_{model_name}_{padded}_fp16.mlpackage"
        if not mlpkg.exists():
            print(f"  Converting {model_name} for size {padded}...")
            do_convert(model_name=model_name, input_size=padded, use_fp16=True)


def measure_power_during(func, duration_hint=5.0):
    """Run func() while measuring power with macmon (sudoless).
    Returns (func_result, avg_power_watts, energy_joules, elapsed_seconds).
    Also returns detailed power breakdown as dict.
    """
    pm_proc = subprocess.Popen(
        ["macmon", "pipe", "-i", "100"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    
    time.sleep(0.3)  # let macmon start
    
    t0 = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - t0
    
    time.sleep(0.2)
    pm_proc.terminate()
    stdout, _ = pm_proc.communicate(timeout=3)
    
    # Parse JSON lines
    import json as _json
    cpu_powers = []
    gpu_powers = []
    ane_powers = []
    all_powers = []
    for line in stdout.decode('utf-8', errors='replace').strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            d = _json.loads(line)
            cpu_powers.append(d.get("cpu_power", 0))
            gpu_powers.append(d.get("gpu_power", 0))
            ane_powers.append(d.get("ane_power", 0))
            all_powers.append(d.get("all_power", 0))
        except _json.JSONDecodeError:
            pass
    
    avg_power = np.mean(all_powers) if all_powers else -1
    energy = avg_power * elapsed if avg_power > 0 else -1
    
    return result, avg_power, energy, elapsed


def bench_coreml(model_name, compute_unit, test_images):
    """Benchmark CoreML backend. Returns (avg_time, avg_power, avg_energy)."""
    from upscale import load_coreml_model, upscale_image_coreml
    
    config = MODEL_CONFIGS[model_name]
    scale = config.get("scale", config.get("upscale", 4))
    
    times = []
    all_powers = []
    all_energies = []
    
    for sz, img in test_images.items():
        padded = get_padded_size(sz)
        try:
            model, out_key = load_coreml_model(
                model_name, padded,
                compute_unit=compute_unit, fp16=True
            )
        except Exception as ex:
            print(f"    Skipping {sz}x{sz} - {ex}")
            continue
        
        # Warmup
        x = np.pad(img, ((0, 10), (0, 10), (0, 0)), mode='reflect')
        x_nchw = np.transpose(x, (2, 0, 1))[None].astype(np.float32)
        for _ in range(WARMUP):
            model.predict({"input": x_nchw})
        
        # Estimate iterations needed for MIN_DURATION
        t_single = time.perf_counter()
        model.predict({"input": x_nchw})
        t_single = time.perf_counter() - t_single
        num_runs = max(3, int(MIN_DURATION / t_single) + 1)
        
        # Timed runs with power measurement
        def run_batch():
            for _ in range(num_runs):
                model.predict({"input": x_nchw})
        
        _, avg_power, energy, elapsed = measure_power_during(run_batch)
        avg_time = elapsed / num_runs
        times.append(avg_time)
        if avg_power > 0:
            all_powers.append(avg_power)
            all_energies.append(energy / num_runs)
    
    return (
        np.mean(times) if times else -1,
        np.mean(all_powers) if all_powers else -1,
        np.mean(all_energies) if all_energies else -1,
    )


def bench_mlx(model_name, test_images):
    """Benchmark MLX backend. Returns (avg_time, avg_power, avg_energy)."""
    try:
        import mlx.core as mx
        MLX_REPO = os.path.expanduser("~/.openclaw/workspace/real-esrgan-mlx")
        import importlib.util
        spec = importlib.util.spec_from_file_location("mlx_upscale", os.path.join(MLX_REPO, "upscale.py"))
        mlx_upscale_mod = importlib.util.module_from_spec(spec)
        # Need mlx model.py too
        model_spec = importlib.util.spec_from_file_location("model", os.path.join(MLX_REPO, "model.py"))
        model_mod = importlib.util.module_from_spec(model_spec)
        sys.modules["model"] = model_mod
        model_spec.loader.exec_module(model_mod)
        spec.loader.exec_module(mlx_upscale_mod)
        mlx_load_model = mlx_upscale_mod.load_model
        mlx_upscale = mlx_upscale_mod.upscale_image
    except ImportError:
        return -1, -1, -1
    
    try:
        model, scale = mlx_load_model(model_name, dtype=mx.float16)
    except Exception as e:
        print(f"    MLX load failed: {e}")
        return -1, -1, -1
    
    times = []
    all_powers = []
    all_energies = []
    
    for sz, img in test_images.items():
        # Warmup
        for _ in range(WARMUP):
            mlx_upscale(model, img, scale, dtype=mx.float16)
            mx.eval(mx.zeros(1))
        
        # Estimate iterations
        t_single = time.perf_counter()
        mlx_upscale(model, img, scale, dtype=mx.float16)
        mx.eval(mx.zeros(1))
        t_single = time.perf_counter() - t_single
        num_runs = max(3, int(MIN_DURATION / t_single) + 1)
        
        def run_batch():
            for _ in range(num_runs):
                mlx_upscale(model, img, scale, dtype=mx.float16)
                mx.eval(mx.zeros(1))
        
        _, avg_power, energy, elapsed = measure_power_during(run_batch)
        avg_time = elapsed / num_runs
        times.append(avg_time)
        if avg_power > 0:
            all_powers.append(avg_power)
            all_energies.append(energy / num_runs)
    
    return (
        np.mean(times) if times else -1,
        np.mean(all_powers) if all_powers else -1,
        np.mean(all_energies) if all_energies else -1,
    )


def main():
    no_power = False
    
    print("Creating test images...")
    test_images = create_test_images()
    print(f"Sizes: {list(test_images.keys())}")
    
    print("\nConverting models...")
    for m in MODELS:
        ensure_converted(m)
    
    results = []
    
    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        
        row = {"model": model_name}
        
        # CoreML CPU+GPU
        print(f"  CoreML CPU+GPU...")
        t, p, e = bench_coreml(model_name, "CPU_AND_GPU", test_images)
        row["coreml_cpugpu_time"] = t
        row["coreml_cpugpu_power"] = p
        row["coreml_cpugpu_energy"] = e
        print(f"    Time: {t:.4f}s | Power: {p:.1f}W | Energy: {e:.2f}J")
        
        # CoreML ALL (ANE)
        print(f"  CoreML ALL (ANE)...")
        t, p, e = bench_coreml(model_name, "ALL", test_images)
        row["coreml_all_time"] = t
        row["coreml_all_power"] = p
        row["coreml_all_energy"] = e
        print(f"    Time: {t:.4f}s | Power: {p:.1f}W | Energy: {e:.2f}J")
        
        # MLX
        print(f"  MLX...")
        t, p, e = bench_mlx(model_name, test_images)
        row["mlx_time"] = t
        row["mlx_power"] = p
        row["mlx_energy"] = e
        print(f"    Time: {t:.4f}s | Power: {p:.1f}W | Energy: {e:.2f}J")
        
        results.append(row)
    
    # Summary table
    print(f"\n{'='*90}")
    print(f"{'Model':<12} | {'Backend':<16} | {'Avg Time':>10} | {'Power (W)':>10} | {'Energy (J)':>10}")
    print(f"{'-'*90}")
    for row in results:
        m = row["model"]
        for backend, prefix in [("CoreML CPU+GPU", "coreml_cpugpu"), 
                                 ("CoreML ALL/ANE", "coreml_all"),
                                 ("MLX fp16", "mlx")]:
            t = row[f"{prefix}_time"]
            p = row[f"{prefix}_power"]
            e = row[f"{prefix}_energy"]
            ts = f"{t:.4f}s" if t > 0 else "N/A"
            ps = f"{p:.1f}" if p > 0 else "N/A"
            es = f"{e:.3f}" if e > 0 else "N/A"
            print(f"{m:<12} | {backend:<16} | {ts:>10} | {ps:>10} | {es:>10}")
        print(f"{'-'*90}")
    print(f"{'='*90}")
    
    # Save results
    with open("benchmark_power_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: benchmark_power_results.json")


if __name__ == "__main__":
    main()
