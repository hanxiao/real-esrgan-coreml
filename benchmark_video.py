"""Benchmark video frame upscaling: baseline vs optimized pipeline.

Usage:
    uv run python benchmark_video.py [--frames-dir /tmp/esrgan_video5/frames/]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from upscale import (
    MODEL_CONFIGS,
    PRE_PAD,
    ensure_model,
    load_coreml_model,
    upscale_image_coreml,
)

RESULTS_FILE = Path(__file__).parent / "results.tsv"

MODEL_NAME = "x4plus"
COMPUTE_UNIT = "CPU_AND_GPU"
TILE_SIZE = 512
TILE_OVERLAP = 32
FP16 = True


def run_baseline(model, out_key, frames_dir: Path, output_dir: Path, model_size, scale):
    """Sequential: read PNG -> infer -> save PNG. Returns list of output arrays."""
    paths = sorted(frames_dir.glob("frame_*.png"))
    n = len(paths)
    outputs = []
    output_dir.mkdir(exist_ok=True)
    for i, p in enumerate(paths):
        # Read
        img = Image.open(p).convert("RGB")
        rgb = np.array(img, dtype=np.float32) / 255.0
        # Infer
        out = upscale_image_coreml(
            model, out_key, rgb, model_size, scale, PRE_PAD,
            TILE_SIZE, TILE_OVERLAP, use_batch=False,
        )
        outputs.append(out)
        # Save
        out_uint8 = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(out_uint8, "RGB").save(output_dir / p.name)
        print(f"\rBaseline: frame {i+1}/{n}", end="", flush=True)
    print()
    return outputs


def run_optimized(frames_dir: Path, output_dir: Path, model_size, scale, model=None, out_key=None):
    """Run the optimized pipeline. Returns list of output arrays."""
    from video_upscale import process_frames_with_io
    return process_frames_with_io(
        frames_dir, output_dir,
        MODEL_NAME, COMPUTE_UNIT, FP16,
        model_size, scale, TILE_SIZE, TILE_OVERLAP,
        model=model, out_key=out_key,
    )


def compare_outputs(baseline: list[np.ndarray], optimized: list[np.ndarray]) -> int:
    """Return max absolute pixel difference (0-255 scale)."""
    max_diff = 0
    for b, o in zip(baseline, optimized):
        b_uint8 = np.clip(b * 255.0, 0, 255).astype(np.uint8)
        o_uint8 = np.clip(o * 255.0, 0, 255).astype(np.uint8)
        diff = int(np.max(np.abs(b_uint8.astype(int) - o_uint8.astype(int))))
        if diff > max_diff:
            max_diff = diff
    return max_diff


def log_result(experiment: str, speedup: float, fps: float, total_time: float, max_diff: int, status: str):
    """Append result to results.tsv."""
    header = "experiment\tspeedup\tfps\ttotal_time\tmax_diff\tstatus\n"
    if not RESULTS_FILE.exists():
        with open(RESULTS_FILE, "w") as f:
            f.write(header)
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{experiment}\t{speedup:.2f}\t{fps:.2f}\t{total_time:.1f}\t{max_diff}\t{status}\n")
    print(f"Logged: {experiment} speedup={speedup:.2f} fps={fps:.2f} time={total_time:.1f}s max_diff={max_diff} {status}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", default="/tmp/esrgan_video5/frames/")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--optimized-only", action="store_true")
    parser.add_argument("--experiment", default="optimized")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    n = len(list(frames_dir.glob("frame_*.png")))
    print(f"Found {n} frames in {frames_dir}")

    scale = MODEL_CONFIGS[MODEL_NAME]["scale"]
    model_size = TILE_SIZE + PRE_PAD

    model, out_key = load_coreml_model(MODEL_NAME, model_size, COMPUTE_UNIT, FP16)

    # Warmup
    print("Warmup...")
    warmup_img = np.array(Image.open(sorted(frames_dir.glob("frame_*.png"))[0]).convert("RGB"), dtype=np.float32) / 255.0
    _ = upscale_image_coreml(model, out_key, warmup_img, model_size, scale, PRE_PAD, TILE_SIZE, TILE_OVERLAP)

    baseline_dir = Path("/tmp/esrgan_video5/baseline_outputs")
    optimized_dir = Path("/tmp/esrgan_video5/optimized_outputs")
    baseline_outputs = None
    baseline_time = None

    if not args.optimized_only:
        print("\n--- Baseline (sequential read -> infer -> save) ---")
        t0 = time.time()
        baseline_outputs = run_baseline(model, out_key, frames_dir, baseline_dir, model_size, scale)
        baseline_time = time.time() - t0
        baseline_fps = n / baseline_time
        print(f"Baseline: {baseline_time:.1f}s total, {baseline_fps:.2f} fps")

        if args.baseline_only:
            log_result("baseline", 1.00, baseline_fps, baseline_time, 0, "KEEP")
            return

    print("\n--- Optimized ---")
    t0 = time.time()
    optimized_outputs = run_optimized(frames_dir, optimized_dir, model_size, scale, model=model, out_key=out_key)
    optimized_time = time.time() - t0
    optimized_fps = n / optimized_time
    print(f"Optimized: {optimized_time:.1f}s total, {optimized_fps:.2f} fps")

    # Compare
    if baseline_outputs is None:
        if not baseline_dir.exists():
            print("No baseline outputs found. Run --baseline-only first.")
            sys.exit(1)
        baseline_outputs = []
        for p in sorted(baseline_dir.glob("frame_*.png")):
            img = Image.open(p).convert("RGB")
            baseline_outputs.append(np.array(img, dtype=np.float32) / 255.0)
        with open(RESULTS_FILE) as f:
            for line in f:
                if line.startswith("baseline"):
                    baseline_time = float(line.split("\t")[3])

    max_diff = compare_outputs(baseline_outputs, optimized_outputs)
    speedup = baseline_time / optimized_time if baseline_time else 0

    print(f"\nMax pixel diff: {max_diff}")
    print(f"Speedup: {speedup:.2f}x")

    status = "KEEP" if max_diff == 0 and speedup > 1.0 else "DISCARD"
    if max_diff > 0:
        status = f"DISCARD (quality loss: max_diff={max_diff})"

    log_result(args.experiment, speedup, optimized_fps, optimized_time, max_diff, status)


if __name__ == "__main__":
    main()
