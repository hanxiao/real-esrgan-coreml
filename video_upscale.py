"""Optimized video upscaling pipeline with fully pipelined I/O and inference.

Threaded read (PNG decode + tile prep) -> main thread inference -> threaded save (uint8 + PNG encode).
All I/O overlaps with GPU inference.
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np
from PIL import Image

from upscale import (
    MODEL_CONFIGS,
    PRE_PAD,
    _blend_weight,
    _prepare_tile,
    _unpad_tile,
    compute_tile_starts,
    load_coreml_model,
)


def _save_frame_png(output_path, frame_out):
    """Save a frame as PNG with fast compression."""
    out_uint8 = np.clip(frame_out * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(out_uint8, "RGB").save(output_path, compress_level=1)


def process_frames_with_io(
    frames_dir: Path,
    output_dir: Path,
    model_name: str,
    compute_unit: str,
    fp16: bool,
    model_size: int,
    scale: int,
    tile_size: int,
    tile_overlap: int,
    model=None,
    out_key: str = None,
) -> list[np.ndarray]:
    """Pipelined: threaded read -> inference -> threaded save.

    Returns list of output float32 arrays for quality comparison.
    """
    if model is None or out_key is None:
        model, out_key = load_coreml_model(model_name, model_size, compute_unit, fp16)

    output_dir.mkdir(exist_ok=True)
    paths = sorted(frames_dir.glob("frame_*.png"))
    n = len(paths)

    # Get frame dimensions from first file
    first_img = Image.open(paths[0]).convert("RGB")
    h, w = first_img.size[1], first_img.size[0]
    c = 3
    out_h, out_w = h * scale, w * scale
    single_tile = (h <= tile_size and w <= tile_size)

    # Precompute tile layout
    y_starts = compute_tile_starts(h, tile_size, tile_overlap)
    x_starts = compute_tile_starts(w, tile_size, tile_overlap)
    tile_specs = []
    for y0 in y_starts:
        for x0 in x_starts:
            tile_specs.append((y0, x0, min(y0 + tile_size, h), min(x0 + tile_size, w)))
    tiles_per_frame = len(tile_specs)

    # Precompute blend weights
    blend_weights = []
    if not single_tile:
        for y0, x0, y1, x1 in tile_specs:
            th, tw = y1 - y0, x1 - x0
            blend_weights.append(_blend_weight(th, tw, y0, x0, y1, x1, h, w, scale, tile_overlap))

    # --- Read thread: decode PNGs and prepare tile inputs ---
    read_queue = Queue(maxsize=4)

    def read_worker():
        for fi, p in enumerate(paths):
            img = Image.open(p).convert("RGB")
            rgb = np.array(img, dtype=np.float32) / 255.0
            tile_inputs = []
            if single_tile:
                chw, ph, pw, th, tw = _prepare_tile(rgb, 0, 0, h, w, model_size, PRE_PAD)
                tile_inputs.append((chw[None], ph, pw, th, tw, 0))
            else:
                for ti, (y0, x0, y1, x1) in enumerate(tile_specs):
                    chw, ph, pw, th, tw = _prepare_tile(rgb, y0, x0, y1, x1, model_size, PRE_PAD)
                    tile_inputs.append((chw[None], ph, pw, th, tw, ti))
            read_queue.put((fi, p.name, tile_inputs))
        read_queue.put(None)

    # --- Save pool ---
    save_pool = ThreadPoolExecutor(max_workers=8)
    save_futures = []

    read_thread = Thread(target=read_worker, daemon=True)
    read_thread.start()

    outputs = [None] * n
    # Pre-allocate input buffer (reused for every predict call)
    input_buf = np.empty((1, c, model_size, model_size), dtype=np.float32)

    while True:
        item = read_queue.get()
        if item is None:
            break

        fi, filename, tile_inputs = item

        if single_tile:
            x, ph, pw, th, tw, ti = tile_inputs[0]
            input_buf[0] = x[0]
            result = model.predict({"input": input_buf})
            out_nchw = result[out_key][0]
            frame_out = _unpad_tile(out_nchw, ph, pw, scale, PRE_PAD).astype(np.float32)
        else:
            frame_out = np.zeros((out_h, out_w, c), dtype=np.float32)
            weight_map = np.zeros((out_h, out_w, 1), dtype=np.float32)
            for x, ph, pw, th, tw, ti in tile_inputs:
                input_buf[0] = x[0]
                result = model.predict({"input": input_buf})
                out_nchw = result[out_key][0]
                tile_out = _unpad_tile(out_nchw, ph, pw, scale, PRE_PAD)
                y0, x0 = tile_specs[ti][0], tile_specs[ti][1]
                bw = blend_weights[ti]
                oy0, ox0 = y0 * scale, x0 * scale
                frame_out[oy0:oy0 + th * scale, ox0:ox0 + tw * scale, :] += tile_out * bw
                weight_map[oy0:oy0 + th * scale, ox0:ox0 + tw * scale, :] += bw
            frame_out = np.clip(frame_out / np.maximum(weight_map, 1e-8), 0.0, 1.0).astype(np.float32)

        outputs[fi] = frame_out
        save_futures.append(save_pool.submit(_save_frame_png, str(output_dir / filename), frame_out))
        print(f"\rPipelined: frame {fi+1}/{n}", end="", flush=True)

    # Wait for saves
    for f in save_futures:
        f.result()
    save_pool.shutdown(wait=True)
    read_thread.join()
    print()
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Optimized video upscaling")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("-o", "--output", help="Output video path")
    args = parser.parse_args()
    print("Use benchmark_video.py for testing")


if __name__ == "__main__":
    main()
