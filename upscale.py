"""Real-ESRGAN inference via CoreML backend with automatic tile-based processing.

Usage:
    python upscale.py input.png -o output.png
    python upscale.py input.png --model animevideo --compute-unit CPU_AND_GPU
    python upscale.py input.png --tile-size 256 --tile-overlap 16
    python upscale.py --convert --model x4plus --size 522

Requires: coremltools (no torch at runtime).
"""

import argparse
import math
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
from PIL import Image

WEIGHTS_DIR = Path(__file__).parent / "weights"
PRE_PAD = 10

MODEL_CONFIGS = {
    "x4plus": dict(scale=4),
    "x2plus": dict(scale=2),
    "anime_6B": dict(scale=4),
    "animevideo": dict(scale=4),
    "general": dict(scale=4),
}


def get_mlpackage_path(model_name: str, input_size: int, fp16: bool = True) -> Path:
    """Return expected .mlpackage path for given model and input size."""
    dtype_label = "fp16" if fp16 else "fp32"
    return WEIGHTS_DIR / f"RealESRGAN_{model_name}_{input_size}_{dtype_label}.mlpackage"


def ensure_model(model_name: str, input_size: int, fp16: bool = True) -> Path:
    """Return .mlpackage path, auto-converting if it doesn't exist."""
    path = get_mlpackage_path(model_name, input_size, fp16)
    if path.exists():
        return path
    print(f"Model not found at {path.name}, auto-converting...")
    from convert import convert
    convert(model_name=model_name, input_size=input_size, use_fp16=fp16)
    return path


def load_coreml_model(model_name: str, input_size: int, compute_unit: str = "ALL", fp16: bool = True):
    """Load CoreML model. Returns (model, output_key_name)."""
    path = ensure_model(model_name, input_size, fp16)

    cu_map = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    }
    cu = cu_map.get(compute_unit.upper(), ct.ComputeUnit.ALL)

    print(f"Loading CoreML model: {path.name} (compute_units={compute_unit})")
    t0 = time.time()
    model = ct.models.MLModel(str(path), compute_units=cu)
    print(f"Loaded in {time.time() - t0:.1f}s")

    spec = model.get_spec()
    out_key = spec.description.output[0].name
    return model, out_key


def pad_reflect_np(img: np.ndarray, pad_bottom: int, pad_right: int) -> np.ndarray:
    """Reflect-pad a (H, W, C) array on bottom and right edges."""
    if pad_bottom == 0 and pad_right == 0:
        return img
    return np.pad(img,
                  ((0, pad_bottom), (0, pad_right), (0, 0)),
                  mode='reflect')


def upscale_tile_coreml(
    model,
    out_key: str,
    tile: np.ndarray,
    model_size: int,
    scale: int = 4,
    pre_pad: int = PRE_PAD,
) -> np.ndarray:
    """Upscale a single tile (H, W, C) float32 [0,1] via CoreML. Returns (H*scale, W*scale, C)."""
    h, w, c = tile.shape

    # Pre-pad (right and bottom only)
    if pre_pad > 0:
        tile = pad_reflect_np(tile, pre_pad, pre_pad)

    # Pad to square (model_size x model_size)
    padded_h, padded_w = tile.shape[:2]
    pad_h = model_size - padded_h
    pad_w = model_size - padded_w
    if pad_h > 0 or pad_w > 0:
        tile = pad_reflect_np(tile, max(pad_h, 0), max(pad_w, 0))

    # HWC -> NCHW
    x = np.transpose(tile, (2, 0, 1))[None].astype(np.float32)

    # Inference
    result = model.predict({"input": x})
    output = result[out_key]  # NCHW

    # NCHW -> HWC
    output = np.transpose(output[0], (1, 2, 0))

    # Remove square padding
    if pad_h > 0 or pad_w > 0:
        output = output[:padded_h * scale, :padded_w * scale, :]

    # Remove pre-pad
    if pre_pad > 0:
        oh, ow = output.shape[0], output.shape[1]
        output = output[:oh - pre_pad * scale, :ow - pre_pad * scale, :]

    return np.clip(output, 0.0, 1.0).astype(np.float32)


def compute_tile_starts(total_size: int, tile_size: int, overlap: int) -> list:
    """Return list of start positions for tiles along one axis."""
    if total_size <= tile_size:
        return [0]
    stride = tile_size - overlap
    positions = []
    pos = 0
    while pos < total_size:
        if pos + tile_size >= total_size:
            positions.append(total_size - tile_size)
            break
        positions.append(pos)
        pos += stride
    return positions


def upscale_image_coreml(
    model,
    out_key: str,
    img_array: np.ndarray,
    model_size: int,
    scale: int = 4,
    pre_pad: int = PRE_PAD,
    tile_size: int = 512,
    tile_overlap: int = 32,
) -> np.ndarray:
    """Upscale (H, W, C) float32 [0,1] image via CoreML with automatic tiling."""
    h, w, c = img_array.shape

    # Check if image fits in a single tile
    if h <= tile_size and w <= tile_size:
        return upscale_tile_coreml(model, out_key, img_array, model_size, scale, pre_pad)

    # Tiled processing
    out_h, out_w = h * scale, w * scale
    output = np.zeros((out_h, out_w, c), dtype=np.float32)
    weight_map = np.zeros((out_h, out_w, 1), dtype=np.float32)

    y_starts = compute_tile_starts(h, tile_size, tile_overlap)
    x_starts = compute_tile_starts(w, tile_size, tile_overlap)
    total = len(y_starts) * len(x_starts)

    tile_num = 0
    for y0 in y_starts:
        for x0 in x_starts:
            tile_num += 1

            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            th = y1 - y0
            tw = x1 - x0

            tile = img_array[y0:y1, x0:x1, :]
            tile_out = upscale_tile_coreml(model, out_key, tile, model_size, scale, pre_pad)

            # Build blend weight with linear ramps in overlap regions
            tile_weight = np.ones((th * scale, tw * scale, 1), dtype=np.float32)
            ramp_pixels = tile_overlap * scale

            if ramp_pixels > 0:
                ramp = np.linspace(0, 1, ramp_pixels, dtype=np.float32)
                # Top ramp (not for first row of tiles)
                if y0 > 0:
                    tile_weight[:ramp_pixels, :, :] *= ramp[:, None, None]
                # Bottom ramp (not for last row of tiles)
                if y1 < h:
                    tile_weight[-ramp_pixels:, :, :] *= ramp[::-1, None, None]
                # Left ramp (not for first column)
                if x0 > 0:
                    tile_weight[:, :ramp_pixels, :] *= ramp[None, :, None]
                # Right ramp (not for last column)
                if x1 < w:
                    tile_weight[:, -ramp_pixels:, :] *= ramp[None, ::-1, None]

            oy0 = y0 * scale
            ox0 = x0 * scale
            oy1 = oy0 + th * scale
            ox1 = ox0 + tw * scale

            output[oy0:oy1, ox0:ox1, :] += tile_out * tile_weight
            weight_map[oy0:oy1, ox0:ox1, :] += tile_weight

            print(f"\rTile {tile_num}/{total}", end="", flush=True)

    if total > 1:
        print()

    output = output / np.maximum(weight_map, 1e-8)
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def process_image(
    input_path: str,
    output_path: str,
    model_name: str = "x4plus",
    compute_unit: str = "ALL",
    fp16: bool = True,
    pre_pad: int = PRE_PAD,
    tile_size: int = 512,
    tile_overlap: int = 32,
):
    """Full pipeline: load image, upscale via CoreML, save."""
    scale = MODEL_CONFIGS[model_name]["scale"]

    img = Image.open(input_path).convert("RGB")
    h, w = img.size[1], img.size[0]  # PIL is (w, h)
    print(f"Input: {w}x{h}")

    # Model size = tile_size + pre_pad (fixed square input)
    model_size = tile_size + pre_pad
    model, out_key = load_coreml_model(model_name, model_size, compute_unit, fp16)

    rgb_array = np.array(img, dtype=np.float32) / 255.0

    print("Upscaling...")
    t0 = time.time()
    output = upscale_image_coreml(
        model, out_key, rgb_array, model_size, scale, pre_pad,
        tile_size, tile_overlap,
    )
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.2f}s")

    output_uint8 = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    result = Image.fromarray(output_uint8, "RGB")
    result.save(output_path)
    oh, ow = result.size[1], result.size[0]
    print(f"Output: {ow}x{oh} -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Real-ESRGAN upscaling with CoreML")
    parser.add_argument("input", nargs="?", help="Input image path")
    parser.add_argument("-o", "--output", help="Output image path")
    parser.add_argument("--model", default="x4plus", choices=list(MODEL_CONFIGS.keys()),
                        help="Model variant (default: x4plus)")
    parser.add_argument("--compute-unit", default="ALL",
                        choices=["ALL", "CPU_AND_GPU", "CPU_ONLY", "CPU_AND_NE"],
                        help="CoreML compute unit (default: ALL)")
    parser.add_argument("--fp32", action="store_true", help="Use fp32 model")
    parser.add_argument("--pre-pad", type=int, default=PRE_PAD)
    parser.add_argument("--tile-size", type=int, default=512,
                        help="Tile size for processing (default: 512)")
    parser.add_argument("--tile-overlap", type=int, default=32,
                        help="Overlap between tiles (default: 32)")
    # Conversion shortcut
    parser.add_argument("--convert", action="store_true",
                        help="Run conversion (requires torch)")
    parser.add_argument("--size", type=int, default=512,
                        help="Input size for conversion")
    args = parser.parse_args()

    if args.convert:
        from convert import convert
        convert(model_name=args.model, input_size=args.size, use_fp16=not args.fp32)
        return

    if not args.input:
        parser.error("input image path required (or use --convert)")

    if args.output is None:
        p = Path(args.input)
        args.output = str(p.with_stem(p.stem + "_coreml_out"))

    process_image(
        args.input, args.output,
        model_name=args.model,
        compute_unit=args.compute_unit,
        fp16=not args.fp32,
        pre_pad=args.pre_pad,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
    )


if __name__ == "__main__":
    main()
