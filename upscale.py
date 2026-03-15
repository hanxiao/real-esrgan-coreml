"""Real-ESRGAN inference via CoreML backend.

Usage:
    python upscale.py input.png -o output.png
    python upscale.py input.png --model animevideo --compute-unit CPU_AND_GPU
    python upscale.py --convert --model x4plus --size 522

Requires: coremltools (no torch at runtime).
"""

import argparse
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


def load_coreml_model(model_name: str, input_size: int, compute_unit: str = "ALL", fp16: bool = True):
    """Load CoreML model. Returns (model, output_key_name)."""
    path = get_mlpackage_path(model_name, input_size, fp16)
    if not path.exists():
        print(f"Model not found at {path}")
        print(f"Run: uv run python convert.py --model {model_name} --size {input_size}" +
              (" --fp32" if not fp16 else ""))
        sys.exit(1)

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


def upscale_image_coreml(
    model,
    out_key: str,
    img_array: np.ndarray,
    scale: int = 4,
    pre_pad: int = PRE_PAD,
) -> np.ndarray:
    """Upscale (H, W, C) float32 [0,1] image via CoreML. Returns (H*scale, W*scale, C) float32."""
    h, w, c = img_array.shape

    # Pre-pad (right and bottom only, matching MLX path)
    if pre_pad > 0:
        img_array = pad_reflect_np(img_array, pre_pad, pre_pad)

    # HWC -> NCHW for CoreML
    x = np.transpose(img_array, (2, 0, 1))[None].astype(np.float32)

    # Run inference
    result = model.predict({"input": x})
    output = result[out_key]  # NCHW

    # NCHW -> HWC
    output = np.transpose(output[0], (1, 2, 0))

    # Remove pre-pad from output
    if pre_pad > 0:
        oh, ow = output.shape[0], output.shape[1]
        output = output[:oh - pre_pad * scale, :ow - pre_pad * scale, :]

    return np.clip(output, 0.0, 1.0).astype(np.float32)


def process_image(
    input_path: str,
    output_path: str,
    model_name: str = "x4plus",
    compute_unit: str = "ALL",
    fp16: bool = True,
    pre_pad: int = PRE_PAD,
):
    """Full pipeline: load image, upscale via CoreML, save."""
    scale = MODEL_CONFIGS[model_name]["scale"]

    img = Image.open(input_path).convert("RGB")
    h, w = img.size[1], img.size[0]  # PIL is (w, h)
    print(f"Input: {w}x{h}")

    # The CoreML model has a fixed input size
    padded_h = h + pre_pad
    padded_w = w + pre_pad
    mlpackage_path = get_mlpackage_path(model_name, padded_h, fp16)
    if not mlpackage_path.exists():
        print(f"No model for size {padded_h}x{padded_w}.")
        print(f"Convert first: uv run python convert.py --model {model_name} --size {max(padded_h, padded_w)}")
        sys.exit(1)

    model, out_key = load_coreml_model(model_name, padded_h, compute_unit, fp16)

    rgb_array = np.array(img, dtype=np.float32) / 255.0

    print("Upscaling...")
    t0 = time.time()
    output = upscale_image_coreml(model, out_key, rgb_array, scale, pre_pad)
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
    )


if __name__ == "__main__":
    main()
