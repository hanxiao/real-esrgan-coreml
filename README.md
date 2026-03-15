# Real-ESRGAN CoreML

Real-ESRGAN image upscaling via CoreML on Apple Silicon. 1.6x faster than MLX, no PyTorch at runtime.

## Performance (M3 Ultra)

| Input | Output | Time |
|-------|--------|------|
| 512x512 | 2048x2048 | 0.40s |
| 1024x1024 | 4096x4096 | 1.58s |

CoreML CPU+GPU mode. ANE dispatch is slower for this architecture (dense concat in RRDBNet).

## Install

```bash
uv sync
```

## Quick Start

```bash
# 1. Convert PyTorch weights to CoreML (one-time, needs torch)
uv pip install torch
uv run python convert.py --size 522    # for 512x512 input (512 + 10 pre-pad)
uv run python convert.py --size 1034   # for 1024x1024 input

# 2. Upscale
uv run python upscale.py input.png -o output.png --compute-unit CPU_AND_GPU
```

## Usage

```bash
# upscale with best speed (CPU+GPU, skip ANE)
uv run python upscale.py photo.jpg -o photo_4x.png --compute-unit CPU_AND_GPU

# upscale with ANE (slower for RRDBNet, but try it)
uv run python upscale.py photo.jpg -o photo_4x.png --compute-unit ALL

# convert for custom input size
uv run python convert.py --size 256    # for 246x246 images (256 - 10 pre-pad)

# fp32 precision
uv run python convert.py --size 522 --fp32
uv run python upscale.py photo.jpg -o out.png --fp32
```

## How It Works

1. **convert.py**: one-time PyTorch -> CoreML conversion (traces RRDBNet, converts via coremltools)
2. **upscale.py**: runtime inference via CoreML (no torch needed)
3. CoreML model has fixed input size -- convert for each size you need
4. Pre-padding (10px reflect) handled automatically

## Input Size

CoreML models are fixed-size. The actual model input = your image size + 10 (pre-pad).

| Image Size | Convert Size |
|-----------|-------------|
| 512x512 | `--size 522` |
| 1024x1024 | `--size 1034` |
| 1920x1080 | `--size 1930` (max dim + 10) |

## Compute Units

| Mode | Speed (512) | Notes |
|------|------------|-------|
| CPU_AND_GPU | 0.40s | fastest for RRDBNet |
| ALL | 1.07s | ANE dispatch overhead hurts |
| CPU_ONLY | slow | fallback only |

## vs MLX

See [real-esrgan-mlx](https://github.com/hanxiao/real-esrgan-mlx) for the pure MLX version (0.63s, no coremltools dependency).

CoreML is 1.6x faster but requires fixed input sizes and coremltools. MLX is more flexible.

## License

MIT. Weights from [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) under BSD-3.
