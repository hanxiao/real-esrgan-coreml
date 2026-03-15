# Real-ESRGAN CoreML

Real-ESRGAN image upscaling via CoreML on Apple Silicon. Up to 1.7x faster than MLX, no PyTorch at runtime.

All 5 official model variants supported.

## Performance (M3 Ultra, fp16, 512x512 input)

| Model | Scale | Output | CoreML (s) | MLX (s) | Speedup | Max Diff |
|-------|-------|--------|-----------|---------|---------|----------|
| x4plus | 4x | 2048x2048 | 0.396 | 0.631 | 1.6x | 0.006 |
| x2plus | 2x | 1024x1024 | 0.104 | 0.179 | 1.7x | 0.004 |
| anime_6B | 4x | 2048x2048 | 0.131 | 0.194 | 1.5x | 0.005 |
| animevideo | 4x | 2048x2048 | 0.018 | 0.018 | 1.0x | 0.007 |
| general | 4x | 2048x2048 | 0.028 | 0.033 | 1.2x | 0.006 |

CoreML CPU+GPU mode. Max diff is vs MLX output (both fp16), in [0, 1] range.

## Models

| Name | Architecture | Params | Use Case |
|------|-------------|--------|----------|
| x4plus | RRDBNet-23 | 64MB | best quality, general photos |
| x2plus | RRDBNet-23 | 64MB | 2x upscale |
| anime_6B | RRDBNet-6 | 17MB | anime images, lighter |
| animevideo | SRVGGNetCompact-16 | 3MB | anime video, fastest |
| general | SRVGGNetCompact-32 | 6MB | general purpose, fast |

## Install

```bash
uv sync
```

## Quick Start

```bash
# 1. Convert model to CoreML (one-time, needs torch)
uv pip install torch
uv run python convert.py --model x4plus --size 522    # for 512x512 input (512 + 10 pre-pad)

# 2. Upscale
uv run python upscale.py input.png -o output.png --model x4plus --compute-unit CPU_AND_GPU
```

## Usage

```bash
# x4plus (default, best quality)
uv run python convert.py --model x4plus --size 522
uv run python upscale.py photo.jpg -o photo_4x.png --model x4plus

# x2plus (2x upscale)
uv run python convert.py --model x2plus --size 522
uv run python upscale.py photo.jpg -o photo_2x.png --model x2plus

# anime_6B (lighter, for anime)
uv run python convert.py --model anime_6B --size 522
uv run python upscale.py anime.png -o anime_4x.png --model anime_6B

# animevideo (fastest, for anime video frames)
uv run python convert.py --model animevideo --size 522
uv run python upscale.py frame.png -o frame_4x.png --model animevideo

# general (fast, general purpose)
uv run python convert.py --model general --size 522
uv run python upscale.py photo.jpg -o photo_4x.png --model general

# convert for custom input size
uv run python convert.py --model x4plus --size 1034   # for 1024x1024 input

# fp32 precision
uv run python convert.py --model x4plus --size 522 --fp32
uv run python upscale.py photo.jpg -o out.png --model x4plus --fp32
```

## How It Works

1. **convert.py**: one-time PyTorch -> CoreML conversion (traces model, converts via coremltools)
2. **upscale.py**: runtime inference via CoreML (no torch needed)
3. CoreML model has fixed input size - convert for each size you need
4. Pre-padding (10px reflect) handled automatically

## Input Size

CoreML models are fixed-size. The actual model input = your image size + 10 (pre-pad).

| Image Size | Convert Size |
|-----------|-------------|
| 512x512 | `--size 522` |
| 1024x1024 | `--size 1034` |
| 1920x1080 | `--size 1930` (max dim + 10) |

## Benchmarking

```bash
# benchmark all 5 models (CoreML vs MLX)
uv pip install torch mlx
uv run python benchmark_all.py
```

## vs MLX

CoreML is 1.5-1.7x faster for RRDBNet models (x4plus, x2plus, anime_6B). SRVGGNetCompact models (animevideo, general) are roughly equivalent since they're already very fast.

MLX is more flexible (dynamic input sizes, no pre-compilation needed).

## License

MIT. Weights from [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) under BSD-3.
