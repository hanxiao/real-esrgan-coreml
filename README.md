# Real-ESRGAN CoreML

Real-ESRGAN image upscaling via CoreML on Apple Silicon. Up to 1.6x faster than MLX, 5-10x more energy efficient with ANE.

All 5 official model variants supported.

## Performance (M3 Ultra, fp16, avg over 256/512/768 inputs, 30s sampling)

### Speed

| Model | CoreML CPU+GPU | CoreML ANE | MLX | Speedup (CoreML vs MLX) |
|-------|---------------|------------|-----|------------------------|
| x4plus | 0.47s | 1.31s | 0.75s | 1.6x |
| x2plus | 0.12s | 0.30s | 0.20s | 1.6x |
| anime_6B | 0.15s | 0.35s | 0.23s | 1.5x |
| animevideo | 0.02s | 0.06s | 0.02s | 1.0x |
| general | 0.03s | 0.10s | 0.04s | 1.2x |

### Power and Energy

| Model | Backend | Power (W) | Energy (J) | Energy vs MLX |
|-------|---------|-----------|------------|---------------|
| x4plus | CPU+GPU | 127 | 60 | 0.6x |
| x4plus | ANE | **10** | **12** | **0.12x** |
| x4plus | MLX | 128 | 99 | 1.0x |
| x2plus | CPU+GPU | 114 | 15 | 0.6x |
| x2plus | ANE | **11** | **3** | **0.12x** |
| x2plus | MLX | 110 | 25 | 1.0x |
| anime_6B | CPU+GPU | 123 | 19 | 0.6x |
| anime_6B | ANE | **10** | **4** | **0.13x** |
| anime_6B | MLX | 131 | 31 | 1.0x |
| animevideo | CPU+GPU | 68 | 2 | 0.5x |
| animevideo | ANE | **8** | **0.5** | **0.18x** |
| animevideo | MLX | 126 | 3 | 1.0x |
| general | CPU+GPU | 86 | 3 | 0.6x |
| general | ANE | **8** | **1** | **0.19x** |
| general | MLX | 133 | 5 | 1.0x |

Power measured with [macmon](https://github.com/vladkens/macmon) (sudoless, 100ms interval, 30s per measurement).

### Summary

- **Fastest**: CoreML CPU+GPU (1.5-1.6x faster than MLX for RRDBNet models)
- **Most energy efficient**: CoreML ANE (8-11W constant, 5-10x less energy than MLX)
- **MLX**: flexible (dynamic input sizes) but slowest and highest power draw

ANE power draw is constant (~10W) regardless of model size, while GPU modes scale with compute (68-133W). For battery-powered devices (iPhone/iPad/MacBook), ANE saves 5-10x energy per upscale.

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

# 2. Upscale (fastest)
uv run python upscale.py input.png -o output.png --model x4plus --compute-unit CPU_AND_GPU

# 2b. Upscale (most energy efficient)
uv run python upscale.py input.png -o output.png --model x4plus --compute-unit ALL
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

## Compute Units

| Mode | Flag | Speed | Power | Best For |
|------|------|-------|-------|----------|
| CPU+GPU | `--compute-unit CPU_AND_GPU` | fastest | 68-127W | desktop, plugged in |
| ALL (ANE) | `--compute-unit ALL` | 2-3x slower | 8-11W | battery, mobile |

## How It Works

1. **convert.py**: one-time PyTorch -> CoreML conversion (traces model, converts via coremltools)
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

## Benchmarking

```bash
# speed + power benchmark (all models, CoreML vs MLX, uses macmon)
uv pip install torch mlx
uv run python benchmark_power.py

# speed-only benchmark
uv run python benchmark_all.py
```

## vs MLX

See [real-esrgan-mlx](https://github.com/hanxiao/real-esrgan-mlx) for the pure MLX version.

| | CoreML CPU+GPU | CoreML ANE | MLX |
|---|---|---|---|
| Speed | fastest | slowest | middle |
| Power | high (68-127W) | **low (8-11W)** | high (110-133W) |
| Energy | middle | **lowest** | highest |
| Input size | fixed (pre-compile) | fixed | dynamic |
| Dependencies | coremltools | coremltools | mlx |

## License

MIT. Weights from [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) under BSD-3.
