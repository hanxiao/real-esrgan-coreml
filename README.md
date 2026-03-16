# Real-ESRGAN CoreML

Real-ESRGAN image upscaling via CoreML on Apple Silicon. Up to 1.6x faster than MLX, 5-10x more energy efficient with ANE.

All 5 official model variants supported. Any input size handled automatically via tiling.

## Quick Start

```bash
uv sync

# Upscale (auto-downloads x4plus model on first run)
uv run python upscale.py photo.jpg -o photo_4x.png

# Fastest (GPU)
uv run python upscale.py photo.jpg -o photo_4x.png --compute-unit CPU_AND_GPU

# Most energy efficient (ANE, batch inference)
uv run python upscale.py photo.jpg -o photo_4x.png --compute-unit ALL
```

No torch needed at runtime. The default model (x4plus) auto-downloads on first use (~30MB).

## Pre-converted Models

Pre-converted CoreML models hosted as individual [GitHub release artifacts](https://github.com/hanxiao/real-esrgan-coreml/releases/tag/v1.0.0). Each model downloads independently on demand.

| Model | Type | Size | Download |
|-------|------|------|----------|
| x4plus | GPU (batch=1) | 30M | [zip](https://github.com/hanxiao/real-esrgan-coreml/releases/download/v1.0.0/RealESRGAN_x4plus_522_fp16.zip) |
| x4plus | ANE (flexbatch) | 30M | [zip](https://github.com/hanxiao/real-esrgan-coreml/releases/download/v1.0.0/RealESRGAN_x4plus_522_fp16_flexbatch.zip) |
| x2plus | GPU | 30M | [zip](https://github.com/hanxiao/real-esrgan-coreml/releases/download/v1.0.0/RealESRGAN_x2plus_522_fp16.zip) |
| x2plus | ANE | 30M | [zip](https://github.com/hanxiao/real-esrgan-coreml/releases/download/v1.0.0/RealESRGAN_x2plus_522_fp16_flexbatch.zip) |
| anime_6B | GPU | 7.9M | [zip](https://github.com/hanxiao/real-esrgan-coreml/releases/download/v1.0.0/RealESRGAN_anime_6B_522_fp16.zip) |
| anime_6B | ANE | 7.9M | [zip](https://github.com/hanxiao/real-esrgan-coreml/releases/download/v1.0.0/RealESRGAN_anime_6B_522_fp16_flexbatch.zip) |
| animevideo | GPU | 1.1M | [zip](https://github.com/hanxiao/real-esrgan-coreml/releases/download/v1.0.0/RealESRGAN_animevideo_522_fp16.zip) |
| animevideo | ANE | 1.1M | [zip](https://github.com/hanxiao/real-esrgan-coreml/releases/download/v1.0.0/RealESRGAN_animevideo_522_fp16_flexbatch.zip) |
| general | GPU | 2.2M | [zip](https://github.com/hanxiao/real-esrgan-coreml/releases/download/v1.0.0/RealESRGAN_general_522_fp16.zip) |
| general | ANE | 2.2M | [zip](https://github.com/hanxiao/real-esrgan-coreml/releases/download/v1.0.0/RealESRGAN_general_522_fp16_flexbatch.zip) |

- **GPU (batch=1)**: fixed input shape, fastest on CPU+GPU
- **ANE (flexbatch)**: flexible batch 1-8, enables batch inference on ANE for 2x speedup
- Models are hardware-independent: works on any Apple Silicon (M1/M2/M3/M4/A-series)

## Performance

Benchmarked on M3 Ultra, fp16, averaged over 256/512/768 input sizes, 30s power sampling via [macmon](https://github.com/vladkens/macmon).

![Benchmark](benchmark.png)

### Speed

| Model | CoreML CPU+GPU | CoreML ANE (batch) | MLX | Speedup vs MLX |
|-------|---------------|-------------------|-----|----------------|
| x4plus | **0.47s** | 2.24s | 0.75s | 1.6x |
| x2plus | **0.12s** | 0.30s | 0.20s | 1.6x |
| anime_6B | **0.15s** | 0.35s | 0.23s | 1.5x |
| animevideo | **0.02s** | 0.06s | 0.02s | 1.0x |
| general | **0.03s** | 0.10s | 0.04s | 1.2x |

### Power and Energy

| Model | Backend | Power (W) | Energy (J) |
|-------|---------|-----------|------------|
| x4plus | CPU+GPU | 127 | 60 |
| x4plus | **ANE** | **10** | **12** |
| x4plus | MLX | 128 | 99 |
| x2plus | CPU+GPU | 114 | 15 |
| x2plus | **ANE** | **11** | **3** |
| x2plus | MLX | 110 | 25 |
| anime_6B | CPU+GPU | 123 | 19 |
| anime_6B | **ANE** | **10** | **4** |
| anime_6B | MLX | 131 | 31 |

ANE power is constant ~10W regardless of model size. GPU/MLX scale with compute at 68-133W.

### ANE Batch Inference

ANE has independent hardware pipelines that can parallelize across tiles:

| x4plus (4 tiles) | Sequential | Batch=4 | Speedup |
|---|---|---|---|
| ANE | 4.53s | **2.24s** | 2.0x |
| GPU | 1.57s | 1.91s | 0.8x (slower) |

GPU is already saturated by single tiles, so batching adds overhead.

### Tile Size

![Tile Benchmark](tile_benchmark.png)

Larger tiles are faster and produce better quality (fewer boundary artifacts). 512 is the optimal balance:

| Tile Size | Tiles | Speed | Max Pixel Diff |
|-----------|-------|-------|----------------|
| 64 | 165 | 3.46s | 0.596 |
| 128 | 42 | 1.80s | 0.518 |
| 256 | 12 | 1.55s | 0.212 |
| **512** | **4** | **1.71s** | **0.075** |

### Summary

| | Speed | Power | Energy | Best For |
|---|---|---|---|---|
| CoreML CPU+GPU | fastest | high (68-127W) | medium | desktop, plugged in |
| CoreML ANE | 2-3x slower | **low (8-11W)** | **lowest** | battery, mobile |
| MLX | middle | high (110-133W) | highest | dynamic input sizes |

## Models

| Name | Architecture | Params | Scale | Use Case |
|------|-------------|--------|-------|----------|
| x4plus | RRDBNet-23 | 64MB | 4x | best quality, general photos |
| x2plus | RRDBNet-23 | 64MB | 2x | 2x upscale |
| anime_6B | RRDBNet-6 | 17MB | 4x | anime images, lighter |
| animevideo | SRVGGNetCompact-16 | 3MB | 4x | anime video, fastest |
| general | SRVGGNetCompact-32 | 6MB | 4x | general purpose, fast |

## Usage

```bash
# Default (x4plus, best quality)
uv run python upscale.py photo.jpg -o photo_4x.png

# Choose model (downloads on first use)
uv run python upscale.py photo.jpg -o photo_4x.png --model anime_6B

# Choose compute unit
uv run python upscale.py photo.jpg -o out.png --compute-unit CPU_AND_GPU  # fastest
uv run python upscale.py photo.jpg -o out.png --compute-unit ALL          # lowest power

# Custom tile size (smaller = less memory, more tiles)
uv run python upscale.py large.jpg -o out.png --tile-size 256

# Convert model manually (requires torch)
uv run python convert.py --model x4plus --size 522
```

## How It Works

1. **Auto-download**: pre-converted CoreML models from GitHub release (no torch needed)
2. **Tiling**: images split into 512x512 tiles with overlap blending
3. **Compute dispatch**: GPU uses sequential batch=1 (optimal), ANE uses batch inference (2x faster)
4. **Fallback**: if download fails, converts locally via torch + coremltools

CoreML models have fixed input sizes. Tiling with a single 522x522 model (512 tile + 10px pre-pad) handles any image size.

## Benchmarking

```bash
# Speed + power benchmark (requires macmon: brew install vladkens/tap/macmon)
uv run python benchmark_power.py

# Speed-only benchmark
uv run python benchmark_all.py
```

## vs MLX

See [real-esrgan-mlx](https://github.com/hanxiao/real-esrgan-mlx) for the pure MLX version.

## License

MIT. Weights from [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) under BSD-3.
