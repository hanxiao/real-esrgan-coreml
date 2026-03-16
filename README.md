# Real-ESRGAN CoreML

Real-ESRGAN image upscaling via CoreML on Apple Silicon. Up to 1.6x faster than MLX, 5-10x more energy efficient with ANE.

All 5 official model variants supported. Any input size handled automatically via tiling.

## Quick Start

```bash
uv sync
# Upscale (auto-downloads pre-converted model from GitHub release)
uv run python upscale.py photo.jpg -o photo_4x.png

# Fastest (GPU)
uv run python upscale.py photo.jpg -o photo_4x.png --compute-unit CPU_AND_GPU

# Most energy efficient (ANE, batch inference)
uv run python upscale.py photo.jpg -o photo_4x.png --compute-unit ALL
```

No torch needed at runtime. Models auto-download on first use (~30MB for x4plus).

## Pre-converted Models

Pre-converted CoreML models are hosted as [GitHub release artifacts](https://github.com/hanxiao/real-esrgan-coreml/releases/tag/v1.0.0). The script downloads them automatically.

| Model | batch=1 (GPU) | flexbatch (ANE) | Size |
|-------|--------------|-----------------|------|
| x4plus | [download](https://github.com/hanxiao/real-esrgan-coreml/releases/download/v1.0.0/RealESRGAN_x4plus_522_fp16.zip) | [download](https://github.com/hanxiao/real-esrgan-coreml/releases/download/v1.0.0/RealESRGAN_x4plus_522_fp16_flexbatch.zip) | 30M |
| x2plus | [download](https://github.com/hanxiao/real-esrgan-coreml/releases/download/v1.0.0/RealESRGAN_x2plus_522_fp16.zip) | [download](https://github.com/hanxiao/real-esrgan-coreml/releases/download/v1.0.0/RealESRGAN_x2plus_522_fp16_flexbatch.zip) | 30M |
| anime_6B | [download](https://github.com/hanxiao/real-esrgan-coreml/releases/download/v1.0.0/RealESRGAN_anime_6B_522_fp16.zip) | [download](https://github.com/hanxiao/real-esrgan-coreml/releases/download/v1.0.0/RealESRGAN_anime_6B_522_fp16_flexbatch.zip) | 7.9M |
| animevideo | [download](https://github.com/hanxiao/real-esrgan-coreml/releases/download/v1.0.0/RealESRGAN_animevideo_522_fp16.zip) | [download](https://github.com/hanxiao/real-esrgan-coreml/releases/download/v1.0.0/RealESRGAN_animevideo_522_fp16_flexbatch.zip) | 1.1M |
| general | [download](https://github.com/hanxiao/real-esrgan-coreml/releases/download/v1.0.0/RealESRGAN_general_522_fp16.zip) | [download](https://github.com/hanxiao/real-esrgan-coreml/releases/download/v1.0.0/RealESRGAN_general_522_fp16_flexbatch.zip) | 2.2M |

**batch=1**: fixed input shape, fastest on CPU+GPU mode.
**flexbatch**: flexible batch 1-8, enables batch inference on ANE for 2x speedup.

Models are hardware-independent -- works on any Apple Silicon (M1/M2/M3/M4/A-series). CoreML JIT-compiles for the target chip on first load.

## Performance

### Speed (M3 Ultra, fp16, avg over 256/512/768 inputs)

| Model | CoreML CPU+GPU | CoreML ANE (batch) | MLX |
|-------|---------------|-------------------|-----|
| x4plus | **0.47s** | 2.24s | 0.75s |
| x2plus | **0.12s** | 0.30s | 0.20s |
| anime_6B | **0.15s** | 0.35s | 0.23s |
| animevideo | **0.02s** | 0.06s | 0.02s |
| general | **0.03s** | 0.10s | 0.04s |

### Power and Energy (M3 Ultra, 30s sampling via macmon)

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

ANE power draw is constant (~10W) regardless of model size, while GPU modes scale with compute (68-133W).

### ANE Batch Inference

ANE benefits from batch inference (processing multiple tiles simultaneously):

| x4plus (4 tiles) | Sequential | Batch=4 | Speedup |
|---|---|---|---|
| ANE | 4.53s | **2.24s** | 2.0x |
| GPU | 1.57s | 1.91s | 0.8x (slower) |

GPU is already saturated by single tiles, so batching adds overhead. ANE has independent hardware pipelines that can truly parallelize.

### Summary

| | Speed | Power | Energy | Best For |
|---|---|---|---|---|
| CoreML CPU+GPU | fastest | high (68-127W) | medium | desktop, plugged in |
| CoreML ANE | 2-3x slower | **low (8-11W)** | **lowest** | battery, mobile |
| MLX | middle | high (110-133W) | highest | dynamic sizes |

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

# Choose model
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
2. **Tiling**: large images split into 512x512 tiles with overlap blending
3. **Compute dispatch**: GPU mode uses batch=1 (optimal), ANE mode uses batch inference (2x faster)
4. **Fallback**: if download fails, converts locally (requires torch + coremltools)

CoreML models have fixed input sizes. Tiling handles any image size using a single 522x522 model (512 tile + 10px pre-pad).

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
