# Autoresearch: Video Pipeline Optimization

## Goal
Optimize the video/GIF upscaling pipeline for maximum throughput (frames per second) with ZERO quality loss.

## Current Baseline
- Sequential per-frame processing: read PNG -> numpy preprocess -> CoreML GPU predict -> numpy postprocess -> save PNG
- x4plus model, 848x456 input, 22 frames: 0.85s/frame
- GPU is idle during CPU I/O operations

## Benchmark Script
Create `benchmark_video.py` that:
1. Uses the test video frames in `/tmp/esrgan_video5/frames/` (22 frames, 848x456)
2. Measures total wall time for processing all frames
3. Compares baseline (current sequential) vs optimized pipeline
4. Verifies output quality: max pixel difference between baseline and optimized must be 0 (bit-identical)
5. Reports: total_time, fps, speedup_ratio

## Optimization Ideas (DO NOT change model inference or tile logic)

### 1. Pipeline Interleaving (threading)
While GPU runs inference on frame N, CPU thread reads frame N+1 and saves frame N-1.
Use `concurrent.futures.ThreadPoolExecutor` or `threading` with queues.

### 2. Skip PNG Intermediate Files  
Instead of: mp4 -> ffmpeg -> PNG files -> Python read -> infer -> PNG files -> ffmpeg -> mp4
Do: mp4 -> ffmpeg pipe (raw RGB stdout) -> infer -> ffmpeg pipe (raw RGB stdin) -> mp4
Use `subprocess.Popen` with pipes, raw video format (`-f rawvideo -pix_fmt rgb24`).

### 3. Combine Both
Pipeline the ffmpeg pipe I/O with inference.

## Constraints
- ZERO quality loss: output must be bit-identical to sequential processing
- Do NOT modify model.py, upscale.py, or convert.py core logic
- Do NOT change tile size, overlap, or pre-pad parameters
- Use CoreML CPU_AND_GPU (not ALL/ANE)
- Python only, no additional compiled dependencies
- Create a new file `video_upscale.py` for the optimized pipeline
- Test with: `uv run python video_upscale.py /path/to/input.mp4 -o /path/to/output.mp4`

## Autoresearch Loop
1. Read `results.tsv` to review history
2. Implement one optimization
3. Run `benchmark_video.py` to measure
4. If speedup > previous best AND quality check passes (max_diff == 0), KEEP
5. If not, DISCARD and try different approach
6. After 5 consecutive discards, try escape strategy

## Results Format (results.tsv)
```
experiment	speedup	fps	total_time	max_diff	status
baseline	1.00	1.18	18.8	0	KEEP
```
