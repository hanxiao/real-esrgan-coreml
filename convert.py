"""Convert PyTorch RealESRGAN RRDBNet to CoreML .mlpackage format.

Requires: torch, coremltools (one-time conversion, not needed at runtime).
Usage: uv run python coreml_convert.py [--size 512] [--fp16]
"""

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

PTH_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
WEIGHTS_DIR = Path(__file__).parent / "weights"


def download_pth() -> Path:
    """Download PyTorch .pth weights if not cached."""
    pth_path = WEIGHTS_DIR / "RealESRGAN_x4plus.pth"
    if pth_path.exists():
        print(f"Found cached {pth_path}")
        return pth_path
    WEIGHTS_DIR.mkdir(exist_ok=True)
    print(f"Downloading {PTH_URL} ...")

    def progress(count, block_size, total_size):
        pct = count * block_size * 100 // total_size
        print(f"\r  {pct}%", end="", flush=True)

    urlretrieve(PTH_URL, pth_path, reporthook=progress)
    print()
    return pth_path


def build_torch_rrdb(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
    """Build PyTorch RRDBNet without importing basicsr (inline definition)."""
    import torch
    import torch.nn as tnn

    class _RDB(tnn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = tnn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = tnn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv3 = tnn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = tnn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = tnn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
            self.lrelu = tnn.LeakyReLU(0.2, True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class _RRDB(tnn.Module):
        def __init__(self):
            super().__init__()
            self.rdb1 = _RDB()
            self.rdb2 = _RDB()
            self.rdb3 = _RDB()

        def forward(self, x):
            out = self.rdb1(x)
            out = self.rdb2(out)
            out = self.rdb3(out)
            return out * 0.2 + x

    class _RRDBNet(tnn.Module):
        def __init__(self):
            super().__init__()
            self.scale = scale
            self.conv_first = tnn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
            self.body = tnn.Sequential(*[_RRDB() for _ in range(num_block)])
            self.conv_body = tnn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = tnn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = tnn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = tnn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = tnn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = tnn.LeakyReLU(0.2, True)

        def forward(self, x):
            feat = self.conv_first(x)
            body_feat = self.conv_body(self.body(feat))
            feat = feat + body_feat
            feat = self.lrelu(self.conv_up1(
                torch.nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
            feat = self.lrelu(self.conv_up2(
                torch.nn.functional.interpolate(feat, scale_factor=2, mode='nearest')))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    return _RRDBNet()


def convert(input_size=512, use_fp16=True):
    """Convert RRDBNet to CoreML .mlpackage."""
    import torch
    import coremltools as ct

    pth_path = download_pth()

    # Load PyTorch model
    print("Building PyTorch model...")
    model = build_torch_rrdb()
    checkpoint = torch.load(str(pth_path), map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Trace
    print(f"Tracing with input shape [1, 3, {input_size}, {input_size}]...")
    example_input = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)

    # Convert to CoreML
    precision = ct.precision.FLOAT16 if use_fp16 else ct.precision.FLOAT32
    dtype_label = "fp16" if use_fp16 else "fp32"
    print(f"Converting to CoreML ({dtype_label})...")

    ct_model = ct.convert(
        traced,
        inputs=[ct.TensorType(shape=(1, 3, input_size, input_size), name="input")],
        compute_precision=precision,
        minimum_deployment_target=ct.target.macOS15,
    )

    # Save
    WEIGHTS_DIR.mkdir(exist_ok=True)
    out_name = f"RealESRGAN_x4plus_{input_size}_{dtype_label}.mlpackage"
    out_path = WEIGHTS_DIR / out_name
    ct_model.save(str(out_path))
    print(f"Saved: {out_path}")

    # Quick verification: run one prediction and check output shape
    print("Verifying output shape...")
    test_input = np.random.randn(1, 3, input_size, input_size).astype(np.float32)
    result = ct_model.predict({"input": test_input})
    out_key = list(result.keys())[0]
    out_shape = result[out_key].shape
    expected = (1, 3, input_size * 4, input_size * 4)
    assert out_shape == expected, f"Shape mismatch: {out_shape} != {expected}"
    print(f"Output shape OK: {out_shape}")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Convert RealESRGAN to CoreML")
    parser.add_argument("--size", type=int, default=512,
                        help="Input spatial size for fixed-shape model (default: 512)")
    parser.add_argument("--fp32", action="store_true", help="Use float32 instead of float16")
    args = parser.parse_args()
    convert(input_size=args.size, use_fp16=not args.fp32)


if __name__ == "__main__":
    main()
