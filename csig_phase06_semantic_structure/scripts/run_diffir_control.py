"""Run a single-ROI DiffIR motion-deblur control candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
from PIL import Image


DIFFIR_CONFIG = {
    "type": "DiffIRS2",
    "n_encoder_res": 5,
    "inp_channels": 3,
    "out_channels": 3,
    "dim": 48,
    "num_blocks": [3, 5, 6, 6],
    "num_refinement_blocks": 4,
    "heads": [1, 2, 4, 8],
    "ffn_expansion_factor": 2,
    "bias": False,
    "LayerNorm_type": "WithBias",
    "n_denoise_res": 1,
    "linear_start": 0.1,
    "linear_end": 0.99,
    "timesteps": 4,
}


def clip_box(box: Sequence[int], width: int, height: int) -> tuple[int, int, int, int]:
    if len(box) != 4:
        raise ValueError("box must contain x0, y0, x1, y1")
    x0, y0, x1, y1 = (int(value) for value in box)
    x0 = max(0, min(width, x0))
    y0 = max(0, min(height, y0))
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("box must have positive area after clipping")
    return x0, y0, x1, y1


def feather_alpha(shape: tuple[int, int], feather: int) -> np.ndarray:
    height, width = (int(value) for value in shape)
    if height < 1 or width < 1 or feather < 0:
        raise ValueError("shape must be positive and feather must be non-negative")
    if feather == 0:
        return np.ones((height, width), dtype=np.float32)
    y, x = np.ogrid[:height, :width]
    distance = np.minimum(
        np.minimum(y, x),
        np.minimum(height - 1 - y, width - 1 - x),
    ).astype(np.float32)
    return np.clip(distance / float(feather), 0.0, 1.0).astype(np.float32)


def blend_roi(
    base_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
    box: Sequence[int],
    feather: int = 32,
) -> np.ndarray:
    base = np.asarray(base_rgb)
    candidate = np.asarray(candidate_rgb)
    if base.ndim != 3 or base.shape[2] != 3 or base.dtype != np.uint8:
        raise ValueError("base_rgb must be an RGB uint8 image")
    if candidate.ndim != 3 or candidate.shape[2] != 3 or candidate.dtype != np.uint8:
        raise ValueError("candidate_rgb must be an RGB uint8 image")
    x0, y0, x1, y1 = clip_box(box, base.shape[1], base.shape[0])
    if candidate.shape[:2] != (y1 - y0, x1 - x0):
        raise ValueError("candidate_rgb shape must match the ROI")
    result = base.copy().astype(np.float32)
    alpha = feather_alpha(candidate.shape[:2], feather)[..., None]
    base_crop = result[y0:y1, x0:x1]
    result[y0:y1, x0:x1] = base_crop * (1.0 - alpha) + candidate.astype(np.float32) * alpha
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def _load_diffir(device: str):
    import torch
    import DiffIR.archs
    from basicsr.archs import build_network

    model = build_network(DIFFIR_CONFIG).to(device).eval()
    root = Path(__file__).resolve().parents[3]
    weights_path = root / "model_cache" / "diffir_motion_deblurring" / "Deblurring-DiffIRS2.pth"
    checkpoint = torch.load(weights_path, map_location="cpu")
    state = checkpoint.get("params_ema", checkpoint.get("params"))
    if state is None:
        raise ValueError(f"No DiffIR state found in {weights_path}")
    model.load_state_dict(state, strict=True)
    return model, weights_path


def _infer_crop(model, crop_rgb: np.ndarray, device: str) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    image = torch.from_numpy(crop_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    image = image.to(device)
    height, width = image.shape[-2:]
    pad_height = (8 - height % 8) % 8
    pad_width = (8 - width % 8) % 8
    if pad_height or pad_width:
        image = functional.pad(image, (0, pad_width, 0, pad_height), mode="reflect")
    with torch.inference_mode():
        output = model(image)
    output = output[..., :height, :width].clamp(0, 1)
    array = output.squeeze(0).permute(1, 2, 0).to("cpu").numpy()
    return np.rint(array * 255.0).astype(np.uint8)


def run_diffir_control(
    input_path: Path | str,
    output_path: Path | str,
    box: Sequence[int],
    feather: int = 32,
    device: str | None = None,
) -> dict[str, object]:
    import torch

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with Image.open(input_path) as image:
        base = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    clipped_box = clip_box(box, base.shape[1], base.shape[0])
    x0, y0, x1, y1 = clipped_box
    base_crop = base[y0:y1, x0:x1]
    model, weights_path = _load_diffir(resolved_device)
    candidate_crop = _infer_crop(model, base_crop, resolved_device)
    blended = blend_roi(base, candidate_crop, clipped_box, feather=feather)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(blended).save(destination)
    Image.fromarray(base_crop).save(destination.with_name(f"{destination.stem}_base_roi.png"))
    Image.fromarray(candidate_crop).save(destination.with_name(f"{destination.stem}_diffir_roi.png"))
    manifest = {
        "candidate": "diffir_motion_deblur_control",
        "input": str(Path(input_path)),
        "output": str(destination),
        "roi": list(clipped_box),
        "feather": feather,
        "device": resolved_device,
        "weights": str(weights_path),
        "training": False,
        "full_set": False,
    }
    destination.with_name(f"{destination.stem}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    diffir_root = root / "third_party" / "DiffIR" / "DiffIR-demotionblur"
    sys.path.insert(0, str(diffir_root))
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roi", nargs=4, type=int, required=True)
    parser.add_argument("--feather", type=int, default=32)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    manifest = run_diffir_control(args.input, args.output, args.roi, args.feather, args.device)
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
