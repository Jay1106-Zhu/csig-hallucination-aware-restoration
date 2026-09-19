"""Native-resolution, fixed-ROI pretrained expert screening on an H200 base."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from run_diffir_control import clip_box, feather_alpha


PHASE = Path(__file__).resolve().parents[1]
MODEL_SPECS = {
    "fftformer_realblur_j": ("FFTformer", "fftformer_arch.py", "fftformer", "options/train/Realblur.yml", 32),
    "restormer_real_denoising": ("Restormer", "restormer_arch.py", "Restormer", "Denoising/Options/RealDenoising_Restormer.yml", 8),
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def context_box(box, image_shape, context):
    if context < 0:
        raise ValueError("Context must be non-negative")
    left, top, right, bottom = box
    height, width = image_shape[:2]
    return clip_box((left - context, top - context, right + context, bottom + context), width, height)


def composite_candidate(base, candidate, box, allowed, feather):
    left, top, right, bottom = clip_box(box, base.shape[1], base.shape[0])
    shape = (bottom - top, right - left)
    if candidate.shape != (*shape, 3) or allowed.shape != shape:
        raise ValueError("Candidate and mask must match the exact ROI shape")
    if base.dtype != np.uint8 or candidate.dtype != np.uint8 or allowed.dtype != bool:
        raise ValueError("Images must be uint8 and mask must be boolean")
    alpha = feather_alpha(shape, feather) * allowed.astype(np.float32)
    if feather and not allowed.all():
        distance = cv2.distanceTransform(allowed.astype(np.uint8), cv2.DIST_L2, 3)
        alpha *= np.clip(distance / feather, 0, 1)
    output = base.copy()
    baseline_roi = base[top:bottom, left:right].astype(np.float32)
    blended = baseline_roi + alpha[..., None] * (candidate.astype(np.float32) - baseline_roi)
    output[top:bottom, left:right] = np.rint(np.clip(blended, 0, 255)).astype(np.uint8)
    return output


def screening_status(global_clip_delta, global_topiq_delta, roi_clip_delta, roi_topiq_delta):
    values = (global_clip_delta, global_topiq_delta, roi_clip_delta, roi_topiq_delta)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("All metric deltas must be finite")
    if global_clip_delta <= 0:
        return "KEEP_H200_PRIMARY_REGRESSION"
    if roi_clip_delta <= 0:
        return "KEEP_H200_LOCAL_REGRESSION"
    if global_topiq_delta < 0 or roi_topiq_delta < 0:
        return "REVIEW_SECONDARY_TRADEOFF"
    return "VISUAL_REVIEW_REQUIRED"


def load_model(name, device):
    import torch
    import yaml

    directory, filename, class_name, config_name, multiple = MODEL_SPECS[name]
    repository = PHASE / "third_party" / directory
    source = repository / "basicsr" / "models" / "archs" / filename
    spec = importlib.util.spec_from_file_location(f"phase06_{name}", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    options = yaml.safe_load((repository / config_name).read_text(encoding="utf-8"))["network_g"]
    options.pop("type")
    model = getattr(module, class_name)(**options)
    weights = PHASE / "models" / f"{name}.pth"
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint.get("params", checkpoint), strict=True)
    model.to(device).eval()
    return model, multiple, {"weights": str(weights), "weights_sha256": file_sha256(weights), "source_sha256": file_sha256(source), "options": options}


def infer_native(model, rgb, device, multiple):
    import torch
    import torch.nn.functional as functional

    height, width = rgb.shape[:2]
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32) / 255.0
    pad_height, pad_width = (-height) % multiple, (-width) % multiple
    if pad_height or pad_width:
        tensor = functional.pad(tensor, (0, pad_width, 0, pad_height), mode="reflect")
    with torch.inference_mode():
        restored = model(tensor)
    if restored.shape != tensor.shape or not torch.isfinite(restored).all():
        raise ValueError("Expert output must be finite and same-resolution")
    cropped = restored[0, :, :height, :width]
    clipped_fraction = float(((cropped < 0) | (cropped > 1)).float().mean().item())
    rgb_output = cropped.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return np.rint(rgb_output * 255).astype(np.uint8), clipped_fraction


def run(config_path: Path, output_dir: Path, device: str):
    import torch
    from transformers import OneFormerConfig
    from oneformer_router import STRUCTURE_LABELS, select_target_mask, target_label_ids

    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "candidate_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Use a new output directory to preserve existing experiments")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    router_config = OneFormerConfig.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny", local_files_only=True)
    ids = set(target_label_ids(router_config.id2label, STRUCTURE_LABELS))
    sign_ids = set(target_label_ids(router_config.id2label, {"signboard"}))
    records, images, masks, baselines = [], {}, {}, {}
    for case_id in config["cases"]:
        baselines[str(case_id)] = {}
        for source, path in {
            "original": PHASE / "data" / "input" / f"case{case_id}.jpg",
            "h200": PHASE / "outputs" / "hypir_h200" / "result" / f"case{case_id}.png",
        }.items():
            with Image.open(path) as image:
                images[case_id, source] = np.array(image.convert("RGB"))
            baselines[str(case_id)][source] = {"path": str(path), "sha256": file_sha256(path)}
        if images[case_id, "original"].shape != images[case_id, "h200"].shape:
            raise ValueError("Original and H200 dimensions differ")
        labels = np.load(PHASE / "outputs" / "round02" / "router" / f"case{case_id}" / f"case{case_id}_label_map.npy")
        protected = cv2.dilate(select_target_mask(labels, sign_ids).astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
        masks[case_id] = select_target_mask(labels, ids) & ~protected

    reference_dir = output_dir / "references"
    reference_dir.mkdir(exist_ok=True)
    for roi in config["rois"]:
        case_id = roi["case_id"]
        for source in config["sources"]:
            image = Image.fromarray(images[case_id, source])
            image.crop(tuple(roi["box"])).save(reference_dir / f"case{case_id}_{roi['name']}_{source}.png")
        preview = Image.fromarray(images[case_id, "original"]).copy()
        ImageDraw.Draw(preview).rectangle(tuple(roi["box"]), outline="yellow", width=8)
        preview.thumbnail((1024, 1024))
        preview.save(reference_dir / f"case{case_id}_{roi['name']}_location.png")

    manifest = {"config": config, "config_sha256": file_sha256(config_path), "script_sha256": file_sha256(Path(__file__)), "baselines": baselines, "environment": {"torch": torch.__version__, "device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None}, "candidates": records}
    for model_name in config["models"]:
        model, multiple, model_metadata = load_model(model_name, device)
        for roi in config["rois"]:
            case_id = roi["case_id"]
            left, top, right, bottom = roi["box"]
            base = images[case_id, "h200"]
            box = context_box(roi["box"], base.shape, config["context"])
            context_left, context_top, context_right, context_bottom = box
            allowed = masks[case_id][top:bottom, left:right] if roi["branch"] == "structure" else np.ones((bottom - top, right - left), dtype=bool)
            for source in config["sources"]:
                identifier = f"case{case_id}_{roi['name']}_{model_name}_from_{source}"
                candidate_dir = output_dir / "candidates" / identifier
                candidate_dir.mkdir(parents=True, exist_ok=True)
                source_image = images[case_id, source]
                crop = source_image[context_top:context_bottom, context_left:context_right]
                torch.manual_seed(config["seed"])
                started = time.perf_counter()
                inferred, clipped_fraction = infer_native(model, crop, device, multiple)
                candidate = inferred[top - context_top:bottom - context_top, left - context_left:right - context_left]
                full = composite_candidate(base, candidate, roi["box"], allowed, config["feather"])
                outside_changes = sum(int(np.count_nonzero(first != second)) for first, second in [
                    (full[:top], base[:top]), (full[bottom:], base[bottom:]),
                    (full[top:bottom, :left], base[top:bottom, :left]), (full[top:bottom, right:], base[top:bottom, right:]),
                ])
                protected_changes = int(np.count_nonzero(full[top:bottom, left:right][~allowed] != base[top:bottom, left:right][~allowed]))
                if outside_changes or protected_changes:
                    raise AssertionError("ROI exterior or protected pixels changed")
                Image.fromarray(candidate).save(candidate_dir / "raw_roi.png")
                Image.fromarray(full[top:bottom, left:right]).save(candidate_dir / "blended_roi.png")
                Image.fromarray(allowed.astype(np.uint8) * 255).save(candidate_dir / "allowed.png")
                full_path = candidate_dir / f"case{case_id}.png"
                Image.fromarray(full).save(full_path)
                record = {"id": identifier, **roi, "model": model_name, "source": source, "context_box": list(box), "resize": False, "context_padding_multiple": multiple, "output_dir": str(candidate_dir), "full_sha256": file_sha256(full_path), "model_metadata": model_metadata, "seconds": time.perf_counter() - started, "clipped_fraction": clipped_fraction, "allowed_ratio": float(allowed.mean()), "outside_changed_channels": outside_changes, "protected_changed_channels": protected_changes}
                records.append(record)
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                print(identifier, f"{record['seconds']:.2f}s", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PHASE / "configs" / "round02.json")
    parser.add_argument("--output-dir", type=Path, default=PHASE / "outputs" / "round02" / "experiments")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.config, args.output_dir, args.device)


if __name__ == "__main__":
    main()
