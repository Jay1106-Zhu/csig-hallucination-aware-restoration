"""Run a single-image OneFormer ADE20K semantic-router POC."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image, ImageDraw

from semantic_structure import extract_structure_priors, propose_rois


DEFAULT_MODEL_ID = "shi-labs/oneformer_ade20k_swin_tiny"
ROUTER_LABELS = frozenset(
    {
        "building",
        "wall",
        "road",
        "sidewalk",
        "window",
        "door",
        "signboard",
        "sky",
        "tree",
        "plant",
        "fence",
        "railing",
        "bridge",
    }
)
STRUCTURE_LABELS = frozenset(
    {
        "building",
        "wall",
        "road",
        "sidewalk",
        "window",
        "door",
        "fence",
        "railing",
        "bridge",
    }
)


def _normalise_label(label: object) -> str:
    return str(label).strip().casefold().replace("_", " ")


def target_label_ids(
    id2label: Mapping[int | str, object],
    target_labels: set[str] | frozenset[str] = ROUTER_LABELS,
) -> dict[int, str]:
    targets = {_normalise_label(label) for label in target_labels}
    selected: dict[int, str] = {}
    for raw_id, raw_label in id2label.items():
        aliases = [_normalise_label(alias) for alias in str(raw_label).split(",")]
        for label in aliases:
            if label in targets:
                selected[int(raw_id)] = label
                break
    return dict(sorted(selected.items()))


def select_target_mask(label_map: np.ndarray, selected_ids: set[int] | frozenset[int]) -> np.ndarray:
    labels = np.asarray(label_map)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("label_map must be a 2D integer array")
    if not selected_ids:
        return np.zeros(labels.shape, dtype=bool)
    return np.isin(labels, np.asarray(sorted(selected_ids), dtype=labels.dtype))


def summarize_labels(label_map: np.ndarray, id2label: Mapping[int | str, object]) -> list[dict[str, object]]:
    labels = np.asarray(label_map)
    if labels.ndim != 2:
        raise ValueError("label_map must be a 2D array")
    canonical = target_label_ids(id2label)
    names = {int(key): canonical.get(int(key), _normalise_label(value)) for key, value in id2label.items()}
    counts = Counter(int(value) for value in labels.reshape(-1))
    total = float(labels.size)
    return [
        {
            "id": label_id,
            "label": names.get(label_id, f"class_{label_id}"),
            "pixels": count,
            "ratio": count / total,
        }
        for label_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _palette(label_id: int) -> tuple[int, int, int]:
    return (
        (37 * label_id + 71) % 256,
        (97 * label_id + 113) % 256,
        (157 * label_id + 173) % 256,
    )


def render_overlay(image_rgb: np.ndarray, label_map: np.ndarray, id2label: Mapping[int | str, object]) -> Image.Image:
    image = np.asarray(image_rgb, dtype=np.uint8)
    labels = np.asarray(label_map)
    if image.ndim != 3 or image.shape[2] != 3 or labels.shape != image.shape[:2]:
        raise ValueError("image and label map shapes do not match")
    overlay = image.astype(np.float32)
    names = target_label_ids(id2label)
    for label_id, label in names.items():
        if label not in ROUTER_LABELS:
            continue
        mask = labels == label_id
        if not mask.any():
            continue
        color = np.asarray(_palette(label_id), dtype=np.float32)
        overlay[mask] = overlay[mask] * 0.55 + color * 0.45
    result = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)
    y = 12
    for label_id, label in names.items():
        if label not in ROUTER_LABELS or not np.any(labels == label_id):
            continue
        draw.rectangle((12, y, 30, y + 18), fill=_palette(label_id))
        draw.text((38, y), label, fill=(255, 255, 255))
        y += 22
    return result


def _load_router(model_id: str, device: str):
    import torch
    from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

    processor = OneFormerProcessor.from_pretrained(model_id)
    model = OneFormerForUniversalSegmentation.from_pretrained(model_id).to(device).eval()
    return processor, model


def infer_label_map(image: Image.Image, processor, model, device: str) -> np.ndarray:
    import torch

    inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    target_size = [image.size[1], image.size[0]]
    segmentation = processor.post_process_semantic_segmentation(outputs, target_sizes=[target_size])[0]
    return segmentation.detach().to("cpu").numpy().astype(np.int32)


def run_router(
    input_path: Path | str,
    output_dir: Path | str,
    model_id: str = DEFAULT_MODEL_ID,
    device: str | None = None,
) -> dict[str, object]:
    import torch

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with Image.open(input_path) as image:
        image = image.convert("RGB")
        image_rgb = np.asarray(image, dtype=np.uint8).copy()
    processor, model = _load_router(model_id, resolved_device)
    label_map = infer_label_map(image, processor, model, resolved_device)
    id2label = model.config.id2label
    selected = target_label_ids(id2label)
    structure_selected = target_label_ids(id2label, STRUCTURE_LABELS)
    structure_mask = select_target_mask(label_map, set(structure_selected))
    priors = extract_structure_priors(image_rgb, structure_mask)
    class_names = {label_id: label for label_id, label in structure_selected.items()}
    rois = propose_rois(
        label_map,
        class_names,
        target_labels=STRUCTURE_LABELS,
        min_area=max(2048, label_map.size // 5000),
        padding=max(16, min(label_map.shape) // 100),
    )
    rois = sorted(rois, key=lambda roi: roi.component_area, reverse=True)[:40]

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem
    np.save(destination / f"{stem}_label_map.npy", label_map)
    Image.fromarray((structure_mask.astype(np.uint8) * 255)).save(destination / f"{stem}_structure_mask.png")
    Image.fromarray(priors["edge"]).save(destination / f"{stem}_structure_edge.png")
    Image.fromarray(priors["lines"]).save(destination / f"{stem}_structure_lines.png")
    render_overlay(image_rgb, label_map, id2label).save(destination / f"{stem}_semantic_overlay.png")

    summaries = summarize_labels(label_map, id2label)
    target_summary = [row for row in summaries if row["id"] in selected]
    report = {
        "input": str(Path(input_path)),
        "model_id": model_id,
        "device": resolved_device,
        "image_shape": list(image_rgb.shape),
        "label_map_shape": list(label_map.shape),
        "target_labels": selected,
        "structure_labels": structure_selected,
        "target_summary": target_summary,
        "structure_mask_ratio": float(structure_mask.mean()),
        "structure_edge_ratio": float(np.count_nonzero(priors["edge"]) / priors["edge"].size),
        "structure_line_ratio": float(np.count_nonzero(priors["lines"]) / priors["lines"].size),
        "structure_rois": [roi.__dict__ for roi in rois],
        "success_status": "requires_visual_review",
        "training": False,
        "full_set": False,
    }
    (destination / f"{stem}_router.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    report = run_router(args.input, args.output_dir, args.model_id, args.device)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
