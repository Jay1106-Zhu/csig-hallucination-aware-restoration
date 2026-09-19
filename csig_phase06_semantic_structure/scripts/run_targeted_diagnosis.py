import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from targeted_diagnosis import TARGET_CASES, diagnose_path, target_case_paths


def render_overlay(image_rgb: np.ndarray, diagnosis) -> Image.Image:
    image = Image.fromarray(image_rgb).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    pixels = np.asarray(overlay).copy()
    pixels[diagnosis.halo_candidate] = (255, 80, 40, 90)
    pixels[diagnosis.highlight_core] = (40, 220, 255, 150)
    overlay = Image.fromarray(pixels, mode="RGBA")
    result = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(result)
    for roi in diagnosis.highlight_rois:
        draw.rectangle((roi.x0, roi.y0, roi.x1 - 1, roi.y1 - 1), outline=(255, 220, 40, 255), width=4)
        draw.text((roi.x0 + 4, roi.y0 + 4), roi.label, fill=(255, 220, 40, 255))
    return result.convert("RGB")


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=project_root / "csig_dataset" / "测试集",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "new_formal" / "csig_phase06_semantic_structure" / "outputs" / "targeted_diagnosis",
    )
    parser.add_argument("--cases", nargs="+", type=int, default=list(TARGET_CASES))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for case_id, path in zip(args.cases, target_case_paths(args.dataset_dir, args.cases)):
        image, diagnosis = diagnose_path(path, case_id)
        case_dir = args.output_dir / f"case{case_id}"
        case_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(case_dir / "original.png")
        Image.fromarray(diagnosis.highlight_core.astype(np.uint8) * 255).save(case_dir / "highlight_core.png")
        Image.fromarray(diagnosis.halo_candidate.astype(np.uint8) * 255).save(case_dir / "halo_candidate.png")
        Image.fromarray(diagnosis.structure_edge).save(case_dir / "structure_edge.png")
        Image.fromarray(diagnosis.structure_lines).save(case_dir / "structure_lines.png")
        render_overlay(image, diagnosis).save(case_dir / "diagnosis_overlay.png")
        records.append(diagnosis.to_dict())

    (args.output_dir / "diagnosis.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
