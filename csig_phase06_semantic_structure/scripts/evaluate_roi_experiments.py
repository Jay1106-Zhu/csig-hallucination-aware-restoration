"""Paired full-image and actual-composite ROI scoring; never auto-deploy experts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw

from roi_experiments import PHASE, file_sha256, screening_status


def paired_metrics(baseline, candidate, roi_baseline, roi_candidate):
    global_clip_delta = candidate["clipiqa"] - baseline["clipiqa"]
    global_topiq_delta = candidate["topiq"] - baseline["topiq"]
    roi_clip_delta = roi_candidate["clipiqa"] - roi_baseline["clipiqa"]
    roi_topiq_delta = roi_candidate["topiq"] - roi_baseline["topiq"]
    return {
        "global_clipiqa": candidate["clipiqa"], "global_topiq": candidate["topiq"],
        "global_clipiqa_delta": global_clip_delta, "global_topiq_delta": global_topiq_delta,
        "roi_clipiqa": roi_candidate["clipiqa"], "roi_topiq": roi_candidate["topiq"],
        "roi_clipiqa_delta": roi_clip_delta, "roi_topiq_delta": roi_topiq_delta,
        "proxy_score": None,
        "screening": screening_status(global_clip_delta, global_topiq_delta, roi_clip_delta, roi_topiq_delta),
        "visual_status": "PENDING",
    }


def write_contacts(manifest, root):
    references = root / "references"
    contact_dir = root / "contacts"
    contact_dir.mkdir(exist_ok=True)
    for roi in manifest["config"]["rois"]:
        case_id, name = roi["case_id"], roi["name"]
        panels = [(source, references / f"case{case_id}_{name}_{source}.png") for source in ["original", "h200"]]
        for candidate in manifest["candidates"]:
            if candidate["case_id"] == case_id and candidate["name"] == name:
                panels.append((f"{candidate['model']} <- {candidate['source']}", Path(candidate["output_dir"]) / "blended_roi.png"))
        with Image.open(panels[0][1]) as reference:
            width, height = reference.size
        canvas = Image.new("RGB", (width * 2, (height + 26) * 3), (25, 25, 25))
        draw = ImageDraw.Draw(canvas)
        for index, (title, path) in enumerate(panels):
            left, top = (index % 2) * width, (index // 2) * (height + 26)
            draw.text((left + 8, top + 7), title, fill="white")
            with Image.open(path) as image:
                canvas.paste(image, (left, top + 26))
        canvas.save(contact_dir / f"case{case_id}_{name}.png")


def evaluate(root: Path, device: str):
    sys.path.insert(0, str(PHASE / "dependencies"))
    import pyiqa
    import torch
    from score_targets import _build_metrics, _load_image_tensor, _metric_value, summarize_scores, ImageScore

    manifest = json.loads((root / "candidate_manifest.json").read_text(encoding="utf-8"))
    expected_count = len(manifest["config"]["models"]) * len(manifest["config"]["sources"]) * len(manifest["config"]["rois"])
    if len(manifest["candidates"]) != expected_count:
        raise ValueError("Candidate manifest is incomplete")
    torch.manual_seed(manifest["config"]["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    resolved, clip_metric, topiq_metric = _build_metrics(device)
    cache = {}

    def measure(path):
        key = str(Path(path).resolve())
        if key not in cache:
            image = _load_image_tensor(Path(path)).to(resolved)
            with torch.inference_mode():
                cache[key] = {"clipiqa": _metric_value(clip_metric(image)), "topiq": _metric_value(topiq_metric(image))}
            del image
            if device == "cuda":
                torch.cuda.empty_cache()
        return cache[key]

    baseline_scores, rows = [], []
    for case_id, sources in manifest["baselines"].items():
        for source in sources.values():
            if file_sha256(Path(source["path"])) != source["sha256"]:
                raise ValueError("Baseline changed since candidate generation")
        path = Path(sources["h200"]["path"])
        score = measure(path)
        baseline_scores.append(ImageScore(int(case_id), path, score["clipiqa"], score["topiq"]))
        measure(Path(sources["original"]["path"]))
    for candidate in manifest["candidates"]:
        case_id, name = candidate["case_id"], candidate["name"]
        directory = Path(candidate["output_dir"])
        full_path = directory / f"case{case_id}.png"
        if file_sha256(full_path) != candidate["full_sha256"]:
            raise ValueError("Candidate changed since generation")
        baseline = measure(manifest["baselines"][str(case_id)]["h200"]["path"])
        roi_base = measure(root / "references" / f"case{case_id}_{name}_h200.png")
        measure(root / "references" / f"case{case_id}_{name}_original.png")
        raw_score = measure(directory / "raw_roi.png")
        blended_score = measure(directory / "blended_roi.png")
        full_score = measure(full_path)
        row = {key: candidate[key] for key in ["id", "case_id", "name", "model", "source"]}
        row.update(paired_metrics(baseline, full_score, roi_base, blended_score))
        row.update({"raw_roi_clipiqa": raw_score["clipiqa"], "raw_roi_topiq": raw_score["topiq"], "outside_changed_channels": candidate["outside_changed_channels"], "protected_changed_channels": candidate["protected_changed_channels"]})
        rows.append(row)
        print(f"{row['id']}: global {row['global_clipiqa_delta']:+.6f}/{row['global_topiq_delta']:+.6f}; blended ROI {row['roi_clipiqa_delta']:+.6f}/{row['roi_topiq_delta']:+.6f}; {row['screening']}", flush=True)
    baseline_path = Path(manifest["baselines"]["42"]["h200"]["path"])
    original_score = measure(baseline_path).copy()
    del cache[str(baseline_path.resolve())]
    repeated_score = measure(baseline_path)
    repeat_delta = {key: repeated_score[key] - original_score[key] for key in original_score}
    checkpoints = Path(torch.hub.get_dir())
    scoring_weights = [checkpoints / "clip" / "RN50.pt", checkpoints / "pyiqa" / "cfanet_nr_koniq_res50-9a73138b.pth"]
    report = {
        "scope": "targeted_diagnostic_subset", "proxy_score": None,
        "primary": "CLIPIQA", "secondary": "TOPIQ NR", "preprocessing": "PIL RGB -> float32 [0,1], native image/ROI; pyiqa default internal preprocessing; no external resize; no GT",
        "pyiqa": pyiqa.__version__, "torch": torch.__version__,
        "metric_weights": [{"path": str(path), "sha256": file_sha256(path)} for path in scoring_weights],
        "baseline_summary": summarize_scores(baseline_scores, "hypir").to_dict(),
        "baseline_case42_repeat_delta": repeat_delta, "scores": cache, "comparisons": rows,
        "script_sha256": file_sha256(Path(__file__)), "candidate_manifest_sha256": file_sha256(root / "candidate_manifest.json"),
        "warning": "Screening is not acceptance. Visual/text authenticity review required; original is not GT. No leaderboard/full-set conclusions.",
    }
    (root / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (root / "comparisons.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_contacts(manifest, root)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=PHASE / "outputs" / "round02" / "experiments")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evaluate(args.experiment_dir, args.device)


if __name__ == "__main__":
    main()
