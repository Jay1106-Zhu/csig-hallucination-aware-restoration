"""Score the targeted case set with the investigation-defined metrics."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

from scoring_protocol import candidate_case_ids, proxy_for_family


@dataclass(frozen=True)
class ImageScore:
    case_id: int
    path: Path
    clipiqa: float
    topiq: float

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "path": str(self.path),
            "clipiqa": self.clipiqa,
            "topiq": self.topiq,
        }


@dataclass(frozen=True)
class ScoreSummary:
    family: str
    count: int
    clipiqa_mean: float
    topiq_mean: float
    proxy_score: float | None
    scope: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize_scores(scores: Iterable[ImageScore], family: str) -> ScoreSummary:
    rows = list(scores)
    if not rows:
        raise ValueError("Cannot summarize an empty score set")
    case_ids = {row.case_id for row in rows}
    if len(case_ids) != len(rows):
        raise ValueError("Duplicate case IDs cannot be summarized as independent images")
    full_set = case_ids == set(range(1, 101))

    clipiqa_mean = sum(row.clipiqa for row in rows) / len(rows)
    topiq_mean = sum(row.topiq for row in rows) / len(rows)
    return ScoreSummary(
        family=family,
        count=len(rows),
        clipiqa_mean=clipiqa_mean,
        topiq_mean=topiq_mean,
        proxy_score=proxy_for_family(clipiqa_mean, family) if full_set else None,
        scope="full_set" if full_set else "targeted_diagnostic_subset",
    )


def _metric_value(output: object) -> float:
    import torch

    if not torch.is_tensor(output):
        output = torch.as_tensor(output)
    return float(output.detach().to("cpu").reshape(-1).mean().item())


def _load_image_tensor(path: Path):
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def _build_metrics(device: str | None):
    import pyiqa
    import torch

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return (
        resolved_device,
        pyiqa.create_metric("clipiqa", device=resolved_device),
        pyiqa.create_metric("topiq_nr", device=resolved_device),
    )


def score_paths(paths: Sequence[tuple[int, Path]], device: str | None = None) -> list[ImageScore]:
    import torch

    resolved_device, clip_metric, topiq_metric = _build_metrics(device)
    rows: list[ImageScore] = []
    for case_id, path in paths:
        image = _load_image_tensor(path).to(resolved_device)
        with torch.inference_mode():
            clipiqa = _metric_value(clip_metric(image))
            topiq = _metric_value(topiq_metric(image))
        rows.append(ImageScore(case_id=case_id, path=path, clipiqa=clipiqa, topiq=topiq))
        del image
        if resolved_device.startswith("cuda"):
            torch.cuda.empty_cache()
    return rows


def discover_case_paths(directory: Path | str) -> list[tuple[int, Path]]:
    root = Path(directory)
    case_ids = candidate_case_ids(root)
    paths_by_id: dict[int, Path] = {}
    for path in root.iterdir():
        if not path.is_file():
            continue
        stem = path.stem.casefold()
        if not stem.startswith("case"):
            continue
        suffix = stem[4:]
        if not suffix.isdigit():
            continue
        paths_by_id[int(suffix)] = path
    return [(case_id, paths_by_id[case_id]) for case_id in case_ids]


def write_score_report(
    scores: Sequence[ImageScore],
    output_dir: Path | str,
    output_stem: str,
    family: str,
    source_dir: Path | str,
) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary = summarize_scores(scores, family)
    csv_path = destination / f"{output_stem}.csv"
    json_path = destination / f"{output_stem}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_id", "path", "clipiqa", "topiq"],
        )
        writer.writeheader()
        for score in scores:
            writer.writerow(score.to_dict())

    payload = {
        "source_dir": str(Path(source_dir)),
        "family": family,
        "metrics": {"primary": "CLIPIQA", "secondary": "TOPIQ NR"},
        "summary": summary.to_dict(),
        "scores": [score.to_dict() for score in scores],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", default="scores")
    parser.add_argument("--family", default="hypir")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    paths = discover_case_paths(args.input_dir)
    scores = score_paths(paths, device=args.device)
    csv_path, json_path = write_score_report(
        scores,
        args.output_dir,
        args.output_stem,
        args.family,
        args.input_dir,
    )
    summary = summarize_scores(scores, args.family)
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "summary": summary.to_dict()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
