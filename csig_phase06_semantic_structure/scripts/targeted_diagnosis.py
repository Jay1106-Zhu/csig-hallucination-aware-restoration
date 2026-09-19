from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from semantic_structure import ROI, extract_structure_priors, propose_rois


TARGET_CASES = (37, 38, 39, 41, 42, 65)


@dataclass
class CaseDiagnosis:
    case_id: int
    image_shape: tuple[int, int, int]
    highlight_core_area_ratio: float
    halo_candidate_area_ratio: float
    edge_area_ratio: float
    line_area_ratio: float
    highlight_core: np.ndarray
    halo_candidate: np.ndarray
    structure_edge: np.ndarray
    structure_lines: np.ndarray
    highlight_rois: tuple[ROI, ...]

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "image_shape": list(self.image_shape),
            "highlight_core_area_ratio": self.highlight_core_area_ratio,
            "halo_candidate_area_ratio": self.halo_candidate_area_ratio,
            "edge_area_ratio": self.edge_area_ratio,
            "line_area_ratio": self.line_area_ratio,
            "highlight_rois": [
                {
                    "label": roi.label,
                    "x0": roi.x0,
                    "y0": roi.y0,
                    "x1": roi.x1,
                    "y1": roi.y1,
                    "component_area": roi.component_area,
                }
                for roi in self.highlight_rois
            ],
        }


def target_case_paths(dataset_dir: Path, case_ids: Iterable[int] = TARGET_CASES) -> list[Path]:
    root = Path(dataset_dir)
    return [root / f"case{case_id}.jpg" for case_id in case_ids]


def load_rgb_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _threshold(gray: np.ndarray, percentile: float, floor: int) -> int:
    return max(floor, int(np.percentile(gray, percentile)))


def _build_highlight_core(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    threshold = _threshold(gray, 99.0, 190)
    saturated = hsv[:, :, 1] >= 45
    return (gray >= threshold) & (saturated | (gray >= 235))


def _build_halo_candidate(image_rgb: np.ndarray, highlight_core: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated = cv2.dilate(highlight_core.astype(np.uint8), kernel) > 0
    ambient_threshold = _threshold(gray, 70.0, 24)
    ambient = gray >= ambient_threshold
    return dilated & ~highlight_core & ambient


def _highlight_rois(highlight_core: np.ndarray, image_shape: tuple[int, int, int]) -> tuple[ROI, ...]:
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    join_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    joined = cv2.morphologyEx(highlight_core.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel)
    joined = cv2.dilate(joined, join_kernel)
    height, width = image_shape[:2]
    minimum_area = max(32, (height * width) // 100000)
    padding = max(8, min(height, width) // 100)
    rois = propose_rois(
        joined.astype(np.int32),
        {1: "highlight_signage"},
        target_labels={"highlight_signage"},
        min_area=minimum_area,
        padding=padding,
    )
    return tuple(rois)


def diagnose_image(image_rgb: np.ndarray, case_id: int) -> CaseDiagnosis:
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image_rgb must be an RGB uint8 image")

    highlight_core = _build_highlight_core(image)
    halo_candidate = _build_halo_candidate(image, highlight_core)
    structure_mask = ~halo_candidate
    priors = extract_structure_priors(image, structure_mask)
    pixel_count = float(image.shape[0] * image.shape[1])
    return CaseDiagnosis(
        case_id=int(case_id),
        image_shape=tuple(int(value) for value in image.shape),
        highlight_core_area_ratio=float(np.count_nonzero(highlight_core) / pixel_count),
        halo_candidate_area_ratio=float(np.count_nonzero(halo_candidate) / pixel_count),
        edge_area_ratio=float(np.count_nonzero(priors["edge"]) / pixel_count),
        line_area_ratio=float(np.count_nonzero(priors["lines"]) / pixel_count),
        highlight_core=highlight_core,
        halo_candidate=halo_candidate,
        structure_edge=priors["edge"],
        structure_lines=priors["lines"],
        highlight_rois=_highlight_rois(highlight_core, tuple(int(value) for value in image.shape)),
    )


def diagnose_path(path: Path, case_id: int) -> tuple[np.ndarray, CaseDiagnosis]:
    image = load_rgb_image(path)
    return image, diagnose_image(image, case_id)


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8)).save(path)
