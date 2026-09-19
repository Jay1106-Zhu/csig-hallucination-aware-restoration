from dataclasses import dataclass
from typing import Iterable, Mapping

import cv2
import numpy as np


@dataclass(frozen=True)
class ROI:
    label: str
    x0: int
    y0: int
    x1: int
    y1: int
    component_area: int


@dataclass(frozen=True)
class CandidateScores:
    h200_overall: float
    expert_overall: float
    h200_structure: float
    expert_structure: float
    expert_artifact_free: bool


@dataclass(frozen=True)
class CandidateDecision:
    winner: str
    reason: str


def _validate_label_map(label_map: np.ndarray) -> np.ndarray:
    array = np.asarray(label_map)
    if array.ndim != 2:
        raise ValueError("label_map must be a 2D array")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("label_map must contain integer labels")
    return array


def resize_label_map_nearest(label_map: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    source = _validate_label_map(label_map)
    if len(target_shape) != 2 or min(target_shape) <= 0:
        raise ValueError("target_shape must contain two positive dimensions")
    target_height, target_width = (int(target_shape[0]), int(target_shape[1]))
    resized = cv2.resize(source, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    return resized.astype(source.dtype, copy=False)


def structure_mask(label_map: np.ndarray, class_ids: Iterable[int]) -> np.ndarray:
    labels = _validate_label_map(label_map)
    ids = np.asarray(list(class_ids), dtype=labels.dtype)
    if ids.size == 0:
        return np.zeros(labels.shape, dtype=bool)
    return np.isin(labels, ids)


def _normalise_gradient(gradient: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.abs(gradient).astype(np.float32)
    selected = values[mask]
    if selected.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    scale = float(np.percentile(selected, 99))
    if scale <= 0:
        return np.zeros(values.shape, dtype=np.uint8)
    normalised = np.clip(values * 255.0 / scale, 0.0, 255.0).astype(np.uint8)
    return np.where(mask, normalised, 0).astype(np.uint8, copy=False)


def extract_structure_priors(image_rgb: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    image = np.asarray(image_rgb)
    structure = np.asarray(mask, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image_rgb must be an RGB uint8 image")
    if structure.shape != image.shape[:2]:
        raise ValueError("mask must match the image spatial shape")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, 50, 150)
    vertical = _normalise_gradient(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), structure)
    horizontal = _normalise_gradient(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3), structure)
    masked_edge = np.where(structure, edge, 0).astype(np.uint8, copy=False)
    lines = np.zeros_like(masked_edge)

    height, width = structure.shape
    threshold = min(64, max(10, min(height, width) // 8))
    minimum_length = min(96, max(5, min(height, width) // 5))
    segments = cv2.HoughLinesP(
        masked_edge,
        rho=1,
        theta=np.pi / 180.0,
        threshold=threshold,
        minLineLength=minimum_length,
        maxLineGap=3,
    )
    if segments is not None:
        for segment in segments.reshape(-1, 4):
            x0, y0, x1, y1 = [int(value) for value in segment]
            cv2.line(lines, (x0, y0), (x1, y1), 255, 1)
    lines = np.where(structure, lines, 0).astype(np.uint8, copy=False)
    return {
        "edge": masked_edge,
        "horizontal": horizontal,
        "vertical": vertical,
        "lines": lines,
    }


def propose_rois(
    label_map: np.ndarray,
    class_names: Mapping[int, str],
    target_labels: Iterable[str],
    min_area: int = 128,
    padding: int = 16,
) -> list[ROI]:
    labels = _validate_label_map(label_map)
    if min_area < 1 or padding < 0:
        raise ValueError("min_area must be positive and padding must be non-negative")
    targets = set(target_labels)
    height, width = labels.shape
    rois: list[ROI] = []

    for class_id, label in class_names.items():
        if label not in targets:
            continue
        binary = (labels == int(class_id)).astype(np.uint8)
        count, components, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for component_index in range(1, count):
            component_area = int(stats[component_index, cv2.CC_STAT_AREA])
            if component_area < min_area:
                continue
            x = int(stats[component_index, cv2.CC_STAT_LEFT])
            y = int(stats[component_index, cv2.CC_STAT_TOP])
            component_width = int(stats[component_index, cv2.CC_STAT_WIDTH])
            component_height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
            rois.append(
                ROI(
                    label=label,
                    x0=max(0, x - padding),
                    y0=max(0, y - padding),
                    x1=min(width, x + component_width + padding),
                    y1=min(height, y + component_height + padding),
                    component_area=component_area,
                )
            )
    return sorted(rois, key=lambda roi: (-roi.component_area, roi.label, roi.y0, roi.x0))


def decide_candidate(scores: CandidateScores, minimum_gain: float = 0.05) -> CandidateDecision:
    values = np.asarray(
        [
            scores.h200_overall,
            scores.expert_overall,
            scores.h200_structure,
            scores.expert_structure,
            minimum_gain,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or minimum_gain < 0:
        raise ValueError("candidate scores and minimum_gain must be finite")
    if not scores.expert_artifact_free:
        return CandidateDecision("H200", "expert rejected: artifact detected")
    if scores.expert_structure < scores.h200_structure:
        return CandidateDecision("H200", "expert rejected: structure score decreased")
    if scores.expert_overall - scores.h200_overall < minimum_gain:
        return CandidateDecision("H200", "expert rejected: insufficient overall gain")
    return CandidateDecision("EXPERT", "expert accepted: clean structural gain")
