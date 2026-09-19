"""Scoring contracts for the Phase06 targeted candidate comparison."""

from __future__ import annotations

from pathlib import Path
import re


HYPIR_PROXY_SLOPE = 3.3091583337497186
HYPIR_PROXY_INTERCEPT = 1.3031216319731522

_CASE_PATTERN = re.compile(r"^case(\d+)\.(?:png|jpg|jpeg)$", re.IGNORECASE)


def apply_hypir_proxy(clipiqa_mean: float) -> float:
    """Map the mean CLIPIQA value to the fixed HYPIR diagnostic proxy."""

    return HYPIR_PROXY_SLOPE * float(clipiqa_mean) + HYPIR_PROXY_INTERCEPT


def proxy_for_family(clipiqa_mean: float, family: str) -> float | None:
    """Apply the proxy only to the HYPIR family, never to cross-model results."""

    if family.casefold() != "hypir":
        return None
    return apply_hypir_proxy(clipiqa_mean)


def _case_ids_in_directory(directory: Path) -> tuple[int, ...]:
    if not directory.is_dir():
        raise ValueError(f"Candidate directory does not exist: {directory}")

    case_ids = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        match = _CASE_PATTERN.match(path.name)
        if match is not None:
            case_ids.append(int(match.group(1)))
    return tuple(sorted(set(case_ids)))


def candidate_case_ids(*directories: Path | str) -> tuple[int, ...]:
    """Return matching case IDs and reject candidate sets with different coverage."""

    if not directories:
        raise ValueError("At least one candidate directory is required")

    inventories = [_case_ids_in_directory(Path(directory)) for directory in directories]
    reference = inventories[0]
    if any(inventory != reference for inventory in inventories[1:]):
        raise ValueError("Candidate directories must contain the same case IDs")
    return reference
