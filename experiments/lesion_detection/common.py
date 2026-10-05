"""Dataset-specific helpers shared by the experiment scripts.

These helpers know this cohort's naming and folder layout, so they live with the
experiment rather than in the ``jannus.evaluation.detection`` library.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
for p in (REPO / "src", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from jannus.evaluation.detection.geometry import (  # noqa: E402
    Spacing,
    native_header_geometry,
    resampled_spacing,
)


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def repo_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def read_ids(path: str | Path) -> list[str]:
    return [ln.strip() for ln in repo_path(path).read_text().splitlines() if ln.strip()]


def patient_id(scan_id: str) -> str:
    """UCSF scan IDs end in a timepoint letter (``UCSF_100238A`` -> ``UCSF_100238``).
    BMS IDs are one scan per patient. See LEAKAGE_FINDING.md."""
    if scan_id.startswith("UCSF_"):
        return re.sub(r"[A-Z]$", "", scan_id)
    return scan_id


def native_seg_path(scan_id: str, native_roots: dict[str, str]) -> Path:
    """Native-space GT for a ``preprocessed_256`` case, which is the source of the true geometry."""
    if scan_id.startswith("BMS_"):
        sid = scan_id.removeprefix("BMS_")
        return Path(native_roots["bms"]) / sid / "seg.nii.gz"
    if scan_id.startswith("UCSF_"):
        sid = scan_id.removeprefix("UCSF_")
        return Path(native_roots["ucsf"]) / sid / f"{sid}_seg.nii.gz"
    raise ValueError(f"unrecognised case id {scan_id!r}")


def true_spacing(scan_id: str, native_roots: dict[str, str], grid_shape: tuple[int, int, int]) -> Spacing:
    """True per-axis spacing of the resampled grid (D4, SPACING_DEFECT.md)."""
    shape, spacing = native_header_geometry(native_seg_path(scan_id, native_roots))
    return resampled_spacing(shape, spacing, grid_shape)


def load_maps_manifest(path: str | Path) -> dict[str, dict]:
    out = {}
    for line in repo_path(path).read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["case"]] = rec  # the last write wins if a case was re-run
    return out
