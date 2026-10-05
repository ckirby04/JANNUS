"""Physical geometry for lesion-level detection evaluation.

Every size and distance in this subpackage is in millimetres computed from a
per-axis spacing tuple. Nothing here assumes isotropy or a 1 mm grid.

The ``preprocessed_256`` development data carry identity affines that misstate
the real voxel size (see ``experiments/lesion_detection/SPACING_DEFECT.md``);
:func:`resampled_spacing` recovers the true spacing from the native header.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

import numpy as np

Spacing = tuple[float, float, float]


def as_spacing(spacing: Sequence[float]) -> Spacing:
    """Validate and normalise a 3-axis spacing. Rejects scalars on purpose."""
    values = tuple(float(s) for s in spacing)
    if len(values) != 3:
        raise ValueError(f"spacing must have 3 axes, got {values}")
    if not all(np.isfinite(v) and v > 0 for v in values):
        raise ValueError(f"spacing must be finite and positive, got {values}")
    return values  # type: ignore[return-value]


def resampled_spacing(
    native_shape: Sequence[int],
    native_spacing: Sequence[float],
    target_shape: Sequence[int],
) -> Spacing:
    """True per-axis spacing of a volume resized from ``native_shape`` to ``target_shape``.

    Resizing preserves the field of view, so each axis keeps its physical extent:
    ``target_spacing = native_spacing * native_shape / target_shape``.
    """
    native = np.asarray(as_spacing(native_spacing), dtype=np.float64)
    n_shape = np.asarray(native_shape, dtype=np.float64)
    t_shape = np.asarray(target_shape, dtype=np.float64)
    if n_shape.shape != (3,) or t_shape.shape != (3,):
        raise ValueError("native_shape and target_shape must have 3 axes")
    return as_spacing(native * n_shape / t_shape)


def native_header_geometry(path: str | Path) -> tuple[tuple[int, int, int], Spacing]:
    """``(shape, spacing)`` of a NIfTI from its header only (no voxel data read)."""
    import nibabel as nib

    header = nib.load(str(path)).header
    shape = tuple(int(s) for s in header.get_data_shape()[:3])
    return shape, as_spacing(header.get_zooms()[:3])  # type: ignore[return-value]


def voxel_volume_mm3(spacing: Sequence[float]) -> float:
    return float(np.prod(as_spacing(spacing)))


def equivalent_spherical_diameter(volume_mm3: float | np.ndarray) -> float | np.ndarray:
    """Diameter of the sphere with the given volume: ``(6 V / pi) ** (1/3)``."""
    return np.cbrt(6.0 * np.asarray(volume_mm3, dtype=np.float64) / np.pi)


def assign_size_bin(diameter_mm: float, bins: Sequence[tuple[str, float, float]]) -> str:
    """Name of the half-open ``[lo, hi)`` bin containing ``diameter_mm``.

    ``bins`` must tile ``[0, inf)`` without gaps; this is checked so a config
    typo cannot silently drop lesions from every bin.
    """
    validate_size_bins(bins)
    for name, lo, hi in bins:
        if lo <= diameter_mm < hi:
            return name
    raise ValueError(f"diameter {diameter_mm} outside all bins")  # unreachable if validated


def validate_size_bins(bins: Sequence[tuple[str, float, float]]) -> None:
    if not bins:
        raise ValueError("at least one size bin is required")
    if bins[0][1] != 0.0:
        raise ValueError("first size bin must start at 0")
    if bins[-1][2] != float("inf"):
        raise ValueError("last size bin must end at inf")
    for (_, _, hi), (_, lo, _) in pairwise(bins):
        if hi != lo:
            raise ValueError(f"size bins must be contiguous; gap or overlap at {hi} / {lo}")
