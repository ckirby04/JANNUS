"""Volume loading and intensity normalisation.

The normalisation here must match what the models were trained against:
**per-channel z-score over the whole volume**. That convention is inherited
from the v1.40 training and cache-build protocol, and changing it silently
degrades every metric. It is therefore implemented once, here, and used by
every entry point — CLI, API, and evaluation alike.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.errors import DataIntegrityError
from ..core.logging import get_logger
from .layout import ResolvedCase

logger = get_logger("data.loading")

#: Volumes whose spatial dims fall outside this envelope are almost certainly
#: not brain MRI (or are a mis-stacked 2D series), and would waste GPU-hours
#: before failing. Advisory: `validate-data` warns, inference proceeds.
PLAUSIBLE_SHAPE_RANGE = (32, 1024)

#: Voxel spacing the models were trained near, in mm. Outside this range,
#: results are extrapolation and the validation report says so.
PLAUSIBLE_SPACING_RANGE = (0.3, 5.0)


@dataclass
class LoadedVolume:
    """A case loaded and normalised, ready for the pipeline."""

    #: (C, H, W, D) float32, z-scored per channel.
    array: np.ndarray
    #: 4x4 affine from the first channel, used to write outputs back.
    affine: np.ndarray
    voxel_spacing: tuple[float, float, float]
    case_id: str
    #: Per-channel pre-normalisation statistics, for the QC report.
    channel_stats: dict[str, dict[str, float]]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.array.shape[1:])  # type: ignore[return-value]


def _load_nifti(path: Path):
    """Load a NIfTI, deferring the nibabel import so `--help` stays fast."""
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover
        raise DataIntegrityError(
            "nibabel is required to read imaging.",
            remedy="pip install 'jannus[segmentation]'",
        ) from exc

    try:
        return nib.load(str(path))
    except Exception as exc:
        raise DataIntegrityError(
            f"Could not read {path.name}: {exc}",
            remedy="Confirm the file is a valid NIfTI and is not truncated.",
        ) from exc


def zscore(array: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Per-volume z-score. Returns the normalised array and its input stats.

    A constant volume (std == 0) is returned as zeros rather than NaNs. That
    case means an all-background or failed acquisition; `validate-data` flags
    it, and letting NaNs reach the network would poison the whole batch.
    """
    mean = float(array.mean())
    std = float(array.std())
    stats = {
        "mean": mean,
        "std": std,
        "min": float(array.min()),
        "max": float(array.max()),
    }
    if std <= 0:
        return np.zeros_like(array, dtype=np.float32), stats
    return ((array - mean) / std).astype(np.float32), stats


def load_case(
    case: ResolvedCase,
    sequences: Sequence[str],
    *,
    normalise: bool = True,
) -> LoadedVolume:
    """Load one case into a normalised ``(C, H, W, D)`` array.

    All channels must share spatial dimensions; JANNUS does not resample
    between sequences, because doing so silently would hide a co-registration
    failure that materially changes the result. A shape mismatch is an error
    the site must resolve upstream.
    """
    arrays = []
    stats: dict[str, dict[str, float]] = {}
    affine: np.ndarray | None = None
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    reference_shape: tuple[int, ...] | None = None

    for channel in sequences:
        path = case.channels[channel]
        image = _load_nifti(path)
        data = np.asarray(image.dataobj, dtype=np.float32)

        if data.ndim == 4 and data.shape[-1] == 1:
            data = data[..., 0]  # squeeze a singleton time/vector dim
        if data.ndim != 3:
            raise DataIntegrityError(
                f"{case.case_id}/{path.name}: expected a 3D volume, got shape {data.shape}."
            )

        if reference_shape is None:
            reference_shape = data.shape
            affine = np.asarray(image.affine, dtype=np.float64)
            zooms = image.header.get_zooms()[:3]
            spacing = tuple(float(z) for z in zooms)  # type: ignore[assignment]
        elif data.shape != reference_shape:
            raise DataIntegrityError(
                f"{case.case_id}: channel {channel!r} ({path.name}) has shape "
                f"{data.shape}, but {sequences[0]!r} has shape {reference_shape}. "
                f"All sequences must be co-registered to a common grid.",
                remedy=(
                    "Co-register and resample the sequences to a shared grid before "
                    "running JANNUS. See docs/DATA_REQUIREMENTS.md."
                ),
            )

        if not np.isfinite(data).all():
            n_bad = int((~np.isfinite(data)).sum())
            logger.warning(
                "%s channel %s contains %d non-finite voxel(s); replaced with 0",
                case.token, channel, n_bad,
            )
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        if normalise:
            data, channel_stats = zscore(data)
        else:
            _, channel_stats = zscore(data)
        stats[channel] = channel_stats
        arrays.append(data)

    volume = np.stack(arrays, axis=0).astype(np.float32)
    assert affine is not None  # guaranteed: sequences is non-empty

    return LoadedVolume(
        array=volume,
        affine=affine,
        voxel_spacing=spacing,
        case_id=case.case_id,
        channel_stats=stats,
    )


def load_mask(path: str | Path) -> np.ndarray:
    """Load a segmentation as a binary ``uint8`` array.

    Any non-zero label is treated as foreground: multi-label ground truth from
    a site that also annotates oedema or resection cavities collapses to the
    metastasis-vs-background task JANNUS is scored on.
    """
    data = np.asarray(_load_nifti(Path(path)).dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    return (np.nan_to_num(data) > 0).astype(np.uint8)


def save_mask(
    mask: np.ndarray,
    affine: np.ndarray,
    path: str | Path,
) -> Path:
    """Write a binary mask as NIfTI, preserving the input geometry.

    The affine comes from the input volume, so the output overlays correctly in
    any viewer the site already uses.
    """
    import nibabel as nib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(mask.astype(np.uint8), affine)
    # Explicit dtype: nibabel otherwise infers from the array and can widen.
    image.set_data_dtype(np.uint8)
    nib.save(image, str(path))
    return path
