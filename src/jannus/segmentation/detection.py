"""
Stage 1: candidate detection sweep.

Runs nnDetection on the full contrast-enhanced MRI volume and returns a
list of 3D bounding boxes with confidence scores. The pipeline then crops
around each box (with configurable padding) and hands the crops to the
segmentation ensemble.

Why two-stage design
--------------------
nnDetection is tuned for recall at a low score threshold — false positives
here are cheap (the segmentation ensemble + stacker will reject them),
while false negatives are unrecoverable further down the pipeline because
the segmentation models never see the missed region. That asymmetry is why
`score_threshold` in `configs/models.yaml` is intentionally set very low
(default 0.05).

Fallback behaviour
------------------
nnDetection is an external heavyweight dependency whose dataset fingerprint
cannot be generated automatically; when it is not installed or not set up,
this module falls back to a `WholeVolumeDetector` that emits a single
bounding box covering the entire volume. The rest of the pipeline still
runs correctly, it just does not benefit from the detection crop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Candidate data class
# ---------------------------------------------------------------------------

@dataclass
class DetectionCandidate:
    """A single 3D candidate region emitted by the detection stage.

    Coordinates are in the full-volume voxel space and follow the
    half-open convention `[start, stop)`, matching numpy slicing.
    """

    start: tuple[int, int, int]   # (h0, w0, d0)
    stop: tuple[int, int, int]    # (h1, w1, d1)
    score: float

    def slicer(self) -> tuple[slice, slice, slice]:
        return (
            slice(self.start[0], self.stop[0]),
            slice(self.start[1], self.stop[1]),
            slice(self.start[2], self.stop[2]),
        )

    def shape(self) -> tuple[int, int, int]:
        return (
            self.stop[0] - self.start[0],
            self.stop[1] - self.start[1],
            self.stop[2] - self.start[2],
        )


# ---------------------------------------------------------------------------
# Detector interface
# ---------------------------------------------------------------------------

class Detector:
    """Base class for stage-1 detectors.

    A detector takes the full 4-channel MRI volume (numpy or torch tensor)
    and returns a list of `DetectionCandidate` objects in ascending score
    order, clipped to `max_candidates` at the call site.
    """

    name: str = "base"

    def detect(self, volume: np.ndarray) -> list[DetectionCandidate]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Whole-volume fallback
# ---------------------------------------------------------------------------

class WholeVolumeDetector(Detector):
    """Fallback detector: returns one candidate covering the entire volume.

    Used when nnDetection is not installed / not set up. The downstream
    pipeline is still correct — it just doesn't get the crop-level speedup
    or the recall boost from a dedicated detector.
    """

    name = "wholevolume"

    def detect(self, volume: np.ndarray) -> list[DetectionCandidate]:
        _, H, W, D = volume.shape
        return [DetectionCandidate(
            start=(0, 0, 0),
            stop=(H, W, D),
            score=1.0,
        )]


# ---------------------------------------------------------------------------
# nnDetection backend
# ---------------------------------------------------------------------------

class NnDetectionDetector(Detector):
    """Wraps nnDetection's RetinaUNet3D inference on a single volume.

    The underlying nnDetection API expects files on disk (dataset format),
    so this adapter writes the input volume to a temporary workspace,
    invokes the predictor, and parses the resulting per-case predictions
    into `DetectionCandidate` objects.

    TODO(manual): Dataset fingerprint generation
        nnDetection requires dataset-specific preprocessing and model
        planning which cannot be automated — run
            nndet_prep   Dataset001_BrainMets
            nndet_unpack Dataset001_BrainMets
            nndet_train  Dataset001_BrainMets -tr v001
        (or use an existing trained model) before this backend can do
        anything useful. See https://github.com/MIC-DKFZ/nnDetection.
    """

    name = "nndetection"

    def __init__(
        self,
        model_path: str | None = None,
        score_threshold: float = 0.05,
        max_candidates: int = 50,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.score_threshold = score_threshold
        self.max_candidates = max_candidates
        self._available = self._try_init_nndet()

    def _try_init_nndet(self) -> bool:
        try:
            # nnDetection's importable module name is `nndet`, not
            # `nndetection`. The import is deferred so that missing
            # dependencies don't break `import detection` at module load.
            import nndet  # noqa: F401
        except Exception:
            return False
        return self.model_path is not None and self.model_path.exists()

    def detect(self, volume: np.ndarray) -> list[DetectionCandidate]:
        if not self._available:
            # Fallback: emit a whole-volume candidate so the pipeline still
            # runs end-to-end. A log warning would be appropriate here, but
            # the pipeline-level wiring logs its detector choice once.
            return WholeVolumeDetector().detect(volume)

        # TODO(manual): plug in a real nnDetection RetinaUNet3D predictor
        # here. The typical call looks roughly like:
        #
        #     from nndet.inference.predictor import Predictor
        #     predictor = Predictor.from_folder(self.model_path)
        #     boxes, scores, _ = predictor.predict_case(volume)
        #
        # but the exact API depends on the nnDetection version; see
        # nnDetection's `projects/Task016_Luna/scripts/inference.py` for
        # a reference. Until that wiring is in place, fall back.
        return WholeVolumeDetector().detect(volume)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_detector(cfg: dict[str, Any]) -> Detector:
    """Build the stage-1 detector from the `detection:` section of
    `configs/models.yaml`.
    """
    if not cfg.get("enabled", True):
        return WholeVolumeDetector()

    backend = cfg.get("backend", "nndetection")
    if backend == "wholevolume":
        return WholeVolumeDetector()
    if backend == "nndetection":
        return NnDetectionDetector(
            model_path=cfg.get("model_path"),
            score_threshold=float(cfg.get("score_threshold", 0.05)),
            max_candidates=int(cfg.get("max_candidates", 50)),
        )
    raise ValueError(f"Unknown detection backend: {backend!r}")


# ---------------------------------------------------------------------------
# Crop extraction utilities
# ---------------------------------------------------------------------------

def expand_candidate(
    candidate: DetectionCandidate,
    volume_shape: tuple[int, int, int],
    padding_voxels: int,
    min_size: tuple[int, int, int],
) -> DetectionCandidate:
    """Expand a candidate bounding box with `padding_voxels` of context and
    ensure it reaches at least `min_size` in every axis, clipping to the
    full-volume shape. Returns a new candidate (input is not mutated).
    """
    H, W, D = volume_shape
    h0, w0, d0 = candidate.start
    h1, w1, d1 = candidate.stop

    # Apply padding
    h0 = max(0, h0 - padding_voxels)
    w0 = max(0, w0 - padding_voxels)
    d0 = max(0, d0 - padding_voxels)
    h1 = min(H, h1 + padding_voxels)
    w1 = min(W, w1 + padding_voxels)
    d1 = min(D, d1 + padding_voxels)

    # Enforce minimum size by symmetric growth around the centre.
    def _grow(lo: int, hi: int, target: int, axis_len: int) -> tuple[int, int]:
        cur = hi - lo
        if cur >= target:
            return lo, hi
        need = target - cur
        grow_lo = need // 2
        grow_hi = need - grow_lo
        new_lo = lo - grow_lo
        new_hi = hi + grow_hi
        if new_lo < 0:
            new_hi -= new_lo
            new_lo = 0
        if new_hi > axis_len:
            new_lo -= (new_hi - axis_len)
            new_hi = axis_len
            new_lo = max(0, new_lo)
        return new_lo, new_hi

    h0, h1 = _grow(h0, h1, min_size[0], H)
    w0, w1 = _grow(w0, w1, min_size[1], W)
    d0, d1 = _grow(d0, d1, min_size[2], D)

    return DetectionCandidate(
        start=(h0, w0, d0),
        stop=(h1, w1, d1),
        score=candidate.score,
    )


def extract_crop(volume: np.ndarray, candidate: DetectionCandidate) -> np.ndarray:
    """Extract the (C, h, w, d) crop described by `candidate` from the
    full-volume (C, H, W, D) array. Slicing is view-only (no copy)."""
    sl = candidate.slicer()
    return volume[:, sl[0], sl[1], sl[2]]
