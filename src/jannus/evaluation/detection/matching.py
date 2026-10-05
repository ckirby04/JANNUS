"""Lesion extraction and prediction-to-lesion matching for detection (CADe-style) grading.

Two hit rules, both described in ``experiments/lesion_detection/DECISIONS.md``:

``overlap`` (primary)
    A predicted component hits a GT lesion when they share at least
    ``min_overlap_vox`` voxels and at least ``min_overlap_frac`` of the lesion.
    The defaults (1 voxel, 0 %) make any touch a hit. A component that touches a
    lesion but falls short of a stricter criterion is a *fragment*: it does not
    detect the lesion, and it is not counted as a false positive either.

``centroid``
    A predicted component hits a GT lesion when the two centroids are within
    ``max_distance_mm`` of each other, measured in physical millimetres.

Both rules count edge cases the same way:

* **split** (several components on one lesion): the lesion is detected once.
  Its detection score is the maximum score over the components that hit it.
  The other components are neither TPs nor FPs; :meth:`MatchResult.n_redundant`
  counts them.
* **merge** (one component on several lesions): every lesion it hits is
  detected, and the component contributes no FP.
* A predicted component is an FP only if it hits no lesion under the rule in
  force. Whether it is an FP does not depend on the score threshold, so FROC
  curves can be swept from a single match.

A component survives a score threshold ``tau`` when ``score >= tau``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .geometry import Spacing, as_spacing, equivalent_spherical_diameter, voxel_volume_mm3

#: Neighbourhood size -> ``generate_binary_structure`` rank.
_CONNECTIVITY_RANK = {6: 1, 18: 2, 26: 3}


def structuring_element(connectivity: int) -> np.ndarray:
    try:
        rank = _CONNECTIVITY_RANK[int(connectivity)]
    except KeyError:
        raise ValueError(f"connectivity must be 6, 18 or 26, got {connectivity}") from None
    return ndimage.generate_binary_structure(3, rank)


@dataclass(frozen=True)
class ComponentTable:
    """Connected components of a binary 3D mask, with physical measurements.

    Component ``k`` (1-based label) lives at array index ``k - 1``.
    """

    labels: np.ndarray            # int32 label volume, 0 = background
    spacing: Spacing
    n_vox: np.ndarray             # (n,) int64
    centroid_mm: np.ndarray       # (n, 3) float64, voxel-index centroid x spacing
    bbox: np.ndarray              # (n, 6) int64: lo_z, lo_y, lo_x, hi_z, hi_y, hi_x (hi exclusive)

    @property
    def n(self) -> int:
        return len(self.n_vox)

    @property
    def volume_mm3(self) -> np.ndarray:
        return self.n_vox * voxel_volume_mm3(self.spacing)

    @property
    def esd_mm(self) -> np.ndarray:
        return np.asarray(equivalent_spherical_diameter(self.volume_mm3), dtype=np.float64)

    @property
    def longest_axis_mm(self) -> np.ndarray:
        """Longest bounding-box edge in mm. Kept for continuity with prior size bins."""
        if self.n == 0:
            return np.zeros(0)
        extents = (self.bbox[:, 3:] - self.bbox[:, :3]) * np.asarray(self.spacing)
        return extents.max(axis=1)


def extract_components(
    mask: np.ndarray,
    spacing: Sequence[float],
    *,
    connectivity: int = 26,
) -> ComponentTable:
    """Label a binary mask and measure every component."""
    if mask.ndim != 3:
        raise ValueError(f"expected a 3D mask, got shape {mask.shape}")
    spacing_t = as_spacing(spacing)
    binary = np.asarray(mask).astype(bool, copy=False)
    labels, n = ndimage.label(binary, structure=structuring_element(connectivity))
    labels = labels.astype(np.int32, copy=False)
    if n == 0:
        return ComponentTable(
            labels=labels,
            spacing=spacing_t,
            n_vox=np.zeros(0, dtype=np.int64),
            centroid_mm=np.zeros((0, 3)),
            bbox=np.zeros((0, 6), dtype=np.int64),
        )
    ids = np.arange(1, n + 1)
    n_vox = np.bincount(labels.ravel(), minlength=n + 1)[1:].astype(np.int64)
    centroid_vox = np.asarray(ndimage.center_of_mass(binary, labels, ids), dtype=np.float64)
    bbox = np.asarray(
        [[s.start for s in sl] + [s.stop for s in sl] for sl in ndimage.find_objects(labels)],
        dtype=np.int64,
    )
    return ComponentTable(
        labels=labels,
        spacing=spacing_t,
        n_vox=n_vox,
        centroid_mm=centroid_vox * np.asarray(spacing_t),
        bbox=bbox,
    )


def overlap_counts(pred: ComponentTable, gt: ComponentTable) -> dict[tuple[int, int], int]:
    """Voxel overlap for every touching ``(pred_id, gt_id)`` pair (1-based ids)."""
    if pred.labels.shape != gt.labels.shape:
        raise ValueError(f"shape mismatch: {pred.labels.shape} vs {gt.labels.shape}")
    both = (pred.labels > 0) & (gt.labels > 0)
    if not both.any():
        return {}
    keys = pred.labels[both].astype(np.int64) * (gt.n + 1) + gt.labels[both]
    uniq, counts = np.unique(keys, return_counts=True)
    return {
        (int(k // (gt.n + 1)), int(k % (gt.n + 1))): int(c)
        for k, c in zip(uniq, counts, strict=True)
    }


def bbox_iou_matrix(pred: ComponentTable, gt: ComponentTable) -> np.ndarray:
    """(n_pred, n_gt) axis-aligned bounding-box IoU. Supplementary only, never a hit rule."""
    if pred.n == 0 or gt.n == 0:
        return np.zeros((pred.n, gt.n))
    lo = np.maximum(pred.bbox[:, None, :3], gt.bbox[None, :, :3])
    hi = np.minimum(pred.bbox[:, None, 3:], gt.bbox[None, :, 3:])
    inter = np.clip(hi - lo, 0, None).prod(axis=2)
    vol_p = (pred.bbox[:, 3:] - pred.bbox[:, :3]).prod(axis=1)
    vol_g = (gt.bbox[:, 3:] - gt.bbox[:, :3]).prod(axis=1)
    union = vol_p[:, None] + vol_g[None, :] - inter
    return inter / union


@dataclass
class MatchResult:
    """Outcome of matching one case's predicted components to its GT lesions.

    Ids in ``pred_hits`` / ``gt_hit_by`` are 1-based component labels.
    """

    rule: str
    pred_scores: np.ndarray                     # (n_pred,)
    pred_hits: list[tuple[int, ...]]            # GT ids each prediction hits under the rule
    pred_touches: np.ndarray                    # (n_pred,) bool: shares >=1 voxel with any GT
    gt_hit_by: list[tuple[int, ...]]            # prediction ids hitting each GT lesion
    gt_overlap_vox: np.ndarray                  # (n_gt,) best voxel overlap with any prediction
    gt_overlap_frac: np.ndarray                 # (n_gt,) best overlap / lesion voxels
    gt_best_bbox_iou: np.ndarray                # (n_gt,) supplementary
    params: dict[str, float] = field(default_factory=dict)

    @property
    def n_pred(self) -> int:
        return len(self.pred_hits)

    @property
    def n_gt(self) -> int:
        return len(self.gt_hit_by)

    @property
    def pred_is_fp(self) -> np.ndarray:
        """FP = hits no lesion. Under ``overlap``, a fragment that touches a lesion is not an FP."""
        hits_none = np.array([len(h) == 0 for h in self.pred_hits], dtype=bool)
        if self.rule == "overlap":
            return hits_none & ~self.pred_touches
        return hits_none

    @property
    def pred_is_fragment(self) -> np.ndarray:
        """Touches a lesion but meets no hit criterion (only possible with strict overlap)."""
        hits_none = np.array([len(h) == 0 for h in self.pred_hits], dtype=bool)
        if self.rule == "overlap":
            return hits_none & self.pred_touches
        return np.zeros(self.n_pred, dtype=bool)

    @property
    def gt_score(self) -> np.ndarray:
        """Per-lesion detection score: max score over hitting predictions, -inf if none."""
        out = np.full(self.n_gt, -np.inf)
        for g, preds in enumerate(self.gt_hit_by):
            if preds:
                out[g] = max(self.pred_scores[p - 1] for p in preds)
        return out

    def surviving(self, tau: float = -np.inf) -> np.ndarray:
        return self.pred_scores >= tau

    def detected(self, tau: float = -np.inf) -> np.ndarray:
        # An unhit lesion scores -inf; test for a hit explicitly so tau=-inf cannot pass it.
        score = self.gt_score
        return np.isfinite(score) & (score >= tau)

    def n_fp(self, tau: float = -np.inf) -> int:
        return int((self.pred_is_fp & self.surviving(tau)).sum())

    def n_split_lesions(self, tau: float = -np.inf) -> int:
        """Lesions hit by two or more surviving predictions."""
        keep = self.surviving(tau)
        return sum(1 for preds in self.gt_hit_by if sum(keep[p - 1] for p in preds) >= 2)

    def n_merge_predictions(self, tau: float = -np.inf) -> int:
        """Surviving predictions that hit two or more lesions."""
        keep = self.surviving(tau)
        return sum(1 for i, h in enumerate(self.pred_hits) if keep[i] and len(h) >= 2)

    def n_redundant(self, tau: float = -np.inf) -> int:
        """Surviving non-FP predictions whose every lesion is already credited to another.

        For each lesion, the credited prediction is the highest-scoring one that hits
        it; ties go to the lower id. A prediction is redundant if it is credited for
        none of the lesions it hits.
        """
        keep = self.surviving(tau)
        credited: set[int] = set()
        for preds in self.gt_hit_by:
            alive = [p for p in preds if keep[p - 1]]
            if alive:
                credited.add(min(alive, key=lambda p: (-self.pred_scores[p - 1], p)))
        return sum(
            1 for i, h in enumerate(self.pred_hits) if keep[i] and h and (i + 1) not in credited
        )


def _check_scores(pred: ComponentTable, scores: np.ndarray | None) -> np.ndarray:
    if scores is None:
        return np.ones(pred.n)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (pred.n,):
        raise ValueError(f"expected {pred.n} scores, got shape {scores.shape}")
    if not np.all(np.isfinite(scores)):
        raise ValueError("component scores must be finite")
    return scores


def _gt_overlap_stats(
    pred: ComponentTable, gt: ComponentTable, overlaps: dict[tuple[int, int], int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    best_vox = np.zeros(gt.n, dtype=np.int64)
    for (_, g), c in overlaps.items():
        best_vox[g - 1] = max(best_vox[g - 1], c)
    frac = best_vox / np.maximum(gt.n_vox, 1) if gt.n else np.zeros(0)
    iou = bbox_iou_matrix(pred, gt)
    best_iou = iou.max(axis=0) if pred.n else np.zeros(gt.n)
    return best_vox, frac, best_iou


def match_by_overlap(
    pred: ComponentTable,
    gt: ComponentTable,
    scores: np.ndarray | None = None,
    *,
    min_overlap_vox: int = 1,
    min_overlap_frac: float = 0.0,
) -> MatchResult:
    """Primary rule. A prediction hits a lesion when their overlap is at least
    ``min(min_overlap_vox, lesion_voxels)`` voxels AND at least
    ``min_overlap_frac`` of the lesion's voxels.

    Capping the voxel criterion at the lesion's own size means a strict setting
    can still be met by a lesion smaller than ``min_overlap_vox``.
    """
    if min_overlap_vox < 1:
        raise ValueError("min_overlap_vox must be >= 1")
    if not 0.0 <= min_overlap_frac <= 1.0:
        raise ValueError("min_overlap_frac must be in [0, 1]")
    scores = _check_scores(pred, scores)
    overlaps = overlap_counts(pred, gt)

    pred_hits: list[list[int]] = [[] for _ in range(pred.n)]
    gt_hit_by: list[list[int]] = [[] for _ in range(gt.n)]
    touches = np.zeros(pred.n, dtype=bool)
    for (p, g), c in sorted(overlaps.items()):
        touches[p - 1] = True
        gt_vox = int(gt.n_vox[g - 1])
        if c >= min(min_overlap_vox, gt_vox) and c / gt_vox >= min_overlap_frac:
            pred_hits[p - 1].append(g)
            gt_hit_by[g - 1].append(p)

    best_vox, frac, best_iou = _gt_overlap_stats(pred, gt, overlaps)
    return MatchResult(
        rule="overlap",
        pred_scores=scores,
        pred_hits=[tuple(h) for h in pred_hits],
        pred_touches=touches,
        gt_hit_by=[tuple(h) for h in gt_hit_by],
        gt_overlap_vox=best_vox,
        gt_overlap_frac=frac,
        gt_best_bbox_iou=best_iou,
        params={"min_overlap_vox": min_overlap_vox, "min_overlap_frac": min_overlap_frac},
    )


def match_by_centroid(
    pred: ComponentTable,
    gt: ComponentTable,
    scores: np.ndarray | None = None,
    *,
    max_distance_mm: float,
) -> MatchResult:
    """Secondary rule. A prediction hits every lesion whose centroid lies within
    ``max_distance_mm`` (inclusive, physical mm) of the prediction's centroid."""
    if not max_distance_mm > 0:
        raise ValueError("max_distance_mm must be positive")
    if pred.spacing != gt.spacing:
        raise ValueError("prediction and GT spacing differ")
    scores = _check_scores(pred, scores)
    overlaps = overlap_counts(pred, gt)

    if pred.n and gt.n:
        diff = pred.centroid_mm[:, None, :] - gt.centroid_mm[None, :, :]
        within = np.sqrt((diff**2).sum(axis=2)) <= max_distance_mm
    else:
        within = np.zeros((pred.n, gt.n), dtype=bool)

    touches = np.zeros(pred.n, dtype=bool)
    for p, _ in overlaps:
        touches[p - 1] = True
    best_vox, frac, best_iou = _gt_overlap_stats(pred, gt, overlaps)
    return MatchResult(
        rule="centroid",
        pred_scores=scores,
        pred_hits=[tuple(int(g) + 1 for g in np.flatnonzero(row)) for row in within],
        pred_touches=touches,
        gt_hit_by=[tuple(int(p) + 1 for p in np.flatnonzero(col)) for col in within.T],
        gt_overlap_vox=best_vox,
        gt_overlap_frac=frac,
        gt_best_bbox_iou=best_iou,
        params={"max_distance_mm": float(max_distance_mm)},
    )


def match(
    pred: ComponentTable,
    gt: ComponentTable,
    scores: np.ndarray | None = None,
    *,
    rule: str = "overlap",
    **params: float,
) -> MatchResult:
    """Dispatch by rule name, as selected in an experiment config."""
    if rule == "overlap":
        return match_by_overlap(pred, gt, scores, **params)  # type: ignore[arg-type]
    if rule == "centroid":
        return match_by_centroid(pred, gt, scores, **params)
    raise ValueError(f"unknown matching rule {rule!r}; expected 'overlap' or 'centroid'")
