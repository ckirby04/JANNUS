"""Cohort evaluation harness.

Wraps :mod:`jannus.evaluation.metrics` into the thing an external site actually
runs: point it at predictions and ground truth, get back per-case metrics,
bootstrap confidence intervals, size-stratified breakdowns, and the RANO-BM
measurable-disease scope — in one pass, with per-case failures isolated so a
single unreadable file cannot destroy a cohort run.

Reporting scopes
----------------
``all``          every annotated lesion, including sub-resolution ones.
``measurable``   RANO-BM measurable disease: longest axis >= 10 mm. This is the
                 scope of the proposed Indication for Use, and the scope on
                 which JANNUS is competitive with the cleared predicate.

Both are always reported. Reporting only the favourable scope would be
misleading, and a validation site should see the full picture.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..core.errors import EvaluationError
from ..core.logging import get_logger
from .metrics import (
    DEFAULT_SIZE_BINS,
    _longest_axis_mm,
    aggregate_stratified,
    aggregate_with_ci,
    compute_case_metrics,
    compute_case_stratified,
)

logger = get_logger("evaluation.harness")

#: RANO-BM measurable-disease threshold, longest axis in millimetres.
MEASURABLE_DISEASE_MM = 10.0

#: Bootstrap resamples for confidence intervals. 1000 is the conventional
#: minimum for a reported 95% CI.
DEFAULT_BOOTSTRAP = 1000


def voxel_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Global voxel-wise Dice. Returns 1.0 when both masks are empty."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    denom = int(pred_b.sum() + gt_b.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * int((pred_b & gt_b).sum()) / denom)


def filter_by_lesion_size(
    mask: np.ndarray,
    spacing: Sequence[float],
    *,
    min_longest_axis_mm: float,
) -> np.ndarray:
    """Drop connected components whose longest axis is below the threshold.

    Applied to *both* prediction and ground truth when scoping to measurable
    disease: scoping only the ground truth would count a correct small-lesion
    detection as a false positive.
    """
    from scipy import ndimage

    labeled, n = ndimage.label(mask.astype(bool))
    if n == 0:
        return mask.astype(np.uint8)
    keep = np.zeros_like(labeled, dtype=bool)
    for idx in range(1, n + 1):
        component = labeled == idx
        if _longest_axis_mm(component, spacing) >= min_longest_axis_mm:
            keep |= component
    return keep.astype(np.uint8)


@dataclass
class CaseMetrics:
    """Metrics for one case, at both reporting scopes."""

    case_token: str
    voxel_dice: float
    all_lesions: dict[str, float]
    measurable: dict[str, float]
    stratified: dict[str, dict] = field(default_factory=dict)
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    shape: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # `stratified` holds label arrays used only for aggregation; the
        # per-case JSON keeps the scalar summary and drops the bulk.
        d["stratified"] = {
            k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, np.ndarray))}
            for k, v in self.stratified.items()
        }
        return d


@dataclass
class CohortResult:
    """Aggregate result over a cohort."""

    n_cases: int
    n_failed: int
    per_case: list[CaseMetrics] = field(default_factory=list)
    aggregate_all: dict[str, dict[str, float]] = field(default_factory=dict)
    aggregate_measurable: dict[str, dict[str, float]] = field(default_factory=dict)
    stratified: dict[str, dict[str, float]] = field(default_factory=dict)
    voxel_dice_all: dict[str, float] = field(default_factory=dict)
    failures: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_cases": self.n_cases,
            "n_failed": self.n_failed,
            "voxel_dice": self.voxel_dice_all,
            "aggregate_all_lesions": self.aggregate_all,
            "aggregate_measurable_disease": self.aggregate_measurable,
            "stratified_by_size": self.stratified,
            "per_case": [c.to_dict() for c in self.per_case],
            "failures": self.failures,
        }

    def headline(self) -> dict[str, float]:
        """The handful of numbers a coordinating site compares across sites."""

        def _mean(agg: dict[str, dict[str, float]], key: str) -> float:
            return float(agg.get(key, {}).get("mean", float("nan")))

        return {
            "voxel_dice": self.voxel_dice_all.get("mean", float("nan")),
            "lesion_sensitivity_all": _mean(self.aggregate_all, "lesion_wise_sensitivity"),
            "fp_per_case_all": _mean(self.aggregate_all, "fp_per_case"),
            "lesion_sensitivity_measurable": _mean(
                self.aggregate_measurable, "lesion_wise_sensitivity"
            ),
            "lesion_dice_measurable": _mean(
                self.aggregate_measurable, "lesion_wise_dice_matched"
            ),
            "hd95_measurable": _mean(self.aggregate_measurable, "hd95_mm"),
        }


def evaluate_case(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    *,
    case_token: str,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    iou_threshold: float = 0.1,
    measurable_mm: float = MEASURABLE_DISEASE_MM,
) -> CaseMetrics:
    """Compute every metric for one case, at both scopes."""
    if prediction.shape != ground_truth.shape:
        raise EvaluationError(
            f"prediction shape {prediction.shape} does not match ground truth "
            f"{ground_truth.shape}"
        )

    pred = (prediction > 0).astype(np.uint8)
    gt = (ground_truth > 0).astype(np.uint8)
    spacing_t = tuple(float(s) for s in spacing)

    all_metrics = compute_case_metrics(
        pred, gt, spacing=spacing_t, iou_threshold=iou_threshold
    )

    pred_m = filter_by_lesion_size(pred, spacing_t, min_longest_axis_mm=measurable_mm)
    gt_m = filter_by_lesion_size(gt, spacing_t, min_longest_axis_mm=measurable_mm)
    measurable_metrics = compute_case_metrics(
        pred_m, gt_m, spacing=spacing_t, iou_threshold=iou_threshold
    )

    stratified = compute_case_stratified(
        pred, gt, spacing=spacing_t, iou_threshold=iou_threshold
    )

    return CaseMetrics(
        case_token=case_token,
        voxel_dice=voxel_dice(pred, gt),
        all_lesions=all_metrics,
        measurable=measurable_metrics,
        stratified=stratified,
        spacing=spacing_t,  # type: ignore[arg-type]
        shape=tuple(pred.shape),
    )


def evaluate_cohort(
    pairs: Sequence[tuple[str, np.ndarray, np.ndarray, Sequence[float]]],
    *,
    iou_threshold: float = 0.1,
    measurable_mm: float = MEASURABLE_DISEASE_MM,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
    size_bins: Sequence[tuple[str, tuple[float, float]]] | None = None,
) -> CohortResult:
    """Evaluate a whole cohort.

    Args:
        pairs: ``(case_token, prediction, ground_truth, spacing)`` per case.
            ``case_token`` must already be pseudonymised — results are shared.
        iou_threshold: IoU above which a predicted component counts as matching
            a ground-truth lesion. 0.1 is deliberately permissive: for a 4 mm
            metastasis a stricter threshold measures annotator boundary
            disagreement more than detection.
        n_bootstrap: resamples for the 95% confidence intervals.

    Per-case failures are captured, not raised, so one bad case cannot void a
    cohort that took days to produce.
    """
    if not pairs:
        raise EvaluationError(
            "No cases to evaluate.",
            remedy="Check that predictions and ground truth share case identifiers.",
        )

    bins = list(size_bins) if size_bins is not None else list(DEFAULT_SIZE_BINS)
    per_case: list[CaseMetrics] = []
    failures: list[dict[str, str]] = []

    for case_token, pred, gt, spacing in pairs:
        try:
            per_case.append(
                evaluate_case(
                    pred, gt,
                    case_token=case_token,
                    spacing=spacing,
                    iou_threshold=iou_threshold,
                    measurable_mm=measurable_mm,
                )
            )
        except Exception as exc:
            logger.error("evaluation failed for %s: %s", case_token, exc)
            failures.append({"case": case_token, "reason": str(exc)})

    if not per_case:
        raise EvaluationError(
            f"All {len(pairs)} case(s) failed evaluation.",
            remedy="See the failure list in the report for per-case reasons.",
        )

    dice_values = np.asarray([c.voxel_dice for c in per_case], dtype=np.float64)
    from .metrics import bootstrap_ci

    dice_low, dice_high = bootstrap_ci(dice_values, n_bootstrap=n_bootstrap)

    return CohortResult(
        n_cases=len(per_case),
        n_failed=len(failures),
        per_case=per_case,
        aggregate_all=aggregate_with_ci(
            [c.all_lesions for c in per_case], n_bootstrap=n_bootstrap
        ),
        aggregate_measurable=aggregate_with_ci(
            [c.measurable for c in per_case], n_bootstrap=n_bootstrap
        ),
        stratified=aggregate_stratified([c.stratified for c in per_case], bins=bins),
        voxel_dice_all={
            "mean": float(dice_values.mean()),
            "std": float(dice_values.std()),
            "ci_low": dice_low,
            "ci_high": dice_high,
            "n": len(dice_values),
        },
        failures=failures,
    )


def match_predictions_to_truth(
    prediction_dir: str | Path,
    index,
    *,
    suffix: str = "_seg.nii.gz",
) -> list[tuple[str, Path, Path, tuple[float, float, float]]]:
    """Pair prediction files with ground truth from a scanned dataset index.

    Returns ``(case_token, prediction_path, ground_truth_path, spacing)``.
    Cases without a prediction or without ground truth are skipped, and the
    count of each is logged so a site notices a partial run.
    """
    import nibabel as nib

    prediction_dir = Path(prediction_dir)
    pairs: list[tuple[str, Path, Path, tuple[float, float, float]]] = []
    missing_pred = 0
    missing_gt = 0

    for case in index.cases:
        if case.ground_truth is None:
            missing_gt += 1
            continue
        pred_path = prediction_dir / f"{case.case_id}{suffix}"
        if not pred_path.is_file():
            missing_pred += 1
            continue
        zooms = nib.load(str(case.channels[next(iter(case.channels))])).header.get_zooms()[:3]
        pairs.append((case.token, pred_path, case.ground_truth, tuple(float(z) for z in zooms)))

    if missing_pred:
        logger.warning("%d case(s) had no prediction file in %s", missing_pred, prediction_dir)
    if missing_gt:
        logger.warning("%d case(s) had no ground truth and were skipped", missing_gt)
    return pairs
