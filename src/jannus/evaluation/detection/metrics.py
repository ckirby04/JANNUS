"""Detection metrics: FROC, fixed-FP operating points, size strata, scan-level rates.

Everything here works on lesion- and component-level records, not on volumes,
so a patient bootstrap can recompute every number thousands of times in
seconds. Records are built once per (case, voxel threshold, hit rule, scorer)
from :class:`~.matching.MatchResult` by :func:`records_from_match`.

Conventions (DECISIONS.md D6, D7):

- A lesion's score is the max score among the predictions that hit it, or
  ``-inf`` if none hit it. A lesion counts as detected at ``tau`` when its
  score is ``>= tau``.
- Sensitivity is pooled: detected lesions over all lesions, across scans.
- FP/scan is FP components over **all** scans, including scans with no
  lesions.
- Every function takes optional per-case weights. The patient bootstrap
  (``bootstrap.py``) passes resampling multiplicities through them, instead
  of copying records.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .matching import MatchResult

#: Fixed operating points, mean FP per scan (task brief).
DEFAULT_FP_TARGETS = (0.5, 1.0, 2.0, 3.0)


@dataclass(frozen=True)
class DetectionData:
    """Flattened lesion and FP records for one cohort under one configuration."""

    case_ids: tuple[str, ...]
    patient_ids: tuple[str, ...]        # per case; scans of one patient share an id
    gt_case: np.ndarray                 # (N_gt,) index into case_ids
    gt_score: np.ndarray                # (N_gt,) -inf if never hit
    gt_bin: np.ndarray                  # (N_gt,) size-bin name of the GT lesion
    fp_case: np.ndarray                 # (N_fp,) index into case_ids
    fp_score: np.ndarray                # (N_fp,)

    def __post_init__(self) -> None:
        if len(self.case_ids) != len(self.patient_ids):
            raise ValueError("one patient id per case is required")
        if len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError("duplicate case id")
        n = len(self.case_ids)
        for name, idx in (("gt_case", self.gt_case), ("fp_case", self.fp_case)):
            if idx.size and (idx.min() < 0 or idx.max() >= n):
                raise ValueError(f"{name} index out of range")
        if not (len(self.gt_case) == len(self.gt_score) == len(self.gt_bin)):
            raise ValueError("gt arrays must align")
        if len(self.fp_case) != len(self.fp_score):
            raise ValueError("fp arrays must align")
        if np.isnan(self.gt_score).any() or not np.all(np.isfinite(self.fp_score)):
            raise ValueError("scores must be finite (gt may be -inf for 'never hit')")

    @property
    def n_cases(self) -> int:
        return len(self.case_ids)

    def subset(self, cases: Sequence[str]) -> DetectionData:
        """The same records restricted to ``cases``, re-indexed."""
        keep = [self.case_ids.index(c) for c in cases]
        remap = -np.ones(self.n_cases, dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        g = remap[self.gt_case] >= 0
        f = remap[self.fp_case] >= 0
        return DetectionData(
            case_ids=tuple(self.case_ids[i] for i in keep),
            patient_ids=tuple(self.patient_ids[i] for i in keep),
            gt_case=remap[self.gt_case[g]],
            gt_score=self.gt_score[g],
            gt_bin=self.gt_bin[g],
            fp_case=remap[self.fp_case[f]],
            fp_score=self.fp_score[f],
        )


def records_from_match(
    match: MatchResult, gt_bins: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(gt_score, gt_bin, fp_score)`` for one case, from a match and the GT size bins."""
    if len(gt_bins) != match.n_gt:
        raise ValueError("one size bin per GT lesion is required")
    return match.gt_score, np.asarray(gt_bins, dtype=object), match.pred_scores[match.pred_is_fp]


def concat_cases(
    per_case: Sequence[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]],
) -> DetectionData:
    """Build :class:`DetectionData` from ``(case_id, patient_id, gt_score, gt_bin, fp_score)`` tuples."""
    gt_case, gt_score, gt_bin, fp_case, fp_score = [], [], [], [], []
    for i, (_, _, gs, gb, fs) in enumerate(per_case):
        gt_case.append(np.full(len(gs), i, dtype=np.int64))
        gt_score.append(np.asarray(gs, dtype=np.float64))
        gt_bin.append(np.asarray(gb, dtype=object))
        fp_case.append(np.full(len(fs), i, dtype=np.int64))
        fp_score.append(np.asarray(fs, dtype=np.float64))

    def cat(parts: list[np.ndarray], dtype) -> np.ndarray:
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)

    return DetectionData(
        case_ids=tuple(c[0] for c in per_case),
        patient_ids=tuple(c[1] for c in per_case),
        gt_case=cat(gt_case, np.int64),
        gt_score=cat(gt_score, np.float64),
        gt_bin=cat(gt_bin, object),
        fp_case=cat(fp_case, np.int64),
        fp_score=cat(fp_score, np.float64),
    )


def _weights(data: DetectionData, case_weight: np.ndarray | None) -> np.ndarray:
    if case_weight is None:
        return np.ones(data.n_cases)
    w = np.asarray(case_weight, dtype=np.float64)
    if w.shape != (data.n_cases,) or (w < 0).any():
        raise ValueError("case_weight must be non-negative, one per case")
    return w


@dataclass(frozen=True)
class FROC:
    """A step FROC curve. Point 0 is ``tau = +inf`` (nothing flagged). Thresholds descend."""

    thresholds: np.ndarray
    sensitivity: np.ndarray
    fp_per_scan: np.ndarray
    n_lesions: float          # weighted lesion count in the denominator
    n_scans: float

    def sensitivity_at_fp(self, target: float) -> tuple[float, float, bool]:
        """``(sensitivity, tau, reached)`` at the most permissive threshold with FP/scan <= target.

        This is a step function with no interpolation. ``reached`` is False when
        even the most permissive threshold stays below the target. The curve
        then ends early: the value returned is the best sensitivity available
        at this voxel threshold, and it is flagged rather than extrapolated.
        """
        ok = np.flatnonzero(self.fp_per_scan <= target + 1e-12)
        i = int(ok[-1])  # point 0 (fp = 0) always qualifies
        return float(self.sensitivity[i]), float(self.thresholds[i]), bool(
            self.fp_per_scan[-1] >= target - 1e-12
        )


def froc(
    data: DetectionData,
    *,
    case_weight: np.ndarray | None = None,
    gt_mask: np.ndarray | None = None,
) -> FROC:
    """FROC over every distinct score. ``gt_mask`` restricts the *lesions*, e.g. to one size
    bin. FPs are never restricted, because an FP has no GT size."""
    w = _weights(data, case_weight)
    n_scans = float(w.sum())
    gsel = np.ones(len(data.gt_score), dtype=bool) if gt_mask is None else np.asarray(gt_mask)
    g_score = data.gt_score[gsel]
    g_w = w[data.gt_case[gsel]]
    f_w = w[data.fp_case]
    n_lesions = float(g_w.sum())

    finite = np.isfinite(g_score)
    taus = np.unique(np.concatenate([g_score[finite], data.fp_score]))[::-1]
    thresholds = np.concatenate([[np.inf], taus])

    def cum_at(scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
        order = np.argsort(-scores, kind="stable")
        s, cw = scores[order], np.cumsum(weights[order])
        # count of weight with score >= tau, for each tau (descending)
        idx = np.searchsorted(-s, -thresholds, side="right")
        return np.where(idx > 0, cw[np.maximum(idx - 1, 0)], 0.0) if len(s) else np.zeros(len(thresholds))

    detected_w = cum_at(np.where(finite, g_score, -np.inf), np.where(finite, g_w, 0.0))
    fp_w = cum_at(data.fp_score, f_w)
    sens = detected_w / n_lesions if n_lesions > 0 else np.full(len(thresholds), np.nan)
    fps = fp_w / n_scans if n_scans > 0 else np.full(len(thresholds), np.nan)
    return FROC(thresholds=thresholds, sensitivity=sens, fp_per_scan=fps,
                n_lesions=n_lesions, n_scans=n_scans)


def cpm(curve: FROC, targets: Sequence[float] = DEFAULT_FP_TARGETS) -> float:
    """Mean sensitivity over the fixed FP targets: the selection objective (D6)."""
    return float(np.mean([curve.sensitivity_at_fp(t)[0] for t in targets]))


def operating_point(
    data: DetectionData,
    tau: float,
    *,
    bins: Sequence[str] = (),
    case_weight: np.ndarray | None = None,
) -> dict[str, float]:
    """Every scalar reported at one score threshold.

    Returns pooled sensitivity, FP/scan, per-bin sensitivity with lesion counts,
    and two scan-level rates, both computed over scans with at least one lesion:
    the fraction where any lesion is flagged, and the fraction where all are.
    """
    w = _weights(data, case_weight)
    det = np.isfinite(data.gt_score) & (data.gt_score >= tau)
    gw = w[data.gt_case]
    out: dict[str, float] = {
        "tau": float(tau),
        "n_scans": float(w.sum()),
        "n_lesions": float(gw.sum()),
        "n_detected": float(gw[det].sum()),
        "sensitivity": float(gw[det].sum() / gw.sum()) if gw.sum() > 0 else float("nan"),
        "n_fp": float(w[data.fp_case][data.fp_score >= tau].sum()),
    }
    out["fp_per_scan"] = out["n_fp"] / out["n_scans"] if out["n_scans"] > 0 else float("nan")
    for b in bins:
        m = data.gt_bin == b
        n = float(gw[m].sum())
        out[f"n_lesions[{b}]"] = n
        out[f"n_detected[{b}]"] = float(gw[m & det].sum())
        out[f"sensitivity[{b}]"] = out[f"n_detected[{b}]"] / n if n > 0 else float("nan")

    n_les = np.bincount(data.gt_case, minlength=data.n_cases)
    n_det = np.bincount(data.gt_case[det], minlength=data.n_cases)
    with_lesion = n_les > 0
    denom = float(w[with_lesion].sum())
    out["n_scans_with_lesions"] = denom
    out["scan_any_flagged"] = float(w[with_lesion & (n_det > 0)].sum() / denom) if denom else float("nan")
    out["scan_all_flagged"] = float(w[with_lesion & (n_det == n_les)].sum() / denom) if denom else float("nan")
    out["n_scans_without_lesions"] = float(w[~with_lesion].sum())
    no_les_fp = np.bincount(data.fp_case[data.fp_score >= tau], minlength=data.n_cases)[~with_lesion]
    out["fp_per_scan_lesion_free"] = (
        float((w[~with_lesion] * no_les_fp).sum() / w[~with_lesion].sum())
        if w[~with_lesion].sum() > 0 else float("nan")
    )
    return out


def summarize_curve(
    data: DetectionData,
    *,
    bins: Sequence[str],
    targets: Sequence[float] = DEFAULT_FP_TARGETS,
    case_weight: np.ndarray | None = None,
) -> dict[str, float]:
    """Sensitivity at each fixed FP target (overall and per bin), plus CPM.

    The per-bin value at an FP target uses the **overall** threshold that meets
    that target. Picking a separate threshold per bin would describe a system
    that knows each lesion's true size in advance.
    """
    overall = froc(data, case_weight=case_weight)
    out: dict[str, float] = {"cpm": cpm(overall, targets)}
    for t in targets:
        sens, tau, reached = overall.sensitivity_at_fp(t)
        out[f"sens@{t:g}"] = sens
        out[f"tau@{t:g}"] = tau
        out[f"reached@{t:g}"] = float(reached)
        op = operating_point(data, tau, bins=bins, case_weight=case_weight)
        for b in bins:
            out[f"sens@{t:g}[{b}]"] = op[f"sensitivity[{b}]"]
            out[f"n_lesions[{b}]"] = op[f"n_lesions[{b}]"]
    return out
