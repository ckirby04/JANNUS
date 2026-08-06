"""
FDA-style evaluation metrics for brain metastasis segmentation.

These mirror the metrics reported in Neosoma Brain Mets (K252922, Dec 2025)
and required for a 510(k) submission under 21 CFR 892.2050:

    1. Lesion-wise sensitivity (per-lesion recall via cc3d matching)
    2. False positive rate per case
    3. Lesion-wise Dice (BraTS-style: unmatched GT -> 0)
    4. Lesion-wise Dice (matched only: Neosoma/FDA reporting style)
    5. HD95 -- 95th percentile Hausdorff distance (mm)
    6. MSD  -- mean surface distance (mm)
    7. Small-lesion sensitivity (< 100 mm^3, i.e. ~ <5mm diameter)
    8. Bootstrap 95% CI on any per-case metric
    9. Stratified sens/FPR/DSC by lesion-size bin (<3mm, 3-5, 5-10, 10-20, >20mm)

Assumes 1mm isotropic voxels (true for the 84 held-out eval cases at 256^3).
If you run this on data with different spacing, pass a `spacing` tuple.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, label

# ---------------------------------------------------------------------------
# Connected-component lesion matching
# ---------------------------------------------------------------------------

def match_lesions(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    iou_threshold: float = 0.1,
) -> dict:
    """Greedy IoU-based matching between predicted and GT connected components.

    Returns dict with:
        pred_labeled, gt_labeled  -- cc-labeled arrays
        n_pred, n_gt              -- total components
        matched                   -- list of (pred_idx, gt_idx) pairs (1-indexed)
        unmatched_pred            -- set of pred component ids with no GT match
        unmatched_gt              -- set of GT component ids with no pred match
    """
    pred_labeled, n_pred = label(pred_bin.astype(bool))
    gt_labeled, n_gt = label(gt_bin.astype(bool))

    # Collect all non-zero overlaps
    candidates: list[tuple[int, int, float]] = []
    for p in range(1, n_pred + 1):
        pred_mask = (pred_labeled == p)
        overlapping_gt = np.unique(gt_labeled[pred_mask])
        overlapping_gt = overlapping_gt[overlapping_gt > 0]
        for g in overlapping_gt:
            gt_mask = (gt_labeled == g)
            inter = int((pred_mask & gt_mask).sum())
            union = int(pred_mask.sum() + gt_mask.sum() - inter)
            iou = inter / union if union > 0 else 0.0
            if iou >= iou_threshold:
                candidates.append((p, int(g), float(iou)))

    # Greedy assignment -- highest IoU first
    candidates.sort(key=lambda x: -x[2])
    matched_pred: set = set()
    matched_gt: set = set()
    pairs: list[tuple[int, int]] = []
    for p, g, _ in candidates:
        if p not in matched_pred and g not in matched_gt:
            pairs.append((p, g))
            matched_pred.add(p)
            matched_gt.add(g)

    return {
        "pred_labeled": pred_labeled,
        "gt_labeled": gt_labeled,
        "n_pred": int(n_pred),
        "n_gt": int(n_gt),
        "matched": pairs,
        "unmatched_pred": set(range(1, n_pred + 1)) - matched_pred,
        "unmatched_gt": set(range(1, n_gt + 1)) - matched_gt,
    }


# ---------------------------------------------------------------------------
# Lesion-wise detection metrics
# ---------------------------------------------------------------------------

def lesion_wise_sensitivity(pred_bin: np.ndarray, gt_bin: np.ndarray,
                             iou_threshold: float = 0.1) -> float:
    """Fraction of GT lesions correctly detected. Neosoma target: >= 0.85."""
    m = match_lesions(pred_bin, gt_bin, iou_threshold)
    if m["n_gt"] == 0:
        return 1.0
    return len(m["matched"]) / m["n_gt"]


def false_positives_per_case(pred_bin: np.ndarray, gt_bin: np.ndarray,
                              iou_threshold: float = 0.1) -> int:
    """Number of predicted components that did not match any GT lesion."""
    m = match_lesions(pred_bin, gt_bin, iou_threshold)
    return len(m["unmatched_pred"])


def lesion_wise_dice_matched(pred_bin: np.ndarray, gt_bin: np.ndarray,
                              iou_threshold: float = 0.1) -> float:
    """Avg Dice over matched lesion pairs ONLY (Neosoma reporting style).

    Unmatched GT and unmatched pred lesions are not penalized here (they
    show up in sensitivity and FPR instead). This is how Neosoma achieved
    0.86 -- matched-only averaging.
    """
    m = match_lesions(pred_bin, gt_bin, iou_threshold)
    if not m["matched"]:
        return float("nan")
    dices = []
    for p_idx, g_idx in m["matched"]:
        pred_mask = (m["pred_labeled"] == p_idx)
        gt_mask = (m["gt_labeled"] == g_idx)
        inter = int((pred_mask & gt_mask).sum())
        denom = int(pred_mask.sum() + gt_mask.sum())
        dices.append(2.0 * inter / denom if denom > 0 else 0.0)
    return float(np.mean(dices))


def lesion_wise_dice_brats(pred_bin: np.ndarray, gt_bin: np.ndarray,
                            iou_threshold: float = 0.1) -> float:
    """BraTS-style: unmatched GT lesions scored 0, unmatched pred ignored.

    Stricter than matched-only; penalizes missed lesions directly.
    """
    m = match_lesions(pred_bin, gt_bin, iou_threshold)
    if m["n_gt"] == 0:
        return 1.0
    dices = []
    for p_idx, g_idx in m["matched"]:
        pred_mask = (m["pred_labeled"] == p_idx)
        gt_mask = (m["gt_labeled"] == g_idx)
        inter = int((pred_mask & gt_mask).sum())
        denom = int(pred_mask.sum() + gt_mask.sum())
        dices.append(2.0 * inter / denom if denom > 0 else 0.0)
    dices.extend([0.0] * len(m["unmatched_gt"]))
    return float(np.mean(dices))


def small_lesion_sensitivity(pred_bin: np.ndarray, gt_bin: np.ndarray,
                              max_voxels: int = 100,
                              iou_threshold: float = 0.1) -> tuple[float, int]:
    """Detection rate on small lesions only (<100 mm^3 at 1mm iso ~ <5mm dia).

    Returns (sensitivity, n_small_lesions).
    """
    m = match_lesions(pred_bin, gt_bin, iou_threshold)
    small_ids = [g for g in range(1, m["n_gt"] + 1)
                 if int((m["gt_labeled"] == g).sum()) < max_voxels]
    if not small_ids:
        return float("nan"), 0
    matched_gt = {g for _, g in m["matched"]}
    hits = sum(1 for g in small_ids if g in matched_gt)
    return hits / len(small_ids), len(small_ids)


# ---------------------------------------------------------------------------
# Boundary-distance metrics
# ---------------------------------------------------------------------------

def _surface_mask(binary_mask: np.ndarray) -> np.ndarray:
    """Extract surface voxels of a 3D binary mask (mask AND NOT eroded(mask))."""
    mask_bool = binary_mask.astype(bool)
    eroded = binary_erosion(mask_bool)
    return mask_bool & ~eroded


def _surface_distances(pred_bin: np.ndarray, gt_bin: np.ndarray,
                       spacing: float | Sequence[float] = 1.0) -> np.ndarray:
    """Bidirectional distances between pred and GT surfaces (in mm).

    Returns a single 1D array of all pred->gt and gt->pred surface distances.
    Empty array if either surface is empty.
    """
    pred_surf = _surface_mask(pred_bin)
    gt_surf = _surface_mask(gt_bin)
    if not pred_surf.any() or not gt_surf.any():
        return np.array([], dtype=np.float32)

    # Distance from each voxel to nearest GT surface voxel, then sample at pred surf
    dt_from_gt = distance_transform_edt(~gt_surf, sampling=spacing)
    dt_from_pred = distance_transform_edt(~pred_surf, sampling=spacing)
    d_pred_to_gt = dt_from_gt[pred_surf]
    d_gt_to_pred = dt_from_pred[gt_surf]
    return np.concatenate([d_pred_to_gt, d_gt_to_pred]).astype(np.float32)


def hd95(pred_bin: np.ndarray, gt_bin: np.ndarray,
         spacing: float | Sequence[float] = 1.0) -> float:
    """95th percentile bidirectional surface distance (mm).

    Neosoma achieved 1.78 mm (1.02-2.54). FDA threshold: <= 2.94 mm.
    Returns NaN if either mask is empty.
    """
    d = _surface_distances(pred_bin, gt_bin, spacing)
    if len(d) == 0:
        return float("nan")
    return float(np.percentile(d, 95))


def mean_surface_distance(pred_bin: np.ndarray, gt_bin: np.ndarray,
                           spacing: float | Sequence[float] = 1.0) -> float:
    """Average bidirectional surface distance (mm).

    Neosoma achieved 0.36 mm (0.16-0.56). FDA threshold: <= 0.66 mm.
    """
    d = _surface_distances(pred_bin, gt_bin, spacing)
    if len(d) == 0:
        return float("nan")
    return float(d.mean())


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(per_case_values: Sequence[float],
                 n_bootstrap: int = 1000,
                 ci: float = 0.95,
                 seed: int = 42) -> tuple[float, float]:
    """Bootstrap 95% CI for the mean of a per-case metric.

    FDA-required reporting format: point estimate + 1000-resample bootstrap.
    NaNs are dropped before resampling.
    """
    vals = np.asarray([v for v in per_case_values
                       if v is not None and not (isinstance(v, float) and np.isnan(v))],
                      dtype=np.float64)
    if len(vals) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    n = len(vals)
    for i in range(n_bootstrap):
        means[i] = rng.choice(vals, n, replace=True).mean()
    low = float(np.percentile(means, (1 - ci) / 2 * 100))
    high = float(np.percentile(means, (1 + ci) / 2 * 100))
    return low, high


# ---------------------------------------------------------------------------
# Batch compute -- all FDA metrics across a list of (pred, gt) cases
# ---------------------------------------------------------------------------

def compute_case_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray,
                         spacing: float | Sequence[float] = 1.0,
                         iou_threshold: float = 0.1,
                         small_lesion_max_voxels: int = 100) -> dict[str, float]:
    """All FDA metrics for a single case. NaNs where undefined."""
    m = match_lesions(pred_bin, gt_bin, iou_threshold)

    # Sensitivity
    if m["n_gt"] == 0:
        sens = float("nan")
    else:
        sens = len(m["matched"]) / m["n_gt"]

    # FPs per case
    fp = len(m["unmatched_pred"])

    # Lesion-wise Dice (both styles)
    if m["matched"]:
        dices_matched = []
        for p_idx, g_idx in m["matched"]:
            pm = (m["pred_labeled"] == p_idx)
            gm = (m["gt_labeled"] == g_idx)
            inter = int((pm & gm).sum())
            denom = int(pm.sum() + gm.sum())
            dices_matched.append(2.0 * inter / denom if denom > 0 else 0.0)
        dice_matched = float(np.mean(dices_matched))
        dice_brats_vals = list(dices_matched) + [0.0] * len(m["unmatched_gt"])
        dice_brats = float(np.mean(dice_brats_vals))
    else:
        dice_matched = float("nan")
        dice_brats = 0.0 if m["n_gt"] > 0 else 1.0

    # Small-lesion sensitivity
    small_ids = [g for g in range(1, m["n_gt"] + 1)
                 if int((m["gt_labeled"] == g).sum()) < small_lesion_max_voxels]
    if not small_ids:
        small_sens = float("nan")
        n_small = 0
    else:
        matched_gt = {g for _, g in m["matched"]}
        hits = sum(1 for g in small_ids if g in matched_gt)
        small_sens = hits / len(small_ids)
        n_small = len(small_ids)

    # Boundary distances
    hd = hd95(pred_bin, gt_bin, spacing=spacing)
    msd = mean_surface_distance(pred_bin, gt_bin, spacing=spacing)

    return {
        "lesion_wise_sensitivity": sens,
        "fp_per_case": float(fp),
        "lesion_wise_dice_matched": dice_matched,
        "lesion_wise_dice_brats": dice_brats,
        "small_lesion_sensitivity": small_sens,
        "n_small_lesions": float(n_small),
        "hd95_mm": hd,
        "msd_mm": msd,
        "n_gt_lesions": float(m["n_gt"]),
        "n_pred_lesions": float(m["n_pred"]),
    }


def aggregate_with_ci(per_case_metrics: list[dict[str, float]],
                      n_bootstrap: int = 1000) -> dict[str, dict[str, float]]:
    """Aggregate per-case metrics into mean + 95% CI per metric."""
    if not per_case_metrics:
        return {}
    keys = list(per_case_metrics[0].keys())
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = [d[k] for d in per_case_metrics]
        vals_clean = np.asarray([v for v in vals
                                  if v is not None and not np.isnan(v)],
                                 dtype=np.float64)
        if len(vals_clean) == 0:
            out[k] = {"mean": float("nan"), "ci_low": float("nan"),
                      "ci_high": float("nan"), "n": 0}
            continue
        low, high = bootstrap_ci(vals_clean, n_bootstrap=n_bootstrap)
        out[k] = {
            "mean": float(vals_clean.mean()),
            "std": float(vals_clean.std()),
            "ci_low": low,
            "ci_high": high,
            "n": len(vals_clean),
        }
    return out


# ---------------------------------------------------------------------------
# Stratified metrics by lesion-size bin
# ---------------------------------------------------------------------------

# Clinical-aligned bins (longest axis in mm). Every model report going forward
# prints these.
DEFAULT_SIZE_BINS = [
    ("<3mm",    (0.0, 3.0)),
    ("3-5mm",   (3.0, 5.0)),
    ("5-10mm",  (5.0, 10.0)),
    ("10-20mm", (10.0, 20.0)),
    (">20mm",   (20.0, float("inf"))),
]


def _longest_axis_mm(mask: np.ndarray,
                     spacing: float | Sequence[float] = 1.0) -> float:
    """Longest bounding-box axis in mm. Standard RANO-BM-style diameter proxy."""
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return 0.0
    extents_vox = coords.max(axis=0) - coords.min(axis=0) + 1
    if isinstance(spacing, (int, float)):
        spacing_t = (float(spacing),) * len(extents_vox)
    else:
        spacing_t = tuple(spacing)
    extents_mm = np.asarray(extents_vox) * np.asarray(spacing_t)
    return float(extents_mm.max())


def _bin_for_diameter(diameter_mm: float,
                      bins: Sequence[tuple[str, tuple[float, float]]]) -> str:
    for bin_label, (lo, hi) in bins:
        if lo <= diameter_mm < hi:
            return bin_label
    return bins[-1][0]


def compute_case_stratified(pred_bin: np.ndarray, gt_bin: np.ndarray,
                            spacing: float | Sequence[float] = 1.0,
                            iou_threshold: float = 0.1,
                            bins: Sequence[tuple[str, tuple[float, float]]] | None = None,
                            ) -> dict[str, dict]:
    """Per-case stratified stats. For each bin:
        - gt_ids, n_gt
        - matched_gt_ids (subset of gt_ids that got matched)
        - matched_dices (Dice of each matched pair whose GT falls in this bin)
        - n_fp_in_bin (unmatched pred components whose OWN size falls in this bin)
    """
    bins = bins or DEFAULT_SIZE_BINS
    m = match_lesions(pred_bin, gt_bin, iou_threshold)
    pred_labeled = m["pred_labeled"]
    gt_labeled = m["gt_labeled"]

    # Classify GTs and preds by size
    gt_bin_for = {}
    for g in range(1, m["n_gt"] + 1):
        comp = (gt_labeled == g)
        diam = _longest_axis_mm(comp, spacing)
        gt_bin_for[g] = _bin_for_diameter(diam, bins)
    pred_bin_for = {}
    for p in range(1, m["n_pred"] + 1):
        comp = (pred_labeled == p)
        diam = _longest_axis_mm(comp, spacing)
        pred_bin_for[p] = _bin_for_diameter(diam, bins)

    matched_gt_ids = {g for _, g in m["matched"]}
    dice_for_match: dict[tuple[int, int], float] = {}
    for p_idx, g_idx in m["matched"]:
        pm = (pred_labeled == p_idx)
        gm = (gt_labeled == g_idx)
        inter = int((pm & gm).sum())
        denom = int(pm.sum() + gm.sum())
        dice_for_match[(p_idx, g_idx)] = 2.0 * inter / denom if denom > 0 else 0.0

    result: dict[str, dict] = {}
    for bin_label, _ in bins:
        gts_in_bin = [g for g, b in gt_bin_for.items() if b == bin_label]
        matched_in_bin = [g for g in gts_in_bin if g in matched_gt_ids]
        dices_in_bin = [
            dice_for_match[(p, g)]
            for p, g in m["matched"]
            if g in gts_in_bin
        ]
        fp_in_bin = [
            p for p in m["unmatched_pred"] if pred_bin_for.get(p) == label
        ]
        result[bin_label] = {
            "n_gt": len(gts_in_bin),
            "n_matched": len(matched_in_bin),
            "matched_dices": dices_in_bin,
            "n_fp_in_bin": len(fp_in_bin),
        }
    return result


def aggregate_stratified(per_case: list[dict[str, dict]],
                          bins: Sequence[tuple[str, tuple[float, float]]] | None = None,
                          ) -> dict[str, dict[str, float]]:
    """Aggregate across cases: sensitivity, FP/case, mean DSC per bin."""
    bins = bins or DEFAULT_SIZE_BINS
    n_cases = max(len(per_case), 1)
    out: dict[str, dict[str, float]] = {}
    for bin_label, _ in bins:
        total_gt = sum(c[bin_label]["n_gt"] for c in per_case)
        total_matched = sum(c[bin_label]["n_matched"] for c in per_case)
        total_fp = sum(c[bin_label]["n_fp_in_bin"] for c in per_case)
        all_dices = []
        for c in per_case:
            all_dices.extend(c[bin_label]["matched_dices"])
        sens = total_matched / total_gt if total_gt > 0 else float("nan")
        fp_per_case = total_fp / n_cases
        mean_dice = float(np.mean(all_dices)) if all_dices else float("nan")
        out[bin_label] = {
            "n_gt_total": total_gt,
            "n_matched_total": total_matched,
            "sensitivity": sens,
            "n_fp_total": total_fp,
            "fp_per_case": fp_per_case,
            "mean_dsc": mean_dice,
        }
    return out


def format_stratified_report(strat: dict[str, dict[str, float]],
                              title: str = "STRATIFIED BY LESION SIZE (longest axis)"
                              ) -> str:
    lines = []
    lines.append("=" * 88)
    lines.append(f"  {title}")
    lines.append("=" * 88)
    lines.append(f"  {'Bin':<10} {'n GT':>8} {'n matched':>12} "
                 f"{'Sens':>10} {'FPs/case':>12} {'Mean DSC':>12}")
    lines.append(f"  {'-' * 82}")
    for bin_label, v in strat.items():
        sens = v["sensitivity"]
        dsc = v["mean_dsc"]
        sens_str = f"{sens:.4f}" if not np.isnan(sens) else "  n/a "
        dsc_str = f"{dsc:.4f}" if not np.isnan(dsc) else "  n/a "
        lines.append(
            f"  {bin_label:<10} {v['n_gt_total']:>8} {v['n_matched_total']:>12} "
            f"{sens_str:>10} {v['fp_per_case']:>12.3f} {dsc_str:>12}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def format_fda_report(agg: dict[str, dict[str, float]]) -> str:
    """Render a Neosoma-style comparison table. Returns multi-line string."""
    # Neosoma K252922 reference values (ASCII-only for Windows cp1252 console)
    neosoma = {
        "lesion_wise_sensitivity": ("0.90 (0.87-0.94)", ">= 0.85"),
        "fp_per_case":             ("0.57 (0.35-0.80)", "<= 5"),
        "lesion_wise_dice_matched":("0.86 (0.83-0.89)", ">= 0.70"),
        "lesion_wise_dice_brats":  ("not reported",     "-"),
        "small_lesion_sensitivity":("not reported",     ">= 0.75 (BMS target)"),
        "hd95_mm":                 ("1.78 (1.02-2.54)", "<= 2.94 mm"),
        "msd_mm":                  ("0.36 (0.16-0.56)", "<= 0.66 mm"),
    }
    display_order = [
        ("Lesion-wise Sensitivity", "lesion_wise_sensitivity"),
        ("False Positives / case",  "fp_per_case"),
        ("Lesion-wise Dice (matched)", "lesion_wise_dice_matched"),
        ("Lesion-wise Dice (BraTS)",   "lesion_wise_dice_brats"),
        ("Small-lesion Sens (<100mm3)", "small_lesion_sensitivity"),
        ("HD95 (mm)", "hd95_mm"),
        ("MSD (mm)",  "msd_mm"),
    ]

    lines = []
    lines.append("=" * 92)
    lines.append("  FDA-STYLE METRICS vs Neosoma Brain Mets (K252922, Dec 2025)")
    lines.append("=" * 92)
    lines.append(f"  {'Metric':<32} {'BMS (mean [95% CI])':<26} "
                 f"{'Neosoma K252922':<22} {'FDA threshold'}")
    lines.append(f"  {'-' * 90}")
    for metric_label, key in display_order:
        if key not in agg:
            continue
        a = agg[key]
        m, lo, hi = a["mean"], a["ci_low"], a["ci_high"]
        bms_str = f"{m:.3f} ({lo:.3f}-{hi:.3f})"
        neo_ach, fda_thresh = neosoma.get(key, ("-", "-"))
        lines.append(f"  {metric_label:<32} {bms_str:<26} {neo_ach:<22} {fda_thresh}")

    # Lesion counts footer
    if "n_gt_lesions" in agg and "n_pred_lesions" in agg:
        n_gt_total = agg["n_gt_lesions"]["mean"] * agg["n_gt_lesions"]["n"]
        n_pred_total = agg["n_pred_lesions"]["mean"] * agg["n_pred_lesions"]["n"]
        n_small_total = (agg["n_small_lesions"]["mean"] * agg["n_small_lesions"]["n"]
                          if "n_small_lesions" in agg else 0)
        lines.append("")
        lines.append(f"  N cases: {agg['n_gt_lesions']['n']}   "
                     f"Total GT lesions: {int(n_gt_total)}   "
                     f"Total pred lesions: {int(n_pred_total)}   "
                     f"Small GT lesions (<100vox): {int(n_small_total)}")
    return "\n".join(lines)
