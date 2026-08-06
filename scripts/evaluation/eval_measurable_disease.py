"""
RANO-BM measurable-disease evaluation.

Per RANO-BM, a lesion is "measurable" if its longest diameter is >= 10mm.
This eval restricts to the measurable-disease subset and computes both
standard FDA metrics and RANO-BM-style metrics:
  - measurable-only FDA metrics (lesion sens, FPR, Dice, HD95, MSD)
  - measurable-disease classification accuracy (per-lesion binary)
  - Sum of Longest Diameters (SoLD) absolute and relative error

Longest diameter is measured as the longest axis of each connected
component's bounding box, at 1mm isotropic spacing.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/eval_measurable_disease.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from scipy.ndimage import label as ndimage_label

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

from jannus.segmentation.stacking import (
    StackingClassifierV2, postprocess_prediction, sliding_window_inference,
)
from jannus.evaluation.metrics import (
    aggregate_with_ci, compute_case_metrics, format_fda_report, match_lesions,
)
from scripts.evaluation.full_stacking_comparison_v2 import (
    OLD_PER_CASE_PATH, voxel_dice, voxel_sens, voxel_prec,
)
from scripts.evaluation.seven_model_stacker import (
    load_seven_preds, build_features_9ch, IN_CHANNELS,
    STACKING_PATCH, STACKING_OVERLAP,
)

CHECKPOINT_PATH = PROJECT / "model" / "stacking_classifier_production.pth"
OUTPUT_PATH = PROJECT / "model" / "evaluation_results" / "measurable_disease_eval.json"

THRESHOLD = 0.55
MEASURABLE_DIAMETER_MM = 10.0   # RANO-BM "measurable" floor


def longest_diameter_mm(component_mask: np.ndarray,
                        spacing: tuple = (1.0, 1.0, 1.0)) -> float:
    """Longest axis of the bounding box (mm). Fast proxy for RANO-BM longest
    diameter -- true longest chord requires convex-hull max distance, which
    is much slower. Bounding-box diagonal is an upper bound; bounding-box
    LONGEST SIDE is the standard axis-aligned measurement that RANO-BM uses
    in practice on axial slices.
    """
    if not component_mask.any():
        return 0.0
    coords = np.argwhere(component_mask)
    extents_vox = coords.max(axis=0) - coords.min(axis=0) + 1
    extents_mm = extents_vox * np.array(spacing)
    return float(extents_mm.max())


def filter_by_measurable(pred_bin: np.ndarray, gt_bin: np.ndarray,
                         min_diameter_mm: float = MEASURABLE_DIAMETER_MM
                         ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Keep only measurable-disease (>= min_diameter_mm longest axis) GT CCs.
    Drop predicted CCs that overlap only with removed non-measurable GT.
    """
    gt_cc, n_gt = ndimage_label(gt_bin)
    pred_cc, n_pred = ndimage_label(pred_bin)

    removed_gt_ids = set()
    gt_filtered = np.zeros_like(gt_bin, dtype=np.uint8)
    for i in range(1, n_gt + 1):
        comp = (gt_cc == i)
        if longest_diameter_mm(comp) < min_diameter_mm:
            removed_gt_ids.add(i)
        else:
            gt_filtered[comp] = 1

    kept_gt = n_gt - len(removed_gt_ids)
    removed_pred = 0
    pred_filtered = np.zeros_like(pred_bin, dtype=np.uint8)
    for i in range(1, n_pred + 1):
        comp = (pred_cc == i)
        overlapping_gt = np.unique(gt_cc[comp])
        overlapping_gt = overlapping_gt[overlapping_gt > 0]
        if len(overlapping_gt) > 0 and all(g in removed_gt_ids for g in overlapping_gt):
            removed_pred += 1
            continue
        pred_filtered[comp] = 1

    return pred_filtered, gt_filtered, {
        "removed_gt": len(removed_gt_ids),
        "removed_pred": removed_pred,
        "kept_gt": kept_gt,
    }


def measurable_classification_stats(pred_bin: np.ndarray, gt_bin: np.ndarray,
                                    min_mm: float = MEASURABLE_DIAMETER_MM):
    """Per-lesion 'is measurable' classification accuracy.

    For each matched GT-pred pair, record:
      - GT measurable? (longest axis >= min_mm)
      - Pred measurable? (longest axis >= min_mm)
    Returns a 2x2 confusion dict plus counts.
    """
    m = match_lesions(pred_bin, gt_bin, iou_threshold=0.1)
    tp_tp = tp_tn = tn_tp = tn_tn = 0  # GT-measurable x pred-measurable
    sold_gt = 0.0
    sold_pred = 0.0
    n_matched_meas = 0
    for p_idx, g_idx in m["matched"]:
        pred_mask = (m["pred_labeled"] == p_idx)
        gt_mask = (m["gt_labeled"] == g_idx)
        gt_d = longest_diameter_mm(gt_mask)
        pd_d = longest_diameter_mm(pred_mask)
        gt_meas = gt_d >= min_mm
        pd_meas = pd_d >= min_mm
        if gt_meas and pd_meas:   tp_tp += 1
        elif gt_meas and not pd_meas: tp_tn += 1
        elif (not gt_meas) and pd_meas: tn_tp += 1
        else:                      tn_tn += 1
        if gt_meas:
            sold_gt += gt_d
            sold_pred += pd_d   # corresponding pred diameter (may be smaller)
            n_matched_meas += 1
    return {
        "both_measurable": tp_tp,
        "gt_meas_only": tp_tn,
        "pred_meas_only": tn_tp,
        "both_non_measurable": tn_tn,
        "sold_gt_matched_mm": sold_gt,
        "sold_pred_matched_mm": sold_pred,
        "n_matched_measurable_gt": n_matched_meas,
    }


def main():
    with open(OLD_PER_CASE_PATH) as f:
        eval_ids = sorted(json.load(f).keys())
    print(f"Eval cases: {len(eval_ids)}", flush=True)
    print(f"Measurable disease threshold: >= {MEASURABLE_DIAMETER_MM}mm "
          f"longest axis\n", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StackingClassifierV2(in_channels=IN_CHANNELS).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {CHECKPOINT_PATH} (epoch {ckpt.get('epoch')}, "
          f"val {ckpt.get('val_loss'):.4f})", flush=True)

    # Inference pass
    print(f"\nRunning inference on {len(eval_ids)} cases...", flush=True)
    cases = {}
    t0 = time.time()
    for i, cid in enumerate(eval_ids):
        preds, gt = load_seven_preds(cid)
        features = build_features_9ch(preds)
        prob = sliding_window_inference(model, features, STACKING_PATCH, device,
                                        overlap=STACKING_OVERLAP)
        pred_bin = postprocess_prediction((prob > THRESHOLD).astype(np.float32),
                                          min_size=0).astype(np.uint8)
        cases[cid] = (pred_bin, (gt > 0.5).astype(np.uint8))
        if (i + 1) % 20 == 0 or i + 1 == len(eval_ids):
            print(f"  [{i+1}/{len(eval_ids)}] ({time.time()-t0:.0f}s)", flush=True)

    # Filter to measurable GT + matching preds
    dices, senss, precs = [], [], []
    fda_per_case = []
    agg_cls = {
        "both_measurable": 0, "gt_meas_only": 0,
        "pred_meas_only": 0, "both_non_measurable": 0,
        "sold_gt_matched_mm": 0.0, "sold_pred_matched_mm": 0.0,
        "n_matched_measurable_gt": 0,
    }
    total_removed_gt = total_removed_pred = total_kept_gt = 0

    print("\nFiltering to measurable disease + computing metrics...", flush=True)
    for cid, (pred_bin, gt_bin) in cases.items():
        # For FDA metrics: use filtered (measurable GT only)
        pf, gf, stats = filter_by_measurable(pred_bin, gt_bin,
                                             MEASURABLE_DIAMETER_MM)
        total_removed_gt += stats["removed_gt"]
        total_removed_pred += stats["removed_pred"]
        total_kept_gt += stats["kept_gt"]
        dices.append(voxel_dice(pf, gf))
        senss.append(voxel_sens(pf, gf))
        precs.append(voxel_prec(pf, gf))
        fda_per_case.append(compute_case_metrics(pf, gf, spacing=1.0))

        # For classification stats: use ORIGINAL (before filtering) so we can
        # characterize how the model measures the measurable ones.
        cls = measurable_classification_stats(pred_bin, gt_bin,
                                              MEASURABLE_DIAMETER_MM)
        for k, v in cls.items():
            agg_cls[k] += v

    agg = aggregate_with_ci(fda_per_case, n_bootstrap=1000)

    print(f"\n{'=' * 90}")
    print(f"  RANO-BM MEASURABLE DISEASE EVAL  (>= {MEASURABLE_DIAMETER_MM}mm)")
    print(f"{'=' * 90}")
    print(f"  Total GT components: removed {total_removed_gt}, kept {total_kept_gt}")
    print(f"  Pred components dropped as don't-care (small-only matches): "
          f"{total_removed_pred}")
    print(f"\n  Overall voxel Dice: {np.mean(dices):.4f}")
    print(f"  Overall sens: {np.mean(senss):.4f}   prec: {np.mean(precs):.4f}")
    print(format_fda_report(agg))

    # Measurable-classification matrix (on matched lesions)
    print(f"\n{'=' * 90}")
    print(f"  MEASURABLE-DISEASE CLASSIFICATION  (per-matched-lesion)")
    print(f"{'=' * 90}")
    tp_tp = agg_cls["both_measurable"]
    tp_tn = agg_cls["gt_meas_only"]
    tn_tp = agg_cls["pred_meas_only"]
    tn_tn = agg_cls["both_non_measurable"]
    total_matched = tp_tp + tp_tn + tn_tp + tn_tn
    print(f"  Both measurable (>=10mm each):  {tp_tp}")
    print(f"  GT measurable, pred not:        {tp_tn}  (pred under-measured)")
    print(f"  Pred measurable, GT not:        {tn_tp}  (pred over-measured)")
    print(f"  Both non-measurable:            {tn_tn}")
    print(f"  Total matched: {total_matched}")
    if total_matched > 0:
        acc = (tp_tp + tn_tn) / total_matched
        print(f"  Measurable-classification accuracy (matched pairs): {acc:.3f}")
        # Cohen's kappa for measurable/non
        po = acc
        p_gt = (tp_tp + tp_tn) / total_matched
        p_pd = (tp_tp + tn_tp) / total_matched
        pe = p_gt * p_pd + (1 - p_gt) * (1 - p_pd)
        kappa = (po - pe) / (1 - pe + 1e-9)
        print(f"  Cohen's kappa: {kappa:.3f}  (RANO-BM target >= 0.85)")

    # SoLD error
    print(f"\n{'=' * 90}")
    print(f"  SUM OF LONGEST DIAMETERS (SoLD) ERROR")
    print(f"{'=' * 90}")
    sold_gt = agg_cls["sold_gt_matched_mm"]
    sold_pd = agg_cls["sold_pred_matched_mm"]
    sold_err = abs(sold_pd - sold_gt)
    sold_err_rel = sold_err / max(sold_gt, 1e-6) * 100
    print(f"  SoLD (GT measurable, matched):   {sold_gt:.2f} mm")
    print(f"  SoLD (pred for those lesions):   {sold_pd:.2f} mm")
    print(f"  Absolute error: {sold_err:.2f} mm")
    print(f"  Relative error: {sold_err_rel:.2f} %   (RANO-BM target: <= 10%)")
    print(f"  N matched measurable GT lesions: {agg_cls['n_matched_measurable_gt']}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "description": "RANO-BM measurable-disease (>=10mm longest axis) eval "
                           "of tuned 7-model V2 stacker on 84 eval cases.",
            "checkpoint": str(CHECKPOINT_PATH),
            "threshold": THRESHOLD,
            "measurable_mm": MEASURABLE_DIAMETER_MM,
            "counts": {
                "total_gt_removed": total_removed_gt,
                "total_gt_kept": total_kept_gt,
                "total_pred_dropped_dont_care": total_removed_pred,
            },
            "voxel_metrics": {
                "dice": float(np.mean(dices)),
                "sens": float(np.mean(senss)),
                "prec": float(np.mean(precs)),
            },
            "fda_metrics": agg,
            "measurable_classification": {
                **agg_cls,
                "total_matched": total_matched,
                "accuracy": (tp_tp + tn_tn) / max(total_matched, 1),
            },
            "sold": {
                "sold_gt_mm": sold_gt,
                "sold_pred_mm": sold_pd,
                "absolute_error_mm": sold_err,
                "relative_error_pct": sold_err_rel,
                "n_matched_measurable_gt": agg_cls["n_matched_measurable_gt"],
            },
            "eval_cases": eval_ids,
        }, f, indent=2, default=str)
    print(f"\n  Results -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
