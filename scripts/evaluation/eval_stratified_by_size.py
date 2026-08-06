"""
Standalone stratified-by-size evaluation of the tuned 7-model V2 stacker.

Reports sensitivity, FP/case, mean DSC for each lesion-size bin:
    <3mm, 3-5mm, 5-10mm, 10-20mm, >20mm
Lesion size = longest bounding-box axis at 1mm isotropic.

Same inference as the main eval, just adds the stratified breakdown on top.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/eval_stratified_by_size.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

from jannus.segmentation.stacking import (
    StackingClassifierV2, postprocess_prediction, sliding_window_inference,
)
from jannus.evaluation.metrics import (
    aggregate_stratified, compute_case_stratified, format_stratified_report,
    aggregate_with_ci, compute_case_metrics, format_fda_report,
)
from scripts.evaluation.full_stacking_comparison_v2 import OLD_PER_CASE_PATH
from scripts.evaluation.seven_model_stacker import (
    load_seven_preds, build_features_9ch, IN_CHANNELS,
    STACKING_PATCH, STACKING_OVERLAP,
)

CHECKPOINT_PATH = PROJECT / "model" / "stacking_classifier_production.pth"
OUTPUT_PATH = PROJECT / "model" / "evaluation_results" / "stratified_by_size_eval.json"

THRESHOLD = 0.55


def main():
    with open(OLD_PER_CASE_PATH) as f:
        eval_ids = sorted(json.load(f).keys())
    print(f"Eval cases: {len(eval_ids)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StackingClassifierV2(in_channels=IN_CHANNELS).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {CHECKPOINT_PATH} (val {ckpt.get('val_loss'):.4f})", flush=True)
    print(f"Threshold: {THRESHOLD}", flush=True)

    strat_per_case = []
    fda_per_case = []
    t0 = time.time()
    for i, cid in enumerate(eval_ids):
        preds, gt = load_seven_preds(cid)
        features = build_features_9ch(preds)
        prob = sliding_window_inference(model, features, STACKING_PATCH, device,
                                        overlap=STACKING_OVERLAP)
        pred_bin = postprocess_prediction((prob > THRESHOLD).astype(np.float32),
                                          min_size=0).astype(np.uint8)
        gt_bin = (gt > 0.5).astype(np.uint8)
        strat_per_case.append(compute_case_stratified(pred_bin, gt_bin, spacing=1.0))
        fda_per_case.append(compute_case_metrics(pred_bin, gt_bin, spacing=1.0))
        if (i + 1) % 20 == 0 or i + 1 == len(eval_ids):
            print(f"  [{i+1}/{len(eval_ids)}] ({time.time() - t0:.0f}s)", flush=True)

    strat_agg = aggregate_stratified(strat_per_case)
    fda_agg = aggregate_with_ci(fda_per_case, n_bootstrap=1000)

    print("\n" + format_stratified_report(
        strat_agg, title="STRATIFIED BY LESION SIZE -- tuned 7-model V2"))
    print("\n" + format_fda_report(fda_agg))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "description": "Stratified by longest-axis bin for tuned 7-model V2.",
            "checkpoint": str(CHECKPOINT_PATH),
            "threshold": THRESHOLD,
            "stratified": strat_agg,
            "fda_metrics": fda_agg,
            "eval_cases": eval_ids,
        }, f, indent=2, default=str)
    print(f"\n  Results -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
