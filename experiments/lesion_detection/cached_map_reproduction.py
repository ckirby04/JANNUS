"""Cached-map reproduction check: does the unchanged stacker reproduce the published
figures when it is fed the base maps it was trained on?

The pre-commitments are in EXPERIMENT_LOG.md (2026-10-05) and ``configs/50_cached_map_reproduction.yaml``.

- The production stacker (`stacking_classifier_production.pth`) is unchanged.
- Six base maps come from ``stacking_cache_v5``: nnU-Net 3D/2D and the patch models.
  Only those model keys are read; ``mask`` is never accessed.
- SwinUNETR is generated fresh with the unchanged adapter, because that cache has no Swin key.
- Settings are those of ``seven_model_stacker.py:evaluate()``:
  ``postprocess_prediction((p > 0.55), 0)``, ``voxel_dice``, ``compute_case_metrics(spacing=1.0)``.
- The gate is judged on all 84 scans. Clean and contaminated strata are reported only.

The result is reported as-is, and nothing is selected from it.

    python experiments/lesion_detection/cached_map_reproduction.py --config experiments/lesion_detection/configs/50_cached_map_reproduction.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from common import REPO, load_yaml, read_ids, repo_path

from jannus.core.checksums import sha256_file
from jannus.core.config import load_config
from jannus.data.layout import scan_dataset
from jannus.data.loading import load_case
from jannus.evaluation.detection.provenance import build_provenance, split_record, write_json
from jannus.evaluation.metrics import compute_case_metrics
from jannus.segmentation.model_adapters import build_adapter
from jannus.segmentation.stacking import (
    STACKING_MODEL_NAMES,
    build_stacking_features_from_preds,
    load_stacking_model,
    postprocess_prediction,
    sliding_window_inference,
)

sys.path.insert(0, str(REPO))
from scripts.evaluation.full_stacking_comparison_v2 import voxel_dice


def aggregate(rows: list[dict]) -> dict:
    return {
        "n_cases": len(rows),
        "voxel_dice_mean": float(np.nanmean([r["voxel_dice"] for r in rows])),
        "lesion_sensitivity_mean": float(np.nanmean([r["lesion_sensitivity"] for r in rows])),
        "fp_total": int(sum(r["fp"] for r in rows)),
        "fp_per_case_mean": float(np.mean([r["fp"] for r in rows])),
        "lesion_dice_matched_mean": float(np.nanmean([r["lesion_dice_matched"] for r in rows])),
        "n_gt_lesions_total": int(sum(r["n_gt_lesions"] for r in rows)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    assert cfg["comparison"] == "strict_greater" and not set(cfg["cache_keys"].values()) & set(cfg["forbidden_keys"])

    cases = read_ids(cfg["cases"])
    strata = {k: set(read_ids(v)) for k, v in cfg["strata"].items()}
    models_cfg = load_config(REPO / "configs/models.yaml")
    device = torch.device("cuda")
    swin = build_adapter({m["name"]: m for m in models_cfg.models}["swin_unetr"], device=device)
    stacker = load_stacking_model(model_dir=REPO / "model", device=device,
                                  in_channels=models_cfg.stacking.in_channels,
                                  checkpoint_name=Path(models_cfg.stacking.classifier_path).name)
    index = {c.case_id: c for c in scan_dataset(cfg["gt_root"], models_cfg.data).cases}

    rows, inputs = [], {}
    cache = Path(cfg["cache_v5"])
    for i, case_id in enumerate(cases, start=1):
        path = cache / f"{case_id}.npz"
        inputs[str(path)] = sha256_file(path)
        with np.load(path) as z:  # model keys only; 'mask' is never read
            cached = {name: z[key].astype(np.float32) for name, key in cfg["cache_keys"].items()}
        vol = torch.from_numpy(load_case(index[case_id], models_cfg.data.sequences).array)
        cached["swin_unetr"] = swin.predict_crop(vol).numpy().astype(np.float32)
        preds = np.stack([cached[n] for n in STACKING_MODEL_NAMES], axis=0)
        prob = sliding_window_inference(stacker, build_stacking_features_from_preds(preds),
                                        models_cfg.stacking.patch_size, device,
                                        overlap=models_cfg.stacking.overlap)
        gt = (np.asarray(nib.load(Path(cfg["gt_root"]) / case_id / "seg.nii.gz").dataobj) > 0.5).astype(np.float32)
        pred = postprocess_prediction((prob > float(cfg["threshold"])).astype(np.float32), cfg["min_component_size"])
        m = compute_case_metrics(pred, gt, spacing=1.0)
        rows.append({"case": case_id, "stratum": next(k for k, v in strata.items() if case_id in v),
                     "voxel_dice": voxel_dice(pred, gt), "lesion_sensitivity": m["lesion_wise_sensitivity"],
                     "fp": m["fp_per_case"], "lesion_dice_matched": m["lesion_wise_dice_matched"],
                     "n_gt_lesions": m["n_gt_lesions"], "n_pred_lesions": m["n_pred_lesions"]})
        print(f"[{i}/{len(cases)}] done", flush=True)  # per-case values are not printed

    overall = aggregate(rows)
    checks = {k: {"logged": cfg["logged"][k], "observed": overall[k], "delta": overall[k] - cfg["logged"][k],
                  "tolerance": tol, "pass": abs(overall[k] - cfg["logged"][k]) <= tol}
              for k, tol in cfg["tolerance"].items()}
    result = {"passed": all(c["pass"] for c in checks.values()), "checks": checks, "overall": overall,
              "by_stratum": {k: aggregate([r for r in rows if r["stratum"] == k]) for k in strata},
              "swin_source": cfg["swin_source"]}
    out = repo_path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_case.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_json(out / "metrics.json", result)
    write_json(out / "provenance.json", build_provenance(
        command="cached_map_reproduction", repo_root=REPO, config=cfg, config_path=args.config,
        splits=[split_record("eval84", repo_path(cfg["cases"]), cases)], inputs=inputs))
    for k, c in checks.items():
        print(f"{'PASS' if c['pass'] else 'FAIL'}  {k:<26} logged {c['logged']:<8} observed {c['observed']:.4f}  "
              f"delta {c['delta']:+.4f}  tol {c['tolerance']}")
    print("CACHED-MAP GATE PASSED" if result["passed"] else "CACHED-MAP GATE FAILED: report as-is, no debugging")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
