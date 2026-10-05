"""Pool the SECONDARY 2-fold results out-of-fold (D5).

Each fold is scored with the configuration and per-target τ frozen on the
*other* fold. For each FP target, pooling shifts every fold's scores by its own
τ (``score - tau_fold``) and applies a single threshold of 0 to the
concatenated records. This is exactly "each scan judged by the operating point
its own fold was given". The patient bootstrap then runs over all 84 scans.

The production baseline (0.55, all components) is identical in both folds, so it
is simply concatenated.

    python experiments/lesion_detection/pool_2fold.py --folds experiments/lesion_detection/configs/32_evaluate_foldB.yaml experiments/lesion_detection/configs/33_evaluate_foldA.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from common import REPO, load_yaml, read_ids, repo_path
from evaluate import clean_json, op_block
from select_operating_point import components_dir, size_bins
from tables import Config, detection_data, load_tables

from jannus.evaluation.detection.metrics import DetectionData
from jannus.evaluation.detection.provenance import build_provenance, split_record, write_json


def shifted(data: DetectionData, tau: float) -> DetectionData:
    return DetectionData(case_ids=data.case_ids, patient_ids=data.patient_ids, gt_case=data.gt_case,
                         gt_score=data.gt_score - tau, gt_bin=data.gt_bin,
                         fp_case=data.fp_case, fp_score=data.fp_score - tau)


def concat(parts: list[DetectionData]) -> DetectionData:
    offset, gc, fc = 0, [], []
    for p in parts:
        gc.append(p.gt_case + offset)
        fc.append(p.fp_case + offset)
        offset += p.n_cases
    return DetectionData(
        case_ids=sum((p.case_ids for p in parts), ()), patient_ids=sum((p.patient_ids for p in parts), ()),
        gt_case=np.concatenate(gc), gt_score=np.concatenate([p.gt_score for p in parts]),
        gt_bin=np.concatenate([p.gt_bin for p in parts]), fp_case=np.concatenate(fc),
        fp_score=np.concatenate([p.fp_score for p in parts]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs=2, required=True, type=Path)
    ap.add_argument("--out", default="experiments/lesion_detection/results/secondary_2fold_pooled")
    args = ap.parse_args()

    per_target: dict[str, list[DetectionData]] = {}
    baseline_parts, splits, cfgs = [], [], []
    for path in args.folds:
        cfg = load_yaml(path)
        cfgs.append(cfg)
        frozen = load_yaml(repo_path(cfg["frozen"]))
        test = read_ids(cfg["test_split"])
        if set(test) & set(read_ids(frozen["selected_on"])):
            raise SystemExit(f"{path}: test fold overlaps its tuning fold")
        splits.append(split_record(cfg["name"], repo_path(cfg["test_split"]), test))
        tables = load_tables(components_dir(cfg["components_config"]), test)
        p = frozen["primary"]["config"]
        sel = Config(p["connectivity"], p["voxel_threshold"], p["rule"], p["scorer"], p["min_esd_mm"])
        d = detection_data(tables, test, sel)
        for t, v in frozen["primary"]["frozen_tau_by_fp_target"].items():
            per_target.setdefault(t, []).append(shifted(d, float(v["tau"])))
        b = cfg["baseline"]
        baseline_parts.append(detection_data(tables, test, Config(b["connectivity"], b["voxel_threshold"],
                                                                  b["rule"], b["scorer"], b["min_esd_mm"])))
    if len(set().union(*(s["case_ids"] for s in splits))) != sum(s["n_cases"] for s in splits):
        raise SystemExit("folds overlap")

    bins = size_bins(cfgs[0]["components_config"])
    bcfg = cfgs[0]["bootstrap"]
    result = {
        "folds": [c["name"] for c in cfgs],
        "baseline_operating_point": op_block(concat(baseline_parts), -np.inf, bins, bcfg),
        "pooled_frozen_operating_points": {t: op_block(concat(parts), 0.0, bins, bcfg)
                                           for t, parts in per_target.items()},
    }
    out = repo_path(args.out)
    write_json(out / "metrics.json", clean_json(result))
    write_json(out / "provenance.json", build_provenance(
        command="pool_2fold", repo_root=REPO, config={"folds": [str(f) for f in args.folds]},
        config_path=";".join(str(f) for f in args.folds), splits=splits))
    for t, r in result["pooled_frozen_operating_points"].items():
        print(f"pooled @ frozen {t} FP: sens {r['point']['sensitivity']:.3f} fp/scan {r['point']['fp_per_scan']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
