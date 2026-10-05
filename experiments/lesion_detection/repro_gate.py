"""Reproduction gate: do the regenerated maps reproduce the published 84-case figures?

The check runs before any detection analysis (DECISIONS D8). It scores the cached
stacker maps exactly as ``scripts/evaluation/seven_model_stacker.py:evaluate()``
did:

- ``postprocess_prediction((p > 0.55).astype(float32), 0)``;
- that script's ``voxel_dice``;
- the existing ``compute_case_metrics`` at ``spacing=1.0``.

It compares the results against the logged aggregates. The gate selects and tunes
nothing. On failure it exits non-zero; the run is then stopped and reported, and
nothing is debugged against these labels.

    python experiments/lesion_detection/repro_gate.py --config experiments/lesion_detection/configs/00_repro_gate.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from jannus.core.checksums import sha256_file  # noqa: E402
from jannus.evaluation.detection.provenance import (  # noqa: E402
    build_provenance,
    split_record,
    write_json,
)
from jannus.evaluation.metrics import compute_case_metrics  # noqa: E402
from jannus.segmentation.stacking import postprocess_prediction  # noqa: E402
from scripts.evaluation.full_stacking_comparison_v2 import voxel_dice  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    split_file = REPO / cfg["cases"]
    case_ids = [ln.strip() for ln in split_file.read_text().splitlines() if ln.strip()]
    manifest = {}
    for line in (REPO / cfg["maps_manifest"]).read_text().splitlines():
        rec = json.loads(line)
        manifest[rec["case"]] = rec  # last write wins if a case was re-run
    missing = [c for c in case_ids if c not in manifest]
    if missing:
        raise SystemExit(f"{len(missing)} case(s) have no cached map yet: {missing[:5]}")
    if cfg["comparison"] != "strict_greater":
        raise SystemExit("the gate must use the original script's p > t comparison")

    t = float(cfg["threshold"])
    rows, inputs = [], {}
    for case_id in case_ids:
        rec = manifest[case_id]
        path = REPO / rec["file"]
        digest = sha256_file(path)
        if digest != rec["sha256"]:
            raise SystemExit(f"{case_id}: cached map hash differs from manifest")
        inputs[rec["file"]] = digest
        p = np.load(path)["stacker"].astype(np.float32)
        gt = (np.asarray(nib.load(Path(cfg["gt_root"]) / case_id / "seg.nii.gz").dataobj) > 0.5)
        gt = gt.astype(np.float32)
        pred = postprocess_prediction((p > t).astype(np.float32), cfg["min_component_size"])
        m = compute_case_metrics(pred, gt, spacing=1.0)
        rows.append({
            "case": case_id,
            "voxel_dice": voxel_dice(pred, gt),
            "lesion_sensitivity": m["lesion_wise_sensitivity"],
            "fp": m["fp_per_case"],
            "lesion_dice_matched": m["lesion_wise_dice_matched"],
            "n_gt_lesions": m["n_gt_lesions"],
            "n_pred_lesions": m["n_pred_lesions"],
        })
        print(f"{case_id}: dice {rows[-1]['voxel_dice']:.4f}", flush=True)

    def nanmean(key: str) -> float:
        return float(np.nanmean([r[key] for r in rows]))

    observed = {
        "voxel_dice_mean": nanmean("voxel_dice"),
        "lesion_sensitivity_mean": nanmean("lesion_sensitivity"),
        "fp_per_case_mean": nanmean("fp"),
        "fp_total": int(sum(r["fp"] for r in rows)),
        "lesion_dice_matched_mean": nanmean("lesion_dice_matched"),
    }
    checks = {}
    for key, tol in cfg["tolerance"].items():
        delta = observed[key] - cfg["logged"][key]
        checks[key] = {
            "logged": cfg["logged"][key],
            "observed": observed[key],
            "delta": delta,
            "tolerance": tol,
            "pass": abs(delta) <= tol,
        }
    passed = all(c["pass"] for c in checks.values())

    out = REPO / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_case.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_json(out / "metrics.json", {"passed": passed, "checks": checks, "observed": observed,
                                      "n_cases": len(rows)})
    write_json(out / "provenance.json", build_provenance(
        command="repro_gate", repo_root=REPO, config=cfg, config_path=args.config,
        splits=[split_record("eval84", split_file, case_ids)], inputs=inputs,
    ))
    for key, c in checks.items():
        print(f"{'PASS' if c['pass'] else 'FAIL'}  {key:<26} logged {c['logged']:<8} "
              f"observed {c['observed']:.4f}  delta {c['delta']:+.4f}  tol {c['tolerance']}")
    print("GATE PASSED" if passed else "GATE FAILED: stop and report; do not debug on these labels")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
