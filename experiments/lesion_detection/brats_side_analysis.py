"""Supplementary analysis (D7): how much does annotation convention change detection status?

The analysis covers the eval-cohort UCSF scans that also have the 2023 BraTS-METS
revision. Each scan is scored at the FROZEN operating points against two
ground truths:

  A   the original annotation used for every main number (``preprocessed_256 seg``)
  C3  BraTS-seg label 3 (enhancing tumour), resized to the 256^3 grid with the same
      nearest-neighbour zoom as preprocess_256.py

Lesions in A and C3 are different objects, so they are paired first: an A lesion
and a C3 lesion are paired when they overlap by at least 1 voxel, using 26-connected
components on the 256^3 grid. For each scan the analysis reports:

- lesion counts and detected counts under each annotation;
- ``changed``: overlapping A-C3 lesion pairs whose detection status differs;
- ``only_in_A`` / ``only_in_C3``: lesions with no counterpart, and how many of those were detected.

Nothing is selected here. The operating points come from the frozen file.

    python experiments/lesion_detection/brats_side_analysis.py --config experiments/lesion_detection/configs/40_brats_side.yaml
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from common import REPO, load_maps_manifest, load_yaml, read_ids, repo_path, true_spacing
from scipy import ndimage

from jannus.evaluation.detection.matching import (
    extract_components,
    match_by_overlap,
    overlap_counts,
)
from jannus.evaluation.detection.provenance import build_provenance, split_record, write_json
from jannus.evaluation.detection.scoring import ScoringInputs, build_scorers, dequantize_uint8


def brats_c3_on_grid(case_id: str, raw_root: str, shape: tuple[int, int, int]) -> np.ndarray | None:
    sid = case_id.removeprefix("UCSF_")
    path = Path(raw_root) / sid / f"{sid}_BraTS-seg.nii.gz"
    if not path.exists():
        return None
    c = np.asarray(nib.load(path).dataobj) == 3
    return ndimage.zoom(c.astype(np.uint8), [t / s for t, s in zip(shape, c.shape, strict=True)],
                        order=0, mode="nearest") > 0


def case_analysis(case_id: str, map_file: str, cfg: dict, specs: dict) -> dict | None:
    maps = np.load(repo_path(map_file))
    stacker = maps["stacker"].astype(np.float32)
    gt_a = np.asarray(nib.load(Path(cfg["gt_root"]) / case_id / "seg.nii.gz").dataobj) > 0.5
    gt_c = brats_c3_on_grid(case_id, cfg["native_roots"]["ucsf"], gt_a.shape)
    if gt_c is None:
        return None
    spacing = true_spacing(case_id, cfg["native_roots"], gt_a.shape)
    inputs = ScoringInputs(stacker=stacker, base=dequantize_uint8(maps["base"]))
    scorers = build_scorers(cfg["scorers"])
    a = extract_components(gt_a, spacing, connectivity=26)
    c = extract_components(gt_c, spacing, connectivity=26)
    pairs = overlap_counts(a, c)  # (A id, C3 id) -> voxels
    a_paired = {p for p, _ in pairs}
    c_paired = {q for _, q in pairs}

    row = {"case": case_id, "n_lesions_A": a.n, "n_lesions_C3": c.n, "n_pairs": len(pairs),
           "dice_A_vs_C3": float(2 * (gt_a & gt_c).sum() / max(gt_a.sum() + gt_c.sum(), 1))}
    for name, s in specs.items():
        p = extract_components(stacker >= s["voxel_threshold"], spacing, connectivity=26)
        score = scorers[s["scorer"]](inputs, p) if p.n else np.zeros(0)
        keep = (score >= s["tau"]) & (p.esd_mm >= s["min_esd_mm"])
        # Detection status uses only the components that survive tau and the ESD cut.
        surv = extract_components(np.isin(p.labels, np.flatnonzero(keep) + 1), spacing, connectivity=26)
        det_a = match_by_overlap(surv, a).detected()
        det_c = match_by_overlap(surv, c).detected()
        changed = sum(1 for (i, j) in pairs if det_a[i - 1] != det_c[j - 1])
        row.update({
            f"{name}:detected_A": int(det_a.sum()), f"{name}:detected_C3": int(det_c.sum()),
            f"{name}:changed_pairs": changed,
            f"{name}:only_A_detected": int(sum(det_a[i - 1] for i in range(1, a.n + 1) if i not in a_paired)),
            f"{name}:only_C3_detected": int(sum(det_c[j - 1] for j in range(1, c.n + 1) if j not in c_paired)),
        })
    row["only_in_A"] = a.n - len(a_paired)
    row["only_in_C3"] = c.n - len(c_paired)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    comp = load_yaml(repo_path(cfg["components_config"]))
    frozen = load_yaml(repo_path(cfg["frozen"]))
    cases = [c for c in read_ids(cfg["cases"]) if c.startswith("UCSF_")]
    manifest = load_maps_manifest(comp["maps_manifest"])
    p = frozen["primary"]["config"]
    specs = {"baseline": {"voxel_threshold": cfg["baseline_voxel_threshold"], "scorer": "max_prob",
                          "tau": -np.inf, "min_esd_mm": 0.0}}
    for t, v in frozen["primary"]["frozen_tau_by_fp_target"].items():
        specs[f"frozen_{t}fp"] = {"voxel_threshold": p["voxel_threshold"], "scorer": p["scorer"],
                                  "tau": float(v["tau"]), "min_esd_mm": p["min_esd_mm"]}
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        rows = [r for r in pool.map(case_analysis, cases, [manifest[c]["file"] for c in cases],
                                    [comp] * len(cases), [specs] * len(cases)) if r is not None]
    clean = set(read_ids(cfg["clean_split"]))
    for r in rows:
        r["stratum"] = "clean" if r["case"] in clean else "contaminated"

    out = repo_path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_scan.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {"n_scans": len(rows), "n_clean": sum(r["stratum"] == "clean" for r in rows)}
    for name in specs:
        ch = np.array([r[f"{name}:changed_pairs"] for r in rows])
        summary[name] = {
            "total_changed_pairs": int(ch.sum()), "scans_with_any_change": int((ch > 0).sum()),
            "changed_per_scan_mean": float(ch.mean()), "changed_per_scan_max": int(ch.max()),
            "detected_A": int(sum(r[f"{name}:detected_A"] for r in rows)),
            "detected_C3": int(sum(r[f"{name}:detected_C3"] for r in rows)),
            "lesions_A": int(sum(r["n_lesions_A"] for r in rows)),
            "lesions_C3": int(sum(r["n_lesions_C3"] for r in rows)),
            "sensitivity_A": sum(r[f"{name}:detected_A"] for r in rows) / max(sum(r["n_lesions_A"] for r in rows), 1),
            "sensitivity_C3": sum(r[f"{name}:detected_C3"] for r in rows) / max(sum(r["n_lesions_C3"] for r in rows), 1),
        }
    summary["only_in_A_total"] = int(sum(r["only_in_A"] for r in rows))
    summary["only_in_C3_total"] = int(sum(r["only_in_C3"] for r in rows))
    write_json(out / "summary.json", summary)
    write_json(out / "provenance.json", build_provenance(
        command="brats_side_analysis", repo_root=REPO, config=cfg, config_path=args.config,
        splits=[split_record("eval84_ucsf", repo_path(cfg["cases"]), cases)],
        inputs={manifest[c]["file"]: manifest[c]["sha256"] for c in cases}, extra={"frozen": frozen}))
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
