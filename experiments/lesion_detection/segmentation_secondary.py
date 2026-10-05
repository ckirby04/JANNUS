"""Secondary segmentation metrics for a detection operating point (D10), and the HD95 test (D8).

A detection operating point defines a mask:

- the stacker map is binarised at the voxel threshold;
- components are labelled;
- components scoring below τ, or with ESD below the minimum, are dropped.

For that mask this module computes voxel Dice and HD95 with the existing,
unmodified ``jannus.evaluation.metrics`` functions, using true spacing. It
also computes HD95 at the old 1 mm assumption as a bridge.

The HD95 test recomputes HD95 after deleting FP components farther than X mm
from every GT voxel, and after deleting every FP. If HD95 drops most of the
way to the all-FPs-removed value once only the *far* FPs are gone, then
"isolated far FPs drive HD95" is confirmed.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from common import repo_path, true_spacing
from scipy import ndimage

from jannus.evaluation.detection.matching import extract_components
from jannus.evaluation.detection.scoring import ScoringInputs, build_scorers, dequantize_uint8
from jannus.evaluation.harness import voxel_dice
from jannus.evaluation.metrics import hd95


def case_segmentation_metrics(case_id: str, map_file: str, gt_root: str, native_roots: dict,
                              specs: dict[str, dict], scorer_cfg: dict, far_cuts_mm: list[float]) -> dict:
    """``specs``: name -> {voxel_threshold, connectivity, scorer, tau, min_esd_mm, hd95_test}."""
    maps = np.load(repo_path(map_file))
    stacker = maps["stacker"].astype(np.float32)
    inputs = ScoringInputs(stacker=stacker, base=dequantize_uint8(maps["base"]))
    gt = np.asarray(nib.load(Path(gt_root) / case_id / "seg.nii.gz").dataobj) > 0.5
    spacing = true_spacing(case_id, native_roots, gt.shape)
    scorers = build_scorers(scorer_cfg)
    dist_to_gt = (ndimage.distance_transform_edt(~gt, sampling=spacing) if gt.any()
                  else np.full(gt.shape, np.inf))

    out: dict = {"case": case_id}
    for name, s in specs.items():
        comps = extract_components(stacker >= s["voxel_threshold"], spacing, connectivity=s["connectivity"])
        ids = np.arange(1, comps.n + 1)
        score = scorers[s["scorer"]](inputs, comps) if comps.n else np.zeros(0)
        keep = (score >= s["tau"]) & (comps.esd_mm >= s["min_esd_mm"])
        mask = np.isin(comps.labels, ids[keep])
        row = {"dice": voxel_dice(mask, gt), "hd95_mm": hd95(mask, gt, spacing=spacing),
               "hd95_mm_1mm_assumption": hd95(mask, gt, spacing=1.0)}
        if s.get("hd95_test") and comps.n:
            touches = np.asarray(ndimage.maximum(gt, comps.labels, ids)) > 0 if gt.any() else np.zeros(comps.n, bool)
            mind = np.asarray(ndimage.minimum(dist_to_gt, comps.labels, ids))
            is_fp = keep & ~touches
            for cut in far_cuts_mm:
                drop = is_fp & (mind > cut)
                row[f"hd95_mm_without_fp_beyond_{cut:g}mm"] = hd95(np.isin(comps.labels, ids[keep & ~drop]), gt,
                                                                   spacing=spacing)
            row["hd95_mm_without_any_fp"] = hd95(np.isin(comps.labels, ids[keep & ~is_fp]), gt, spacing=spacing)
            row["n_fp"] = int(is_fp.sum())
        out[name] = row
    return out
