"""Compare the candidate UCSF ground-truth files (Q5 in PLAN.md).

Candidates per UCSF subject:
  A  Data/preprocessed_256/train/UCSF_<id>/seg.nii.gz      (what the stacker cache / 84-case eval used)
  B  Raw Data/UCSF_BrainMetastases/train/<id>/<id>_seg.nii.gz        (original release, native space)
  C  Raw Data/UCSF_BrainMetastases/train/<id>/<id>_BraTS-seg.nii.gz  (BraTS-METS 2023 revision, 324 cases)

For each subject: label values, which native file A was derived from (Dice of A
against each native mask zoomed to 256^3 with nearest neighbour, the same
operation as scripts/preprocessing/preprocess_256.py), and lesion counts
(26-connectivity, native grid). Read-only on all inputs; writes one CSV.

    python experiments/lesion_detection/tools/compare_ucsf_gt.py --out <csv>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

PRE = Path("G:/BrainMetShare/Data/preprocessed_256/train")
RAW = Path("G:/BrainMetShare/Raw Data/UCSF_BrainMetastases/train")
CONN26 = ndimage.generate_binary_structure(3, 3)


def _zoom_to_256(mask: np.ndarray) -> np.ndarray:
    factors = [256 / s for s in mask.shape]
    return ndimage.zoom(mask.astype(np.uint8), factors, order=0, mode="nearest") > 0


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = int(a.sum() + b.sum())
    return 1.0 if denom == 0 else 2.0 * int((a & b).sum()) / denom


def _n_lesions(mask: np.ndarray) -> int:
    return int(ndimage.label(mask, structure=CONN26)[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for case_dir in sorted(PRE.glob("UCSF_*")):
        sid = case_dir.name.removeprefix("UCSF_")
        a = nib.load(case_dir / "seg.nii.gz").get_fdata() > 0.5
        b_img = nib.load(RAW / sid / f"{sid}_seg.nii.gz")
        b_raw = np.asarray(b_img.dataobj)
        b = b_raw > 0
        row = {
            "case": case_dir.name,
            "B_labels": " ".join(str(int(v)) for v in np.unique(b_raw)),
            "dice_A_vs_zoomB": round(_dice(a, _zoom_to_256(b)), 4),
            "lesions_B": _n_lesions(b),
            "vox_B": int(b.sum()),
        }
        c_path = RAW / sid / f"{sid}_BraTS-seg.nii.gz"
        if c_path.exists():
            c_raw = np.asarray(nib.load(c_path).dataobj)
            c_enh = c_raw == 3
            c_any = c_raw > 0
            row.update({
                "C_labels": " ".join(str(int(v)) for v in np.unique(c_raw)),
                "C_same_shape": c_raw.shape == b_raw.shape,
                "dice_A_vs_zoomCenh": round(_dice(a, _zoom_to_256(c_enh)), 4),
                "dice_B_vs_Cenh": round(_dice(b, c_enh), 4) if c_raw.shape == b_raw.shape else "",
                "dice_B_vs_Cany": round(_dice(b, c_any), 4) if c_raw.shape == b_raw.shape else "",
                "lesions_Cenh": _n_lesions(c_enh),
                "vox_Cenh": int(c_enh.sum()),
            })
        rows.append(row)
        print(row, flush=True)

    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "case", k))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
