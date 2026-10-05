"""Classify the 84-case cohort by training overlap and write the clean/contaminated splits.

Inputs (all read-only):
  configs/splits/eval84_ids.txt               84 eval scan IDs
  configs/splits/nnunet_splits_final_ids.json Dataset001 folds (fold 0 trained
                                               nnU-Net 3D, nnU-Net 2D and SwinUNETR)
  Data/preprocessed_256/train/                the 566-scan development pool; every
                                               non-eval scan was in the stacker's
                                               train/internal-val pool

Outputs (configs/splits/):
  eval84_leakage.csv       one row per eval scan with every overlap flag
  clean_ids.txt            scans whose PATIENT has no scan in fold-0 training
                           and none in the stacker pool (an upper bound, see
                           LEAKAGE_FINDING.md)
  contaminated_ids.txt     every other eval scan

Patient identity: UCSF subject IDs end in a timepoint letter (100238A..G are
one patient; confirmed against TableS1_UCSF_BrainMetastases_SubjectInfo.xlsx).
BMS_Mets_NNN IDs are one scan per patient.

    python experiments/lesion_detection/tools/build_leakage_splits.py --data-root <Data>
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

SPLITS = Path(__file__).resolve().parent.parent / "configs" / "splits"


def patient_id(scan_id: str) -> str:
    if scan_id.startswith("UCSF_"):
        return re.sub(r"[A-Z]$", "", scan_id)
    return scan_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    args = ap.parse_args()

    eval_ids = [ln.strip() for ln in (SPLITS / "eval84_ids.txt").read_text().splitlines() if ln.strip()]
    eval_set = set(eval_ids)
    folds = json.loads((SPLITS / "nnunet_splits_final_ids.json").read_text())
    fold0_train = set(folds[0]["train"])
    fold0_val = set(folds[0]["val"])
    pool = sorted(p.name for p in (args.data_root / "preprocessed_256" / "train").iterdir() if p.is_dir())
    stacker_pool = [s for s in pool if s not in eval_set]

    fold0_train_patients = {patient_id(s) for s in fold0_train}
    stacker_patients = {patient_id(s) for s in stacker_pool}
    eval_scans_per_patient: dict[str, int] = {}
    for s in eval_ids:
        eval_scans_per_patient[patient_id(s)] = eval_scans_per_patient.get(patient_id(s), 0) + 1

    rows = []
    for s in eval_ids:
        pid = patient_id(s)
        row = {
            "scan": s,
            "patient": pid,
            "source": "BMS" if s.startswith("BMS_") else "UCSF",
            "eval_scans_of_patient": eval_scans_per_patient[pid],
            "scan_in_fold0_train": s in fold0_train,
            "scan_in_fold0_val": s in fold0_val,
            "scan_in_dataset001": s in fold0_train or s in fold0_val,
            "patient_in_fold0_train": pid in fold0_train_patients,
            "patient_in_stacker_pool": pid in stacker_patients,
        }
        row["clean"] = not (row["patient_in_fold0_train"] or row["patient_in_stacker_pool"])
        rows.append(row)

    clean = {r["patient"] for r in rows if r["clean"]}
    dirty = {r["patient"] for r in rows if not r["clean"]}
    assert not clean & dirty, "a patient cannot be both clean and contaminated"

    with open(SPLITS / "eval84_leakage.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for name, keep in (("clean_ids.txt", True), ("contaminated_ids.txt", False)):
        ids = sorted(r["scan"] for r in rows if r["clean"] is keep)
        (SPLITS / name).write_text("\n".join(ids) + "\n", newline="\n")

    def count(pred) -> tuple[int, int]:
        sel = [r for r in rows if pred(r)]
        return len(sel), len({r["patient"] for r in sel})

    summary = {
        "eval scans / patients": count(lambda r: True),
        "scan itself in fold-0 train": count(lambda r: r["scan_in_fold0_train"]),
        "patient in fold-0 train (any timepoint)": count(lambda r: r["patient_in_fold0_train"]),
        "patient in stacker pool": count(lambda r: r["patient_in_stacker_pool"]),
        "patient in both": count(lambda r: r["patient_in_fold0_train"] and r["patient_in_stacker_pool"]),
        "clean": count(lambda r: r["clean"]),
        "contaminated": count(lambda r: not r["clean"]),
    }
    for k, (n_scans, n_pat) in summary.items():
        print(f"{k:<42} {n_scans:>3} scans  {n_pat:>3} patients")


if __name__ == "__main__":
    main()
