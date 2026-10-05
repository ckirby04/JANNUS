"""Patient-grouped, stratified, fixed-seed 2-fold split of the 84-case cohort (D5, SECONDARY).

The split is stratified on two patient-level flags taken from **ground truth only**,
never from predictions or metrics:

- does the patient have any GT lesion with ESD < 5 mm (true spacing)?
- does the patient have two or more GT lesions?

Within each of the four strata, patients are shuffled with the fixed seed and
dealt alternately to A and B. All scans of a patient land in the same fold.

The GT lesion fields come from the 26-connected lesion rows of the component
tables, which do not depend on the voxel threshold.

    python experiments/lesion_detection/tools/make_2fold.py --seed 20261004
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EXP, patient_id, read_ids
from select_operating_point import components_dir

COMPONENTS = "experiments/lesion_detection/configs/components.yaml"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--small-mm", type=float, default=5.0)
    args = ap.parse_args()

    cases = read_ids("experiments/lesion_detection/configs/splits/eval84_ids.txt")
    comp_dir = components_dir(COMPONENTS)
    lesions: dict[str, list[float]] = {}
    for c in cases:
        with gzip.open(comp_dir / f"{c}.json.gz", "rt") as fh:
            t = next(t for t in json.load(fh)["tables"] if t["connectivity"] == 26)
        lesions.setdefault(patient_id(c), []).extend(les["esd_mm"] for les in t["lesions"])

    strata: dict[tuple[bool, bool], list[str]] = {}
    for pid in sorted(lesions):
        esd = lesions[pid]
        strata.setdefault((any(e < args.small_mm for e in esd), len(esd) >= 2), []).append(pid)
    rng = np.random.default_rng(args.seed)
    fold = {}
    for key in sorted(strata):
        pids = list(strata[key])
        rng.shuffle(pids)
        for i, pid in enumerate(pids):
            fold[pid] = "A" if i % 2 == 0 else "B"

    out = EXP / "configs" / "splits"
    summary = {"seed": args.seed, "small_mm": args.small_mm, "strata": {}}
    for f in ("A", "B"):
        ids = sorted(c for c in cases if fold[patient_id(c)] == f)
        (out / f"fold_{f}_ids.txt").write_text("\n".join(ids) + "\n", newline="\n")
        summary[f] = {"n_scans": len(ids), "n_patients": len({patient_id(c) for c in ids}),
                      "n_clean_scans": len(set(ids) & set(read_ids("experiments/lesion_detection/configs/splits/clean_ids.txt")))}
    for key, pids in sorted(strata.items()):
        summary["strata"][f"small={key[0]},multi={key[1]}"] = {
            "A": sum(fold[p] == "A" for p in pids), "B": sum(fold[p] == "B" for p in pids)}
    (out / "fold_2way_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    assert not ({patient_id(c) for c in read_ids(out / "fold_A_ids.txt")}
                & {patient_id(c) for c in read_ids(out / "fold_B_ids.txt")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
