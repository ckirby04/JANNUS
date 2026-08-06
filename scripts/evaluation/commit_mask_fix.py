"""
Commit a radiologist-edited mask back into the training pipeline.

Updates the mask in both stacking caches (stacking_cache_v5/ and
stacking_cache_v5_256/) for the given case, so the next training run picks
up the cleaned label. Preserves all other keys (predictions) untouched.

Also logs the change to model/review_gallery/edits_log.jsonl so we have an
audit trail.

Usage:
    python scripts/evaluation/commit_mask_fix.py <case_id> <path/to/edited.nii.gz>
    python scripts/evaluation/commit_mask_fix.py CASE_ID ~/edits/CASE_ID.nii.gz
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent.parent
OLD_CACHE = PROJECT / "model" / "stacking_cache_v5"
NEW_CACHE = PROJECT / "model" / "stacking_cache_v5_256"
LOG_PATH = PROJECT / "model" / "review_gallery" / "edits_log.jsonl"
ORIG_MASKS_DIR = PROJECT / "model" / "review_gallery" / "original_masks"


def main():
    if len(sys.argv) != 3:
        print("Usage: commit_mask_fix.py <case_id> <path/to/edited-mask.nii.gz>")
        sys.exit(1)
    case_id = sys.argv[1]
    new_mask_path = Path(sys.argv[2])
    if not new_mask_path.exists():
        print(f"ERROR: {new_mask_path} not found")
        sys.exit(1)

    old_npz = OLD_CACHE / f"{case_id}.npz"
    new_npz = NEW_CACHE / f"{case_id}.npz"
    if not old_npz.exists():
        print(f"ERROR: {old_npz} not found")
        sys.exit(1)
    if not new_npz.exists():
        print(f"ERROR: {new_npz} not found")
        sys.exit(1)

    # Load new mask
    new_mask = nib.load(str(new_mask_path)).get_fdata()
    new_mask = (new_mask > 0.5).astype(np.uint8)
    if new_mask.shape != (256, 256, 256):
        print(f"ERROR: new mask shape {new_mask.shape} != (256, 256, 256)")
        sys.exit(1)
    print(f"Loaded new mask: {new_mask.shape}, {int(new_mask.sum())} fg voxels")

    # Backup original mask
    ORIG_MASKS_DIR.mkdir(parents=True, exist_ok=True)
    orig_backup = ORIG_MASKS_DIR / f"{case_id}_original.npz"
    if not orig_backup.exists():
        orig = np.load(old_npz)
        np.savez_compressed(orig_backup, mask=orig["mask"])
        print(f"Backed up original mask -> {orig_backup}")
    else:
        print(f"Original already backed up at {orig_backup} (keeping)")

    # Update OLD_CACHE npz
    old_data = np.load(old_npz)
    out = {k: old_data[k] for k in old_data.keys()}
    out["mask"] = new_mask
    np.savez_compressed(old_npz, **out)
    print(f"Updated {old_npz}")

    # Update NEW_CACHE npz (same case, different base preds)
    new_data = np.load(new_npz)
    out = {k: new_data[k] for k in new_data.keys()}
    out["mask"] = new_mask.astype(np.float32)
    np.savez_compressed(new_npz, **out)
    print(f"Updated {new_npz}")

    # Append to audit log
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "case_id": case_id,
            "source_file": str(new_mask_path),
            "fg_voxels_new": int(new_mask.sum()),
            "fg_voxels_orig": int(np.load(orig_backup)["mask"].sum()),
        }) + "\n")
    print(f"Logged edit to {LOG_PATH}")

    print(f"\nDone. Next training run will use the edited mask for {case_id}.")


if __name__ == "__main__":
    main()
