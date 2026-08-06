"""
Refresh ONLY the SwinUNETR predictions in model/stacking_cache_v5_256/.

After extending SwinUNETR training, the old swin_unetr predictions in the
256^3 cache are stale. This script re-runs SwinUNETR on every case and
overwrites only the `swin_unetr` key in each npz, leaving nnunet_3d,
nnunet_2d, and mask untouched. nnU-Net predictions in the v2 hybrid
pipeline come from model/stacking_cache_v5/ and don't need touching.

Usage:
    python scripts/training/refresh_swin_unetr_cache.py
    python scripts/training/refresh_swin_unetr_cache.py --force   # overwrite even if recent
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

os.environ.setdefault("nnUNet_raw", str(PROJECT / "nnUNet" / "nnUNet_raw"))
os.environ.setdefault("nnUNet_preprocessed", str(PROJECT / "nnUNet" / "nnUNet_preprocessed"))
os.environ.setdefault("nnUNet_results", str(PROJECT / "nnUNet" / "nnUNet_results"))

from jannus.segmentation.dataset import BrainMetDataset
from jannus.segmentation.model_adapters import build_adapter

CONFIG_PATH = PROJECT / "configs" / "models.yaml"
CACHE_DIR = PROJECT / "model" / "stacking_cache_v5_256"
TARGET_SIZE = (256, 256, 256)

DATA_SOURCES = [
    (PROJECT / "data" / "train", ['t1_pre', 't1_gd', 'flair', 'bravo']),
    (PROJECT / "data" / "preprocessed_256" / "train", ['t1_pre', 't1_gd', 'flair', 't2']),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Refresh every case, even if cache file exists")
    args = ap.parse_args()

    print("=" * 66)
    print("  Refresh SwinUNETR predictions in stacking_cache_v5_256")
    print("=" * 66)

    if not CACHE_DIR.exists():
        raise SystemExit(f"Cache dir not found: {CACHE_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    entries = {m["name"]: m for m in cfg.get("models", [])}
    entry = entries.get("swin_unetr")
    if entry is None:
        raise SystemExit("configs/models.yaml missing 'swin_unetr'")

    adapter = build_adapter(entry, device=device, stub=False)
    if getattr(adapter, "_stub", False):
        raise SystemExit("SwinUNETR adapter fell back to stub -- need real weights")
    print(f"  Loaded SwinUNETR adapter from: {entry.get('weights')}")

    # Collect all cases (dedup by case_id) from both data sources
    seen = set()
    all_cases = []
    for data_dir, sequences in DATA_SOURCES:
        if not data_dir.exists():
            print(f"  Skipping {data_dir} (not found)")
            continue
        ds = BrainMetDataset(
            data_dir=str(data_dir),
            sequences=sequences,
            target_size=TARGET_SIZE,
            augment=False,
        )
        for idx in range(len(ds)):
            cid = ds.cases[idx].name
            if cid not in seen:
                seen.add(cid)
                all_cases.append((idx, ds))

    # Filter to cases that have an existing cache file (we only refresh what's there)
    cached = {p.stem for p in CACHE_DIR.glob("*.npz")}
    all_cases = [(idx, ds) for idx, ds in all_cases if ds.cases[idx].name in cached]
    total = len(all_cases)
    print(f"  Cases to refresh: {total}")

    if total == 0:
        print("  Nothing to do.")
        return

    t_start = time.time()
    refreshed = 0

    for i, (idx, ds) in enumerate(all_cases):
        cid = ds.cases[idx].name
        out_path = CACHE_DIR / f"{cid}.npz"

        image, mask, _ = ds[idx]
        if hasattr(image, "numpy"):
            image = image.numpy()
        if hasattr(mask, "numpy"):
            mask = mask.numpy()
        image_t = torch.from_numpy(np.asarray(image).astype(np.float32))

        t0 = time.time()
        new_swin = adapter.predict_crop(image_t)
        if isinstance(new_swin, torch.Tensor):
            new_swin = new_swin.detach().cpu().numpy()
        new_swin = new_swin.astype(np.float32)
        elapsed = time.time() - t0

        # Read existing fields, overwrite only swin_unetr
        existing = np.load(out_path)
        out_data = {k: existing[k] for k in existing.keys()}
        out_data["swin_unetr"] = new_swin
        np.savez_compressed(out_path, **out_data)
        refreshed += 1

        if refreshed % 10 == 0 or refreshed == 1:
            rate = (time.time() - t_start) / refreshed
            remaining = (total - refreshed) * rate
            print(f"    [{refreshed}/{total}] {cid}  "
                  f"({elapsed:.1f}s)  ~{remaining / 60:.0f}min left")

    print(f"\n  Refreshed {refreshed} cases in "
          f"{(time.time() - t_start) / 60:.1f} min")
    print(f"  Cache: {CACHE_DIR}")


if __name__ == "__main__":
    main()
