"""Compare today's nnU-Net 3D/2D outputs with BOTH cached nnU-Net variants (diagnosis only).

- ``stacking_cache_v5`` (``nnunet``, ``nnunet_2d``): described as "native-preprocessed".
  This is what the production stacker was trained and evaluated on.
- ``stacking_cache_v5_256`` (``nnunet_3d``, ``nnunet_2d``): nnU-Net run on the resampled
  256^3 grid. stacking_v4_vs_v5_writeup.md:55 describes its calibration as broken.

The comparison uses the same 5 tuning cases, reads model keys only (never ``mask``)
and uses no labels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO
from diagnose_reproduction import NEW_CACHE, OLD_CACHE, compare

from jannus.core.config import load_config
from jannus.data.layout import scan_dataset
from jannus.data.loading import load_case
from jannus.evaluation.detection.provenance import write_json
from jannus.segmentation.model_adapters import build_adapter

CASES = ["BMS_Mets_039", "BMS_Mets_051", "UCSF_100102A", "UCSF_100110A", "UCSF_100319A"]

cfg = load_config(REPO / "configs/models.yaml")
entries = {m["name"]: m for m in cfg.models}
adapters = {n: build_adapter(entries[n], device=torch.device("cuda")) for n in ("nnunet_3d", "nnunet_2d")}
by_id = {c.case_id: c for c in scan_dataset("G:/BrainMetShare/Data/preprocessed_256/train", cfg.data).cases}
out = {}
for case in CASES:
    vol = torch.from_numpy(load_case(by_id[case], cfg.data.sequences).array)
    fresh = {n: a.predict_crop(vol).numpy() for n, a in adapters.items()}
    with np.load(OLD_CACHE / f"{case}.npz") as o, np.load(NEW_CACHE / f"{case}.npz") as w:
        cached = {"v5:nnunet_3d": o["nnunet"], "v5:nnunet_2d": o["nnunet_2d"],
                  "v5_256:nnunet_3d": w["nnunet_3d"], "v5_256:nnunet_2d": w["nnunet_2d"]}
    out[case] = {
        "fresh_3d_vs_v5": compare(fresh["nnunet_3d"], cached["v5:nnunet_3d"]),
        "fresh_3d_vs_v5_256": compare(fresh["nnunet_3d"], cached["v5_256:nnunet_3d"]),
        "fresh_2d_vs_v5": compare(fresh["nnunet_2d"], cached["v5:nnunet_2d"]),
        "fresh_2d_vs_v5_256": compare(fresh["nnunet_2d"], cached["v5_256:nnunet_2d"]),
        "v5_vs_v5_256_3d": compare(cached["v5:nnunet_3d"], cached["v5_256:nnunet_3d"]),
        "v5_vs_v5_256_2d": compare(cached["v5:nnunet_2d"], cached["v5_256:nnunet_2d"]),
    }
    print(case, {k: (round(v["pearson_r"] or 0, 3), round(v["dice_ge05"], 3), round(v["max_abs_diff"], 3))
                 for k, v in out[case].items()}, flush=True)
write_json(REPO / "experiments/lesion_detection/results/reproduction_diagnosis/nnunet_cache_variants.json", out)
