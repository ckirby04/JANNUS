"""Diagnose the reproduction-gate failure (REPRODUCTION_FAILURE.md). Diagnosis only.

Scope, set by the owner's one-time exception:

- 3-5 TUNING (contaminated) cases;
- read-only access to ``1.40/model/stacking_cache_v5/`` and ``stacking_cache_v5_256/``;
- NO labels: the cache files' ``mask`` key is never read;
- no change to any pipeline code, config or weight.

Instrumentation uses forward pre-hooks attached at runtime only.

Per case and per base model, it reports:

- the input tensor each network actually receives (shape, dtype, mean/std/min/max);
- the output space (probability or logit);
- fresh-vs-cached voxelwise agreement: Pearson r, max/mean abs diff, Dice of >=0.5 masks, voxel counts;
- the same label-free cross-model statistics computed on the CACHED maps.

It also runs a stacker swap test: the production stacker is run on (a) all cached
maps, (b) all fresh maps, and (c) fresh maps with one model at a time replaced by
its cached map. Agreement with (a) shows which model's change moves the fused output.

    python experiments/lesion_detection/tools/diagnose_reproduction.py --cases BMS_Mets_039 ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import REPO, read_ids, repo_path

from jannus.core.checksums import sha256_file
from jannus.core.config import load_config
from jannus.data.layout import scan_dataset
from jannus.data.loading import load_case
from jannus.evaluation.detection.provenance import build_provenance, write_json
from jannus.segmentation.pipeline import BrainMetPipeline
from jannus.segmentation.stacking import (
    STACKING_MODEL_NAMES,
    build_stacking_features_from_preds,
    sliding_window_inference,
)

OLD_CACHE = Path("G:/BrainMetShare/1.40/model/stacking_cache_v5")
NEW_CACHE = Path("G:/BrainMetShare/1.40/model/stacking_cache_v5_256")
# Cache key per production model name (from scripts/evaluation/seven_model_stacker.py:load_seven_preds).
CACHE_KEY = {"nnunet_3d": (OLD_CACHE, "nnunet"), "nnunet_2d": (OLD_CACHE, "nnunet_2d"),
             "patch_8": (OLD_CACHE, "patch_8"), "patch_12": (OLD_CACHE, "patch_12"),
             "patch_24": (OLD_CACHE, "patch_24"), "patch_36": (OLD_CACHE, "patch_36"),
             "swin_unetr": (NEW_CACHE, "swin_unetr")}
FORBIDDEN_KEYS = {"mask"}


def tstats(x) -> dict:
    a = x.detach().float().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, dtype=np.float32)
    out = {"shape": list(a.shape), "dtype": str(x.dtype), "mean": float(a.mean()), "std": float(a.std()),
           "min": float(a.min()), "max": float(a.max())}
    if a.ndim >= 4:  # per-channel stats on the channel axis just before the spatial dims
        ch_axis = a.ndim - 4
        out["per_channel_mean"] = [float(v) for v in np.moveaxis(a, ch_axis, 0).reshape(a.shape[ch_axis], -1).mean(1)]
        out["per_channel_std"] = [float(v) for v in np.moveaxis(a, ch_axis, 0).reshape(a.shape[ch_axis], -1).std(1)]
    return out


def compare(fresh: np.ndarray, cached: np.ndarray) -> dict:
    f, c = fresh.astype(np.float64).ravel(), cached.astype(np.float64).ravel()
    fm, cm = f >= 0.5, c >= 0.5
    denom = fm.sum() + cm.sum()
    return {"pearson_r": float(np.corrcoef(f, c)[0, 1]) if f.std() > 0 and c.std() > 0 else None,
            "max_abs_diff": float(np.abs(f - c).max()), "mean_abs_diff": float(np.abs(f - c).mean()),
            "dice_ge05": float(2 * (fm & cm).sum() / denom) if denom else 1.0,
            "fresh_vox_ge05": int(fm.sum()), "cached_vox_ge05": int(cm.sum()),
            "frac_voxels_abs_diff_gt_0.1": float((np.abs(f - c) > 0.1).mean())}


def cross_model(maps: dict[str, np.ndarray]) -> dict:
    names = list(maps)
    fg = np.stack([maps[n] >= 0.5 for n in names])
    out = {}
    for i, n in enumerate(names):
        others = np.delete(fg, i, 0).sum(0) >= 4
        denom = fg[i].sum() + others.sum()
        out[n] = {"vox_ge05": int(fg[i].sum()), "max": float(maps[n].max()),
                  "dice_vs_majority_of_others": float(2 * (fg[i] & others).sum() / denom) if denom else 1.0}
    return out


def run_stacker(pipe: BrainMetPipeline, preds: list[np.ndarray]) -> np.ndarray:
    feats = build_stacking_features_from_preds(np.stack(preds).astype(np.float32))
    return sliding_window_inference(pipe.stacking_model, feats, pipe.stacker_patch, pipe.device,
                                    overlap=pipe.stacker_overlap).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--out", default="experiments/lesion_detection/results/reproduction_diagnosis")
    args = ap.parse_args()

    tuning = set(read_ids("experiments/lesion_detection/configs/splits/contaminated_ids.txt"))
    clean = set(read_ids("experiments/lesion_detection/configs/splits/clean_ids.txt"))
    if not 3 <= len(args.cases) <= 5 or any(c not in tuning or c in clean for c in args.cases):
        raise SystemExit("scope: 3-5 tuning (contaminated) cases only")

    cfg = load_config(REPO / "configs/models.yaml")
    pipe = BrainMetPipeline.from_config(cfg.source_path, device=torch.device("cuda"), verify=True)
    index = scan_dataset("G:/BrainMetShare/Data/preprocessed_256/train", cfg.data)
    by_id = {c.case_id: c for c in index.cases}

    # Runtime-only hooks: record the first input each network sees, per adapter call.
    captured: dict[str, dict] = {}

    def hook_for(name):
        def _hook(module, inputs):
            if name not in captured:
                captured[name] = tstats(inputs[0])
        return _hook

    handles = []
    for a in pipe.model_adapters:
        net = getattr(a, "model", None)
        if net is None and getattr(a, "_predictor", None) is not None:
            net = a._predictor.network
        handles.append(net.register_forward_pre_hook(hook_for(a.name)))

    out_dir = repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"cases": {}, "cache_files": {}}
    for case_id in args.cases:
        vol = load_case(by_id[case_id], cfg.data.sequences)
        rec: dict = {"pipeline_input": tstats(vol.array), "loader_spacing": list(vol.voxel_spacing),
                     "network_input": {}, "adapter_input": {}, "output_range": {}, "fresh_vs_cached": {}}
        fresh: dict[str, np.ndarray] = {}
        vol_t = torch.from_numpy(np.ascontiguousarray(vol.array))
        for a in pipe.model_adapters:
            captured.pop(a.name, None)
            rec["adapter_input"][a.name] = tstats(vol_t)
            fresh[a.name] = a.predict_crop(vol_t).numpy().astype(np.float32)
            rec["network_input"][a.name] = captured.get(a.name)
        cached: dict[str, np.ndarray] = {}
        for n in STACKING_MODEL_NAMES:
            root, key = CACHE_KEY[n]
            path = root / f"{case_id}.npz"
            with np.load(path) as z:
                rec.setdefault("cache_keys", {})[root.name] = sorted(z.files)
                assert key not in FORBIDDEN_KEYS
                cached[n] = z[key].astype(np.float32)  # only the model key is read; 'mask' is never accessed
            report["cache_files"].setdefault(str(path), sha256_file(path))
        for n in STACKING_MODEL_NAMES:
            rec["output_range"][n] = {"fresh": [float(fresh[n].min()), float(fresh[n].max())],
                                      "cached": [float(cached[n].min()), float(cached[n].max())],
                                      "cached_dtype": str(cached[n].dtype), "cached_shape": list(cached[n].shape)}
            rec["fresh_vs_cached"][n] = compare(fresh[n], cached[n])
        rec["cross_model_fresh"] = cross_model(fresh)
        rec["cross_model_cached"] = cross_model(cached)

        # Stacker swap test (labels never used; the reference is the all-cached stacker output).
        names = list(STACKING_MODEL_NAMES)
        ref = run_stacker(pipe, [cached[n] for n in names])
        swaps = {"all_fresh": [fresh[n] for n in names]}
        for n in names:
            swaps[f"fresh_but_cached_{n}"] = [cached[m] if m == n else fresh[m] for m in names]
        swaps["fresh_but_cached_all_patch"] = [cached[m] if m.startswith("patch_") else fresh[m] for m in names]
        swaps["fresh_but_cached_patch_8_12"] = [cached[m] if m in ("patch_8", "patch_12") else fresh[m] for m in names]
        swaps["cached_but_fresh_all_patch"] = [fresh[m] if m.startswith("patch_") else cached[m] for m in names]
        rec["stacker_ref_vox_ge055"] = int((ref >= 0.55).sum())
        rec["stacker_swap_vs_all_cached"] = {}
        for k, preds in swaps.items():
            s = run_stacker(pipe, preds)
            sm, rm = s >= 0.55, ref >= 0.55
            denom = sm.sum() + rm.sum()
            rec["stacker_swap_vs_all_cached"][k] = {
                "dice_ge055_vs_all_cached": float(2 * (sm & rm).sum() / denom) if denom else 1.0,
                "vox_ge055": int(sm.sum()), "max_abs_diff": float(np.abs(s - ref).max())}
        report["cases"][case_id] = rec
        print(case_id, {n: round(v["dice_ge05"], 3) for n, v in rec["fresh_vs_cached"].items()}, flush=True)
        print("   swap:", {k: round(v["dice_ge055_vs_all_cached"], 3)
                           for k, v in rec["stacker_swap_vs_all_cached"].items()}, flush=True)
        del fresh, cached

    for h in handles:
        h.remove()
    write_json(out_dir / "diagnosis.json", report)
    write_json(out_dir / "provenance.json", build_provenance(
        command="diagnose_reproduction", repo_root=REPO, config={"cases": args.cases},
        config_path="(command line)", splits=[{"name": "tuning_subset", "case_ids": sorted(args.cases)}],
        inputs=report["cache_files"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
