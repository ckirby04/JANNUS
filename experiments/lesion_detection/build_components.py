"""Build lesion and component tables from the cached probability maps.

For every case, connectivity, voxel threshold and hit rule this writes, to
``<output_dir>/<config_sha12>/<case>.json.gz``:

``lesions``     one row per GT lesion: sizes (true spacing, plus 1 mm-assumption
                bridge), size bins, best overlap, best bbox IoU, the ids of the
                components that hit it, and its detection score under every scorer
                (``null`` = never hit)
``components``  one row per predicted component: size, FP / fragment flags, number
                of lesions hit, distance to the nearest GT voxel (true mm), and
                every score
``counts``      split / merge / redundant counts with no score filter

Selection never happens here. The output is fully determined by the config and
the cached maps, so a re-run reproduces it byte for byte.

    python experiments/lesion_detection/build_components.py --config experiments/lesion_detection/configs/components.yaml
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from common import (
    load_maps_manifest,
    load_yaml,
    patient_id,
    read_ids,
    repo_path,
    true_spacing,
)
from scipy import ndimage

from jannus.core.checksums import sha256_file
from jannus.evaluation.detection.geometry import (
    assign_size_bin,
)
from jannus.evaluation.detection.matching import extract_components, match
from jannus.evaluation.detection.provenance import config_hash
from jannus.evaluation.detection.scoring import (
    ScoringInputs,
    build_scorers,
    dequantize_uint8,
    score_components,
)

ISO = (1.0, 1.0, 1.0)


def _bins(raw: list) -> list[tuple[str, float, float]]:
    return [(str(n), float(lo), float(hi)) for n, lo, hi in raw]


def _f(x: float) -> float | None:
    """JSON-safe float: -inf (never hit) becomes null; values are rounded for stable output."""
    return None if not np.isfinite(x) else round(float(x), 6)


def process_case(case_id: str, cfg: dict, map_file: str) -> dict:
    t0 = time.time()
    maps = np.load(repo_path(map_file))
    stacker = maps["stacker"].astype(np.float32)
    inputs = ScoringInputs(stacker=stacker, base=dequantize_uint8(maps["base"]),
                           base_names=tuple(str(n) for n in maps["base_names"]))
    gt = np.asarray(nib.load(Path(cfg["gt_root"]) / case_id / "seg.nii.gz").dataobj) > 0.5
    spacing = true_spacing(case_id, cfg["native_roots"], gt.shape)
    scorers = build_scorers(cfg["scorers"])
    bins, bridge_bins = _bins(cfg["size_bins"]), _bins(cfg["bridge_size_bins"])
    # Distance from every voxel to the nearest GT voxel, in true mm. It is inf if the GT is empty.
    dist_to_gt = (ndimage.distance_transform_edt(~gt, sampling=spacing) if gt.any()
                  else np.full(gt.shape, np.inf, dtype=np.float64))

    out: dict = {"case": case_id, "patient": patient_id(case_id), "spacing_mm": list(spacing),
                 "grid": list(gt.shape), "tables": []}
    for conn in cfg["connectivity"]:
        g = extract_components(gt, spacing, connectivity=conn)
        g_iso = extract_components(gt, ISO, connectivity=conn)  # same labels; 1 mm-assumption sizes
        lesion_base = [
            {"lesion": i + 1, "n_vox": int(g.n_vox[i]),
             "esd_mm": _f(g.esd_mm[i]), "longest_axis_mm": _f(g.longest_axis_mm[i]),
             "esd_mm_1mm": _f(g_iso.esd_mm[i]), "longest_axis_mm_1mm": _f(g_iso.longest_axis_mm[i]),
             "bin": assign_size_bin(float(g.esd_mm[i]), bins),
             "bridge_bin": assign_size_bin(float(g_iso.longest_axis_mm[i]), bridge_bins)}
            for i in range(g.n)
        ]
        for thr in cfg["voxel_thresholds"]:
            mask = stacker >= thr if cfg["binarize"] == ">=" else stacker > thr
            p = extract_components(mask, spacing, connectivity=conn)
            scores = score_components(scorers, inputs, p)
            dist = (np.asarray(ndimage.minimum(dist_to_gt, p.labels, np.arange(1, p.n + 1)))
                    if p.n else np.zeros(0))
            for rule_name, rule_cfg in cfg["rules"].items():
                params = {k: v for k, v in rule_cfg.items() if k != "rule"}
                m = match(p, g, rule=rule_cfg["rule"], **params)
                gt_scores = {}
                for sname, svals in scores.items():
                    # Lesion score = max over its hitting components, per scorer (D2).
                    gt_scores[sname] = [
                        _f(max(svals[q - 1] for q in hits)) if hits else None for hits in m.gt_hit_by
                    ]
                lesions = [
                    {**lb, "overlap_vox": int(m.gt_overlap_vox[i]),
                     "overlap_frac": _f(m.gt_overlap_frac[i]),
                     "bbox_iou": _f(m.gt_best_bbox_iou[i]),
                     "hits": list(m.gt_hit_by[i]),
                     **{f"score_{s}": gt_scores[s][i] for s in scores}}
                    for i, lb in enumerate(lesion_base)
                ]
                fp, frag = m.pred_is_fp, m.pred_is_fragment
                components = [
                    {"comp": j + 1, "n_vox": int(p.n_vox[j]), "esd_mm": _f(p.esd_mm[j]),
                     "is_fp": bool(fp[j]), "is_fragment": bool(frag[j]),
                     "n_lesions_hit": len(m.pred_hits[j]), "dist_to_gt_mm": _f(dist[j]),
                     **{f"score_{s}": _f(v[j]) for s, v in scores.items()}}
                    for j in range(p.n)
                ]
                out["tables"].append({
                    "connectivity": conn, "voxel_threshold": thr, "rule": rule_name,
                    "lesions": lesions, "components": components,
                    "counts": {"split_lesions": m.n_split_lesions(), "merge_components": m.n_merge_predictions(),
                               "redundant_components": m.n_redundant(), "fragments": int(frag.sum())},
                })
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    digest = config_hash(cfg)
    out_dir = repo_path(cfg["output_dir"]) / digest[:12]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    manifest = load_maps_manifest(cfg["maps_manifest"])
    cases = []
    for f in cfg["cases"]:
        cases += [c for c in read_ids(f) if c not in cases]
    if args.only:
        cases = [c for c in cases if c in set(args.only)]
    missing = [c for c in cases if c not in manifest]
    if missing:
        raise SystemExit(f"{len(missing)} case(s) not yet in the maps manifest: {missing[:5]}")
    for c in cases:  # never build from a map that differs from the one inference recorded
        if sha256_file(repo_path(manifest[c]["file"])) != manifest[c]["sha256"]:
            raise SystemExit(f"{c}: cached map hash differs from manifest")
    todo = [c for c in cases if not (out_dir / f"{c}.json.gz").exists()]
    print(f"config {digest[:12]}: {len(cases)} cases, {len(todo)} to build, {cfg['workers']} workers",
          flush=True)

    with ProcessPoolExecutor(max_workers=int(cfg["workers"])) as pool:
        futures = {pool.submit(process_case, c, cfg, manifest[c]["file"]): c for c in todo}
        for i, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            path = out_dir / f"{res['case']}.json.gz"
            tmp = path.with_suffix(".partial")
            # mtime=0 keeps the gzip header deterministic, so re-runs are byte-identical.
            with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
                gz.write(json.dumps(res, sort_keys=True).encode("utf-8"))
            tmp.replace(path)
            print(f"[{i}/{len(todo)}] {res['case']} {res['seconds']}s", flush=True)
    print(f"done -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
