"""Evaluate frozen operating points ONCE on a held-out split, plus the production baseline.

Writes ``results/<run_id>/``:

- ``metrics.json``: every number, with patient-bootstrap CIs and the split it came from;
- FROC CSVs and figures;
- ``provenance.json``.

REPORT.md is rendered from these files by ``make_report.py``.

Run-once rule: if ``results/<run_id>/metrics.json`` already exists, the script
refuses to run. ``--reproduce`` writes to ``results/<run_id>_reproduce/``
instead and reports whether metrics.json is byte-identical. That check is for
reproducibility only; it selects nothing.

    python experiments/lesion_detection/evaluate.py --config experiments/lesion_detection/configs/20_evaluate_primary.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise, product
from pathlib import Path

import numpy as np
from common import REPO, load_maps_manifest, load_yaml, read_ids, repo_path
from segmentation_secondary import case_segmentation_metrics
from select_operating_point import components_dir, size_bins
from tables import Config, detection_data, edge_counts, fp_components, load_tables

from jannus.core.checksums import sha256_file
from jannus.evaluation.detection.bootstrap import bootstrap
from jannus.evaluation.detection.metrics import FROC, froc, operating_point, summarize_curve
from jannus.evaluation.detection.provenance import (
    build_provenance,
    config_hash,
    split_record,
    write_json,
)
from jannus.evaluation.detection.report import plot_froc_comparison, plot_threshold_grid


def clean_json(obj):
    """NaN -> null, +/-inf -> "inf"/"-inf", so metrics.json is strict JSON."""
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return obj
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [clean_json(v) for v in obj]
    if isinstance(obj, np.floating):
        return clean_json(float(obj))
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def boot(data, stat, b) -> dict:
    r = bootstrap(data, stat, n_resamples=b["n_resamples"], seed=b["seed"], ci=b["ci"])
    return {"point": r.point, "ci_low": r.ci_low, "ci_high": r.ci_high, "n_valid": r.n_valid,
            "n_resamples": r.n_resamples, "seed": r.seed, "n_patients": r.n_patients}


def op_block(data, tau, bins, b) -> dict:
    return boot(data, lambda d, w: operating_point(d, tau, bins=bins, case_weight=w), b)


def curve_block(data, bins, targets, b) -> dict:
    return boot(data, lambda d, w: summarize_curve(d, bins=bins, targets=targets, case_weight=w), b)


def curve_rows(curve: FROC, label: str) -> list[dict]:
    return [{"curve": label, "tau": t, "fp_per_scan": f, "sensitivity": s}
            for t, f, s in zip(curve.thresholds, curve.fp_per_scan, curve.sensitivity, strict=True)]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(clean_json(rows))


def fp_characterization(fps: list[dict], bands: list[float], bins: list[tuple]) -> dict:
    esd = np.array([f["esd_mm"] for f in fps], dtype=float)
    dist = np.array([np.inf if f["dist_to_gt_mm"] is None else f["dist_to_gt_mm"] for f in fps], dtype=float)
    out = {"n_fp": len(fps)}
    if fps:
        out["esd_mm_quantiles"] = {q: float(np.quantile(esd, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
        out["dist_mm_quantiles"] = {q: float(np.quantile(dist, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    out["by_size_bin"] = {name: int(((esd >= lo) & (esd < hi)).sum()) for name, lo, hi in bins}
    out["by_distance_band_mm"] = {f"{lo:g}-{hi:g}": int(((dist >= lo) & (dist < hi)).sum())
                                  for lo, hi in pairwise(bands)}
    out["no_gt_in_scan"] = int(np.isinf(dist).sum())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--reproduce", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    frozen = load_yaml(repo_path(cfg["frozen"]))
    run_id = f"{cfg['name']}_{config_hash({'eval': cfg, 'frozen': frozen})[:8]}"
    out = repo_path(cfg["results_dir"]) / (run_id + ("_reproduce" if args.reproduce else ""))
    if (out / "metrics.json").exists():
        raise SystemExit(f"{out} already evaluated. The test split is evaluated once; "
                         "use --reproduce for a reproducibility check.")
    out.mkdir(parents=True, exist_ok=True)

    test = read_ids(cfg["test_split"])
    tuned_on = read_ids(frozen["selected_on"])
    if set(test) & set(tuned_on):
        raise SystemExit("test split overlaps the split the operating point was selected on")
    comp_cfg = load_yaml(repo_path(cfg["components_config"]))
    comp_dir = components_dir(cfg["components_config"])
    if str(comp_dir.relative_to(REPO)) != frozen["components_dir"].replace("\\", "/") and \
            str(comp_dir.relative_to(REPO)).replace("\\", "/") != frozen["components_dir"].replace("\\", "/"):
        raise SystemExit("frozen choice was made on different component tables")
    bins = size_bins(cfg["components_config"])
    bin_defs = [(n, float(lo), float(hi)) for n, lo, hi in comp_cfg["size_bins"]]
    bridge = [b[0] for b in comp_cfg["bridge_size_bins"]]
    targets, b = cfg["fp_targets"], cfg["bootstrap"]
    tables = load_tables(comp_dir, test)

    p = frozen["primary"]["config"]
    sel = Config(p["connectivity"], p["voxel_threshold"], p["rule"], p["scorer"], p["min_esd_mm"])
    bl = Config(cfg["baseline"]["connectivity"], cfg["baseline"]["voxel_threshold"], cfg["baseline"]["rule"],
                cfg["baseline"]["scorer"], cfg["baseline"]["min_esd_mm"])
    frozen_tau = {t: float(v["tau"]) for t, v in frozen["primary"]["frozen_tau_by_fp_target"].items()}
    d_sel, d_bl = detection_data(tables, test, sel), detection_data(tables, test, bl)

    m: dict = {"run_id": run_id, "split": {"name": "test", "label": cfg["test_label"], "file": cfg["test_split"],
                                           "n_scans": len(test), "n_patients": len(set(d_sel.patient_ids))},
               "selected_on": {"file": frozen["selected_on"], "n_scans": frozen["selected_on_n_scans"]},
               "selected_config": p, "frozen_tau_by_fp_target": frozen["primary"]["frozen_tau_by_fp_target"],
               "baseline_config": {**cfg["baseline"], "tau": "-inf"}, "size_bins": bins}

    # A. Operating points with CIs: baseline (tau=-inf) and every frozen tau.
    m["baseline_operating_point"] = op_block(d_bl, -np.inf, bins, b)
    m["frozen_operating_points"] = {t: op_block(d_sel, tau, bins, b) for t, tau in frozen_tau.items()}
    # B. Threshold-free test FROC summaries.
    m["selected_curve_summary"] = curve_block(d_sel, bins, targets, b)
    m["baseline_curve_summary"] = curve_block(d_bl, bins, targets, b)
    # C. FROC CSVs + the full grid across all voxel thresholds and scorers (D5.1).
    rows, grid_rows = [], []
    for name, data in (("selected", d_sel), ("baseline_0.55_max_prob", d_bl)):
        rows += curve_rows(froc(data), f"{name}|all")
        for bn in bins:
            rows += curve_rows(froc(data, gt_mask=data.gt_bin == bn), f"{name}|{bn}")
    write_csv(out / "froc_selected_vs_baseline.csv", rows)
    grid_curves: dict = {}
    (out / "froc_grid").mkdir(exist_ok=True)
    for thr, scorer in product(comp_cfg["voxel_thresholds"], frozen_grid_scorers(cfg, frozen)):
        c = Config(sel.connectivity, thr, sel.rule, scorer, 0.0)
        data = detection_data(tables, test, c)
        curve = froc(data)
        grid_curves[(scorer, thr)] = curve
        s = summarize_curve(data, bins=bins, targets=targets)
        grid_rows.append({"voxel_threshold": thr, "scorer": scorer, **s})
        write_csv(out / "froc_grid" / f"{c.key()}.csv", curve_rows(curve, c.key()))
    write_csv(out / "grid_test_summary.csv", grid_rows)
    m["grid_test_summary"] = grid_rows
    # D. Secondary hit rules and 6-connectivity, at the selected (thr, scorer) and at baseline.
    m["secondary_rules"] = {}
    for rule in cfg["secondary_rules"]:
        cs = Config(sel.connectivity, sel.voxel_threshold, rule, sel.scorer, sel.min_esd_mm)
        cb = Config(bl.connectivity, bl.voxel_threshold, rule, bl.scorer, bl.min_esd_mm)
        m["secondary_rules"][rule] = {
            "selected_curve_summary": curve_block(detection_data(tables, test, cs), bins, targets, b),
            "selected_frozen_tau_1fp": op_block(detection_data(tables, test, cs), frozen_tau.get("1", -np.inf), bins, b),
            "baseline_operating_point": op_block(detection_data(tables, test, cb), -np.inf, bins, b),
        }
    c6 = cfg["connectivity_sensitivity"]
    m["connectivity_6"] = {
        "baseline_operating_point": op_block(detection_data(tables, test, Config(c6, bl.voxel_threshold, bl.rule, bl.scorer)), -np.inf, bins, b),
        "selected_curve_summary": curve_block(detection_data(tables, test, Config(c6, sel.voxel_threshold, sel.rule, sel.scorer, sel.min_esd_mm)), bins, targets, b),
    }
    # E. Bridge: baseline by prior size convention (longest axis, assumed 1 mm iso).
    m["bridge_baseline_by_longest_axis_1mm"] = op_block(
        detection_data(tables, test, bl, bin_field="bridge_bin"), -np.inf, bridge, b)
    # F. Optional min-size filter variant (D12).
    if "filtered" in frozen:
        f = frozen["filtered"]["config"]
        fc = Config(f["connectivity"], f["voxel_threshold"], f["rule"], f["scorer"], f["min_esd_mm"])
        d_f = detection_data(tables, test, fc)
        m["filtered"] = {"config": f, "frozen_tau_by_fp_target": frozen["filtered"]["frozen_tau_by_fp_target"],
                         "frozen_operating_points": {t: op_block(d_f, float(v["tau"]), bins, b)
                                                     for t, v in frozen["filtered"]["frozen_tau_by_fp_target"].items()},
                         "curve_summary": curve_block(d_f, bins, targets, b)}
    # G. FP characterisation.
    bands = [float(x) for x in cfg["fp_characterization"]["distance_bands_mm"]]
    m["fp_characterization"] = {
        "baseline": fp_characterization(fp_components(tables, test, bl, -np.inf), bands, bin_defs),
        **{f"selected_frozen_tau_{t}fp": fp_characterization(fp_components(tables, test, sel, tau), bands, bin_defs)
           for t, tau in frozen_tau.items()},
    }
    write_csv(out / "fp_components_baseline.csv", fp_components(tables, test, bl, -np.inf))
    # I. Edge-case counts.
    m["edge_counts"] = {"baseline": edge_counts(tables, test, bl), "selected": edge_counts(tables, test, sel)}
    # J. Strata (never used for selection).
    m["strata"] = []
    for st in cfg.get("strata", []):
        ids = read_ids(st["split"])
        st_tables = load_tables(comp_dir, ids)
        ds, db = detection_data(st_tables, ids, sel), detection_data(st_tables, ids, bl)
        m["strata"].append({"label": st["label"], "file": st["split"], "n_scans": len(ids),
                            "baseline_operating_point": op_block(db, -np.inf, bins, b),
                            "frozen_operating_points": {t: op_block(ds, tau, bins, b) for t, tau in frozen_tau.items()},
                            "selected_curve_summary": curve_block(ds, bins, targets, b)})
    # H. Segmentation secondary + HD95 test (parallel over cases).
    manifest = load_maps_manifest(comp_cfg["maps_manifest"])
    inputs = {manifest[c]["file"]: manifest[c]["sha256"] for c in test}
    if cfg["segmentation"]["enabled"]:
        specs = {"baseline": {"voxel_threshold": bl.voxel_threshold, "connectivity": bl.connectivity,
                              "scorer": "max_prob", "tau": -np.inf, "min_esd_mm": 0.0, "hd95_test": True}}
        for t, tau in frozen_tau.items():
            specs[f"selected_frozen_tau_{t}fp"] = {"voxel_threshold": sel.voxel_threshold,
                                                  "connectivity": sel.connectivity, "scorer": sel.scorer,
                                                  "tau": tau, "min_esd_mm": sel.min_esd_mm, "hd95_test": True}
        with ProcessPoolExecutor(max_workers=cfg["segmentation"]["workers"]) as pool:
            seg = list(pool.map(case_segmentation_metrics, test, [manifest[c]["file"] for c in test],
                                [comp_cfg["gt_root"]] * len(test), [comp_cfg["native_roots"]] * len(test),
                                [specs] * len(test), [comp_cfg["scorers"]] * len(test),
                                [cfg["segmentation"]["far_fp_cuts_mm"]] * len(test)))
        flat = [{"case": r["case"], "config": k, **v} for r in seg for k, v in r.items() if k != "case"]
        write_csv(out / "segmentation_per_case.csv", flat)
        m["segmentation_secondary"] = {}
        case_index = {c: i for i, c in enumerate(test)}
        for k in specs:
            keys = sorted({kk for r in seg for kk in r[k]})
            per = {kk: np.array([r[k].get(kk, np.nan) for r in sorted(seg, key=lambda r: case_index[r["case"]])],
                                dtype=float) for kk in keys}

            def stat(d, w, per=per):
                res = {}
                for kk, v in per.items():
                    ok = np.isfinite(v) & (w > 0)
                    res[kk] = float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else float("nan")
                return res
            m["segmentation_secondary"][k] = boot(d_sel, stat, b)

    m = clean_json(m)
    write_json(out / "metrics.json", m)
    write_json(out / "provenance.json", build_provenance(
        command="evaluate" + (" --reproduce" if args.reproduce else ""), repo_root=REPO, config=cfg,
        config_path=args.config,
        splits=[split_record("test", repo_path(cfg["test_split"]), test),
                split_record("selected_on", repo_path(frozen["selected_on"]), tuned_on)],
        inputs={**inputs, **{f"{comp_dir.relative_to(REPO)}/{c}.json.gz":
                             sha256_file(comp_dir / f"{c}.json.gz") for c in test},
                cfg["frozen"]: sha256_file(repo_path(cfg["frozen"]))},
        extra={"run_id": run_id, "frozen": frozen}))

    # Figures. Operating points are marked on the all-lesion panel.
    op_bl = operating_point(d_bl, -np.inf)
    marks = [("baseline point: 0.55, all components", op_bl["fp_per_scan"], op_bl["sensitivity"])]
    for t, tau in frozen_tau.items():
        op = operating_point(d_sel, tau)
        marks.append((f"frozen tau @ {t} FP/scan", op["fp_per_scan"], op["sensitivity"]))
    panels = [("All lesions", [("selected config", froc(d_sel)), ("baseline (0.55, max_prob)", froc(d_bl))],
               len(d_sel.gt_score))]
    for bn in bins:
        n = int((d_sel.gt_bin == bn).sum())
        panels.append((f"GT ESD {bn} mm", [("selected config", froc(d_sel, gt_mask=d_sel.gt_bin == bn)),
                                           ("baseline (0.55, max_prob)", froc(d_bl, gt_mask=d_bl.gt_bin == bn))], n))
    plot_froc_comparison(panels, out / "froc_selected_vs_baseline.png",
                         points=[marks] + [[] for _ in bins])
    grid_panels = [(f"{s} (test, {cfg['test_label'].split(' (')[0]})",
                    [(thr, grid_curves[(s, thr)]) for thr in comp_cfg["voxel_thresholds"]],
                    sel.voxel_threshold if s == sel.scorer else None)
                   for s in frozen_grid_scorers(cfg, frozen)]
    plot_threshold_grid(grid_panels, out / "froc_test_full_grid.png")

    print(f"wrote {out}")
    if args.reproduce:
        orig = repo_path(cfg["results_dir"]) / run_id / "metrics.json"
        if orig.exists():
            a, c = json.loads(orig.read_text()), json.loads((out / "metrics.json").read_text())
            a.pop("run_id", None)
            c.pop("run_id", None)
            print("REPRODUCED: identical metrics" if a == c else "NOT IDENTICAL: see diff")
    return 0


def frozen_grid_scorers(cfg, frozen) -> list[str]:
    sel_cfg = load_yaml(repo_path(frozen["selection_config"]))
    return list(sel_cfg["grid"]["scorers"])


if __name__ == "__main__":
    raise SystemExit(main())
