"""Choose the operating point on a TUNING split and freeze it (D6, D5.1).

The grid is voxel threshold x scorer, at fixed connectivity and hit rule. The
objective is CPM, the mean sensitivity at 0.5, 1, 2 and 3 FP/scan. Ties go to
the higher voxel threshold, then to the scorer listed first in the config.

For the chosen configuration, the per-lesion score threshold τ that meets each
FP target on the tuning split is also frozen. The test evaluation reports both:

- the test FROC, a threshold-free summary;
- the frozen τ applied to test, the deployable operating point, which shows
  whether the calibration carries over.

When ``filtering.enabled`` is set, a second frozen variant is selected that also
searches a minimum component ESD (D12). The primary choice is unaffected.

This script reads ONLY the tuning split. It refuses to run if that split
overlaps any split listed in ``must_be_disjoint_from``.

    python experiments/lesion_detection/select_operating_point.py --config experiments/lesion_detection/configs/10_select_primary.yaml
"""

from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path

import yaml
from common import REPO, load_yaml, read_ids, repo_path
from tables import Config, detection_data, load_tables

from jannus.evaluation.detection.metrics import froc, summarize_curve
from jannus.evaluation.detection.provenance import (
    build_provenance,
    config_hash,
    split_record,
    write_json,
)


def components_dir(components_config: str) -> Path:
    cc = load_yaml(repo_path(components_config))
    return repo_path(cc["output_dir"]) / config_hash(cc)[:12]


def size_bins(components_config: str) -> list[str]:
    return [b[0] for b in load_yaml(repo_path(components_config))["size_bins"]]


def evaluate_grid(tables, cases, grid, esd_values, targets, bins) -> list[dict]:
    rows = []
    for thr, scorer, esd in product(grid["voxel_thresholds"], grid["scorers"], esd_values):
        cfg = Config(grid["connectivity"], thr, grid["rule"], scorer, esd)
        data = detection_data(tables, cases, cfg)
        s = summarize_curve(data, bins=bins, targets=targets)
        rows.append({"key": cfg.key(), "connectivity": cfg.connectivity, "voxel_threshold": thr,
                     "rule": cfg.rule, "scorer": scorer, "min_esd_mm": esd, **s})
    return rows


def choose(rows: list[dict], scorer_order: list[str]) -> dict:
    # Max CPM; tie -> higher voxel threshold -> earlier scorer -> smaller min ESD.
    return max(rows, key=lambda r: (round(r["cpm"], 12), r["voxel_threshold"],
                                    -scorer_order.index(r["scorer"]), -r["min_esd_mm"]))


def freeze(choice: dict, tables, cases, targets) -> dict:
    cfg = Config(choice["connectivity"], choice["voxel_threshold"], choice["rule"],
                 choice["scorer"], choice["min_esd_mm"])
    curve = froc(detection_data(tables, cases, cfg))
    taus = {}
    for t in targets:
        sens, tau, reached = curve.sensitivity_at_fp(t)
        taus[f"{t:g}"] = {"tau": tau, "tuning_sensitivity": sens, "reached": reached}
    return {"config": {"connectivity": cfg.connectivity, "voxel_threshold": cfg.voxel_threshold,
                       "rule": cfg.rule, "scorer": cfg.scorer, "min_esd_mm": cfg.min_esd_mm},
            "tuning_cpm": choice["cpm"], "frozen_tau_by_fp_target": taus}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    cases = read_ids(cfg["tune_split"])
    for other in cfg.get("must_be_disjoint_from", []):
        overlap = set(cases) & set(read_ids(other))
        if overlap:
            raise SystemExit(f"tuning split overlaps {other}: {sorted(overlap)[:5]} -- refusing")

    comp_dir = components_dir(cfg["components_config"])
    tables = load_tables(comp_dir, cases)
    bins, targets, grid = size_bins(cfg["components_config"]), cfg["fp_targets"], cfg["grid"]

    rows = evaluate_grid(tables, cases, grid, [0.0], targets, bins)
    primary = choose(rows, grid["scorers"])
    frozen = {"name": cfg["name"], "selected_on": cfg["tune_split"],
              "selected_on_n_scans": len(cases), "selection_config": str(args.config),
              "selection_config_sha256": config_hash(cfg), "components_dir": str(comp_dir.relative_to(REPO)),
              "primary": freeze(primary, tables, cases, targets)}

    if cfg.get("filtering", {}).get("enabled"):
        frows = evaluate_grid(tables, cases, grid, cfg["filtering"]["min_esd_mm"], targets, bins)
        frozen["filtered"] = freeze(choose(frows, grid["scorers"]), tables, cases, targets)
        write_csv(repo_path(cfg["results_dir"]) / "grid_with_filtering.csv", frows)

    out = repo_path(cfg["results_dir"])
    write_csv(out / "grid.csv", rows)
    frozen_path = repo_path(cfg["frozen_out"])
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_path.write_text(yaml.safe_dump(frozen, sort_keys=False))
    write_json(out / "provenance.json", build_provenance(
        command="select_operating_point", repo_root=REPO, config=cfg, config_path=args.config,
        splits=[split_record("tune", repo_path(cfg["tune_split"]), cases)],
        extra={"frozen_out": str(frozen_path.relative_to(REPO)), "frozen": frozen,
               "components_dir": str(comp_dir.relative_to(REPO))},
    ))
    p = frozen["primary"]
    print(f"[{cfg['name']}] selected on {len(cases)} tuning scans: {p['config']} "
          f"CPM {p['tuning_cpm']:.4f}")
    for t, v in p["frozen_tau_by_fp_target"].items():
        print(f"  frozen tau @ {t} FP/scan: {v['tau']:.4f} (tuning sens {v['tuning_sensitivity']:.3f}"
              f"{'' if v['reached'] else ', curve ends early'})")
    if "filtered" in frozen:
        print(f"  filtered variant: {frozen['filtered']['config']} CPM {frozen['filtered']['tuning_cpm']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
