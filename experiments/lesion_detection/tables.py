"""Load component tables and turn them into DetectionData for one configuration.

A *configuration* is (connectivity, voxel threshold, hit rule, scorer, min ESD).
The optional min-ESD filter (D12) drops predicted components smaller than the
cut before scoring. Each lesion's score is then recomputed from the components
that hit it and survive the cut, so a lesion hit only by a dropped component
becomes a miss.

Split discipline: :func:`load_tables` loads exactly the case list it is given.
The callers (select / evaluate) pass a split file and nothing else, and each
refuses a case list that does not match its declared split.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from common import patient_id

from jannus.evaluation.detection.metrics import DetectionData, concat_cases


@dataclass(frozen=True)
class Config:
    connectivity: int
    voxel_threshold: float
    rule: str
    scorer: str
    min_esd_mm: float = 0.0

    def key(self) -> str:
        return (f"c{self.connectivity}_t{self.voxel_threshold:.2f}_{self.rule}_{self.scorer}"
                f"_esd{self.min_esd_mm:g}")


def load_tables(components_dir: Path, cases: list[str]) -> dict[str, dict]:
    out = {}
    for c in cases:
        path = components_dir / f"{c}.json.gz"
        if not path.exists():
            raise SystemExit(f"missing component table for {c}: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            out[c] = json.load(fh)
    return out


def _table(case_tables: dict, cfg: Config) -> dict:
    for t in case_tables["tables"]:
        if (t["connectivity"] == cfg.connectivity and t["rule"] == cfg.rule
                and abs(t["voxel_threshold"] - cfg.voxel_threshold) < 1e-9):
            return t
    raise KeyError(f"{case_tables['case']}: no table for {cfg}")


def case_records(case_tables: dict, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(gt_score, gt_bin, fp_score)`` for one case under ``cfg``."""
    t = _table(case_tables, cfg)
    key = f"score_{cfg.scorer}"
    comps = {c["comp"]: c for c in t["components"]}
    kept = {cid for cid, c in comps.items() if (c["esd_mm"] or 0.0) >= cfg.min_esd_mm}
    gt_score = np.array([
        max((comps[h][key] for h in les["hits"] if h in kept), default=-np.inf)
        for les in t["lesions"]
    ], dtype=np.float64)
    gt_bin = np.array([les["bin"] for les in t["lesions"]], dtype=object)
    fp_score = np.array([c[key] for cid, c in comps.items() if c["is_fp"] and cid in kept],
                        dtype=np.float64)
    return gt_score, gt_bin, fp_score


def detection_data(tables: dict[str, dict], cases: list[str], cfg: Config,
                   *, bin_field: str = "bin") -> DetectionData:
    rows = []
    for c in cases:
        gs, gb, fs = case_records(tables[c], cfg)
        if bin_field != "bin":
            gb = np.array([les[bin_field] for les in _table(tables[c], cfg)["lesions"]], dtype=object)
        rows.append((c, patient_id(c), gs, gb, fs))
    return concat_cases(rows)


def fp_components(tables: dict[str, dict], cases: list[str], cfg: Config, tau: float) -> list[dict]:
    """FP components surviving ``tau`` and the min-ESD cut, tagged with their case (D8)."""
    out = []
    for c in cases:
        t = _table(tables[c], cfg)
        for comp in t["components"]:
            if (comp["is_fp"] and comp[f"score_{cfg.scorer}"] >= tau
                    and (comp["esd_mm"] or 0.0) >= cfg.min_esd_mm):
                out.append({"case": c, **comp})
    return out


def edge_counts(tables: dict[str, dict], cases: list[str], cfg: Config) -> dict[str, int]:
    """Split, merge, redundant and fragment counts summed over cases, with no score filter."""
    total: dict[str, int] = {}
    for c in cases:
        for k, v in _table(tables[c], cfg)["counts"].items():
            total[k] = total.get(k, 0) + int(v)
    return total
