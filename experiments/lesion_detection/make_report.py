"""Render results/<run_id>/REPORT.md from metrics.json and the side analyses.

The report never states a sensitivity without its FP/scan and its size breakdown,
and every per-bin number carries its lesion count. Prose interpretation comes from
``INTERPRETATION.md`` in the same directory, written after the numbers were
examined. It is inserted verbatim and labelled as interpretation.

    python experiments/lesion_detection/make_report.py --run experiments/lesion_detection/results/<run_id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import repo_path

from jannus.evaluation.detection.report import fmt_ci, md_table

NA = "n/a"


def f(x, d=3):
    if x is None or isinstance(x, str):
        return NA if x is None else x
    return f"{x:.{d}f}"


def ci(block: dict, key: str, d: int = 3) -> str:
    p = block["point"].get(key)
    if p is None:
        return NA
    lo, hi = block["ci_low"].get(key), block["ci_high"].get(key)
    return fmt_ci(p, lo if lo is not None else float("nan"), hi if hi is not None else float("nan"), d)


def counts(block: dict, b: str) -> str:
    n = block["point"].get(f"n_lesions[{b}]")
    k = block["point"].get(f"n_detected[{b}]")
    return f"{int(k)}/{int(n)}" if n else "0/0"


def op_rows(label: str, block: dict, bins: list[str]) -> list:
    pt = block["point"]
    return [label, ci(block, "sensitivity"), f"{int(pt['n_detected'])}/{int(pt['n_lesions'])}",
            ci(block, "fp_per_scan", 2), *[f"{ci(block, f'sensitivity[{b}]', 2)} ({counts(block, b)})" for b in bins],
            ci(block, "scan_any_flagged", 2), ci(block, "scan_all_flagged", 2)]


def op_table(entries: list[tuple[str, dict]], bins: list[str]) -> str:
    header = ["Operating point", "Sensitivity [95% CI]", "Lesions detected", "FP/scan [95% CI]",
              *[f"Sens. {b} mm (k/n)" for b in bins], "Scans: any flagged", "Scans: all flagged"]
    return md_table(header, [op_rows(lbl, blk, bins) for lbl, blk in entries])


def curve_table(entries: list[tuple[str, dict]], targets: list[str], bins: list[str]) -> str:
    header = ["Curve", "CPM", *[f"Sens @ {t} FP/scan" for t in targets]]
    rows = []
    for lbl, blk in entries:
        cells = [lbl, ci(blk, "cpm")]
        for t in targets:
            reached = blk["point"].get(f"reached@{t}")
            cells.append(ci(blk, f"sens@{t}") + ("" if reached else " (curve ends below)"))
        rows.append(cells)
    return md_table(header, rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--secondary", default="experiments/lesion_detection/results/secondary_2fold_pooled")
    ap.add_argument("--brats", default="experiments/lesion_detection/results/brats_side_analysis")
    ap.add_argument("--gate", default="experiments/lesion_detection/results/repro_gate")
    args = ap.parse_args()
    run = repo_path(args.run)
    m = json.loads((run / "metrics.json").read_text())
    prov = json.loads((run / "provenance.json").read_text())
    bins = m["size_bins"]
    targets = list(m["frozen_tau_by_fp_target"])
    sel = m["selected_config"]
    L: list[str] = []
    w = L.append

    w(f"# Lesion-level detection evaluation: {m['run_id']}\n")
    w(f"Generated from `metrics.json`. Commit `{prov['git'].get('commit', '?')[:12]}`"
      f"{' (dirty tree)' if prov['git'].get('dirty') else ''}, based on tag `{prov['git']['base_tag']}` "
      f"(`{prov['git']['base_commit'][:12]}`). Every number below names the split it came from.\n")
    w("> **Read this first.**")
    w("> - **This repository has no untouched test set.** The 84-case cohort was previously used to choose "
      "the 0.55 threshold (Dice sweep) and to choose the stacker variant.")
    w("> - The cohort also overlaps training (LEAKAGE_FINDING.md):")
    w(">   - 24 scans are in nnU-Net fold-0 training;")
    w(">   - 33 of its 77 patients have other timepoints in the stacker's training pool.")
    w(f"> - The **test split here is the {m['split']['label']}**. Even \"clean\" is an upper bound, because "
      "the patch models' training sets and the provenance of the synthetic BMS cases are unknown.")
    w(f"> - The operating point was chosen only on the {m['selected_on']['n_scans']} contaminated scans, "
      "then evaluated once here.")
    w("> - Sizes and distances use true voxel spacing recovered from native headers (SPACING_DEFECT.md). "
      "They are **not** comparable to prior JANNUS size bins, ≥10 mm-only figures or HD95. "
      "Bridging numbers are given in the section on secondary rules and conventions.\n")

    w("## Configuration\n")
    w(md_table(["", "Connectivity", "Voxel threshold", "Hit rule", "Per-lesion score", "Min ESD (mm)", "Per-lesion τ"], [
        ["Baseline (production)", 26, f(m["baseline_config"]["voxel_threshold"], 2), "overlap ≥1 voxel",
         "none (all components kept)", 0, "-inf (all kept)"],
        ["Selected (on contaminated scans)", sel["connectivity"], f(sel["voxel_threshold"], 2), sel["rule"],
         sel["scorer"], sel["min_esd_mm"], "frozen per FP target (below)"],
    ]))
    w("")
    w(md_table(["FP target (tuning)", "Frozen τ", "Tuning sensitivity", "Tuning curve reached target"],
               [[t, f(v["tau"], 4), f(v["tuning_sensitivity"]), "yes" if v["reached"] else "no (ends below)"]
                for t, v in m["frozen_tau_by_fp_target"].items()]))
    w(f"\nTest split: `{m['split']['file']}`, {m['split']['n_scans']} scans / {m['split']['n_patients']} patients. "
      f"Bootstrap: {m['baseline_operating_point']['n_resamples']} patient-level resamples, "
      f"seed {m['baseline_operating_point']['seed']}.\n")

    w("## Headline: baseline vs selected operating points on the test split\n")
    w("The selected rows apply the τ frozen on the tuning split to the test scans. These are deployable "
      "operating points, so the realised FP/scan on test can differ from the target. Every sensitivity "
      "carries its FP/scan and size breakdown, and each per-bin cell shows detected/total GT lesions.\n")
    entries = [("Baseline: 0.55, all components", m["baseline_operating_point"])]
    entries += [(f"Selected, τ frozen @ {t} FP/scan", m["frozen_operating_points"][t]) for t in targets]
    w(op_table(entries, bins))
    n_bins = {b: int(m["baseline_operating_point"]["point"].get(f"n_lesions[{b}]", 0)) for b in bins}
    w(f"\n**Per-bin GT lesion counts on the test split:** {', '.join(f'{b} mm: {n}' for b, n in n_bins.items())}. "
      "These bins are **underpowered**. With so few lesions per bin, and lesions clustered within patients, "
      "the confidence intervals are wide, and differences between configurations in the small bins "
      "should not be read as established.\n")

    w("## Threshold-free summary: sensitivity read off the test FROC\n")
    w("This is the standard FROC reading: sensitivity at each FP rate on the test curve itself, "
      "with no operating point carried over. It describes the curve, not a deployable setting.\n")
    w(curve_table([("Selected config", m["selected_curve_summary"]),
                   ("Baseline config (0.55; max_prob as score)", m["baseline_curve_summary"])], targets, bins))
    w("")
    w(md_table(["Curve", *[f"{b} mm @ {t} FP" for t in ("1", "2") for b in bins]],
               [[lbl, *[f"{ci(blk, f'sens@{t}[{b}]', 2)} (n={n_bins[b]})" for t in ("1", "2") for b in bins]]
                for lbl, blk in (("Selected", m["selected_curve_summary"]),
                                 ("Baseline", m["baseline_curve_summary"]))]))
    w("\n![FROC, selected vs baseline, overall and per size bin](froc_selected_vs_baseline.png)\n")

    w("## Full test-set FROC across every voxel threshold and scorer\n")
    w("This is shown because a threshold tuned on leaked cases may be miscalibrated for clean data. "
      "If another threshold or scorer clearly dominates the selected one *on test*, that is evidence of "
      "miscalibration. It is **not** a licence to switch, because switching would be tuning on test.\n")
    w("![Test FROC grid](froc_test_full_grid.png)\n")
    g = sorted(m["grid_test_summary"], key=lambda r: (r["scorer"], r["voxel_threshold"]))
    w(md_table(["Scorer", "Voxel thr", "CPM", *[f"Sens @ {t}" for t in targets]],
               [[r["scorer"], f(r["voxel_threshold"], 2), f(r["cpm"]),
                 *[f(r[f"sens@{t}"]) + ("" if r[f"reached@{t}"] else "*") for t in targets]] for r in g]))
    w("\n`*` means the curve ends below that FP rate; the value is the end-of-curve sensitivity.\n")

    w("## False positives\n")
    fpc = m["fp_characterization"]
    w(md_table(["Operating point", "FPs", "ESD median (mm)", *[f"ESD {b}" for b in bins],
                *[f"{k} mm from GT" for k in fpc["baseline"]["by_distance_band_mm"]]],
               [[k, v["n_fp"], f((v.get("esd_mm_quantiles") or {}).get("0.5"), 1),
                 *[v["by_size_bin"][b] for b in bins], *v["by_distance_band_mm"].values()]
                for k, v in fpc.items()]))
    w("\nDistances are from the nearest FP voxel to the nearest GT voxel, in true mm. An FP in a scan "
      "with no GT lesion counts in the last band.\n")
    if "segmentation_secondary" in m:
        s = m["segmentation_secondary"]["baseline"]
        w("**HD95 test (baseline mask, true spacing; mean over scans [95% CI]):**\n")
        rows = [[k.replace("hd95_mm", "HD95").replace("_", " "), ci(m["segmentation_secondary"]["baseline"], k, 2)]
                for k in s["point"] if k.startswith("hd95_mm")]
        w(md_table(["Variant", "mm"], rows))
        w("")

    w("## Secondary: segmentation metrics for the same configurations\n")
    if "segmentation_secondary" in m:
        w(md_table(["Configuration", "Voxel Dice", "HD95 (mm, true spacing)", "HD95 (mm, old 1 mm assumption)"],
                   [[k, ci(v, "dice"), ci(v, "hd95_mm", 2), ci(v, "hd95_mm_1mm_assumption", 2)]
                    for k, v in m["segmentation_secondary"].items()]))
    w("\nDice and HD95 are secondary here. Removing low-score components can change them in either direction.\n")

    w("## Secondary hit rules, connectivity, and bridging to prior conventions\n")
    rows = [("Primary: overlap ≥1 voxel, 26-conn, baseline", m["baseline_operating_point"])]
    for rule, r in m["secondary_rules"].items():
        rows.append((f"{rule}, baseline", r["baseline_operating_point"]))
        rows.append((f"{rule}, selected τ @ 1 FP", r["selected_frozen_tau_1fp"]))
    rows.append(("6-connectivity, baseline", m["connectivity_6"]["baseline_operating_point"]))
    w(op_table(rows, bins))
    br = m["bridge_baseline_by_longest_axis_1mm"]
    bb = [k[len("sensitivity["):-1] for k in br["point"] if k.startswith("sensitivity[")]
    w("\n**Bridge to prior size convention.** The baseline is shown again, binned by longest bounding-box "
      "axis at the old 1 mm isotropic assumption (prior JANNUS convention). Lesion membership per bin "
      "differs from the ESD bins above:\n")
    w(op_table([("Baseline, prior bins", br)], bb))
    w(f"\nEdge cases on test, counted with no score filter: baseline {m['edge_counts']['baseline']}; "
      f"selected {m['edge_counts']['selected']}.\n")

    if "filtered" in m:
        w("## Optional minimum-size filter (chosen on the tuning split)\n")
        fc = m["filtered"]["config"]
        w(f"Filtered variant: {fc}. Its effect on FP/scan and small-lesion recall at the same FP targets:\n")
        w(op_table([(f"Filtered, τ frozen @ {t}", m["filtered"]["frozen_operating_points"][t]) for t in targets]
                   + [(f"Unfiltered, τ frozen @ {t}", m["frozen_operating_points"][t]) for t in targets], bins))
        w("")

    for st in m.get("strata", []):
        w(f"## Stratum: {st['label']}\n")
        w("Shown for completeness. These scans were used for selection, so the selected rows are **in-sample**.\n")
        w(op_table([("Baseline: 0.55, all components", st["baseline_operating_point"])]
                   + [(f"Selected, τ @ {t}", st["frozen_operating_points"][t]) for t in targets], bins))
        w("")

    sec = repo_path(args.secondary) / "metrics.json"
    if sec.exists():
        s = json.loads(sec.read_text())
        w("## Secondary analysis: patient-grouped 2-fold over all 84 scans\n")
        w("Each fold is evaluated with the configuration frozen on the other fold, then pooled out-of-fold. "
          "Both folds mix clean and contaminated scans, so this inherits the leakage. It is a sensitivity "
          "analysis, not a headline. Per-direction details are in the fold result directories.\n")
        w(op_table([("Baseline: 0.55, all components (84 scans)", s["baseline_operating_point"])]
                   + [(f"Pooled out-of-fold, τ @ {t}", s["pooled_frozen_operating_points"][t]) for t in targets],
                   bins))
        w("")
    brats = repo_path(args.brats) / "summary.json"
    if brats.exists():
        b = json.loads(brats.read_text())
        w("## Supplementary: original vs BraTS-revised annotation\n")
        w(f"This covers {b['n_scans']} eval-cohort UCSF scans that have the BraTS-METS revision "
          f"({b['n_clean']} clean). Lesions are paired across annotations by overlap. "
          "A *changed pair* is one whose detection status differs between the two annotations. "
          f"{b['only_in_A_total']} lesions exist only in the original annotation and "
          f"{b['only_in_C3_total']} only in the revision.\n")
        w(md_table(["Operating point", "Sens. original", "Sens. revised", "Changed pairs (total)",
                    "Scans with ≥1 change", "Mean changes / scan", "Max / scan"],
                   [[k, f"{f(v['sensitivity_A'])} ({v['detected_A']}/{v['lesions_A']})",
                     f"{f(v['sensitivity_C3'])} ({v['detected_C3']}/{v['lesions_C3']})", v["total_changed_pairs"],
                     v["scans_with_any_change"], f(v["changed_per_scan_mean"], 2), v["changed_per_scan_max"]]
                    for k, v in b.items() if isinstance(v, dict)]))
        w("")
    gate = repo_path(args.gate) / "metrics.json"
    if gate.exists():
        gm = json.loads(gate.read_text())
        w("## Reproduction gate\n")
        w(md_table(["Check", "Logged", "Observed", "Δ", "Tolerance", "Result"],
                   [[k, v["logged"], f(v["observed"], 4), f(v["delta"], 4), v["tolerance"],
                     "PASS" if v["pass"] else "FAIL"] for k, v in gm["checks"].items()]))
        w("")

    interp = run / "INTERPRETATION.md"
    if interp.exists():
        w("## Interpretation (written after examining the numbers above)\n")
        w(interp.read_text().strip() + "\n")

    w("## Limitations\n")
    w("- **Standalone FROC does not establish reader benefit.** These numbers describe the algorithm alone. "
      "Any claim that JANNUS helps radiologists find lesions (a CADe claim) needs a reader study: "
      "multi-reader, multi-case, with and without the aid. Nothing here substitutes for one.")
    w(f"- **Small samples.** The test split has {m['split']['n_scans']} scans from {m['split']['n_patients']} "
      f"patients, with {', '.join(f'{n} lesions {b} mm' for b, n in n_bins.items())}. "
      "The sub-5 mm results, which matter most clinically, rest on few lesions from few patients, and their "
      "CIs are wide. They are reported, not established.")
    w("- **No untouched test set.** The 84 scans informed the 0.55 threshold and the stacker choice, and they "
      "overlap base-model and stacker training at scan and patient level. The \"clean\" subset is clean only "
      "as far as this repo can show (patch-model training data and synthetic-case provenance are unknown).")
    w("- **The selection set is leaked.** The operating point was tuned on contaminated scans, where base "
      "models may be over-confident. The full test grid is shown so any miscalibration is visible.")
    w("- **Ground truth is one annotation convention.** The supplementary BraTS comparison shows how much "
      "lesion-level detection status depends on which annotation is used.")
    w("- **Geometry.** Sizes use spacing reconstructed from native headers for volumes that were resized "
      "without correct affines (SPACING_DEFECT.md). The reconstruction is exact for the zoom used, but the "
      "256³ grids are themselves resampled, so very small lesions are distorted by the resampling.")
    w("- **Scope.** Single retrospective cohort, BMS and UCSF scanners only. No prospective or external data. "
      "This experiment does not change the intended use (docs/INTENDED_USE.md is untouched).\n")

    w("## Reproduce\n")
    w("See `experiments/lesion_detection/README.md`. Inputs to this run, with SHA-256 hashes, are in "
      "`provenance.json` (`inputs_sha256`, `checkpoints`, `splits`).\n")
    (run / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {run / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
