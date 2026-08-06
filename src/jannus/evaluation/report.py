"""Render evaluation results into shareable reports.

The Markdown report is the artefact a validation site sends back. It is
deliberately self-contained and PHI-free: pseudonymised case tokens, aggregate
metrics, the provenance block, and the site's QC warnings. A coordinating site
can read it without access to the imaging.
"""

from __future__ import annotations

from typing import Any

from .harness import MEASURABLE_DISEASE_MM, CohortResult

#: Published JANNUS figures on the 84-case internal held-out cohort. Shown
#: beside a site's numbers purely as a reference point — a site whose numbers
#: differ has not necessarily done anything wrong, since cohort difficulty,
#: annotation style and acquisition protocol all move these.
REFERENCE_INTERNAL = {
    "voxel_dice": 0.7858,
    "lesion_sensitivity_all": 0.810,
    "fp_per_case_all": 1.68,
    "lesion_sensitivity_measurable": 0.943,
    "lesion_dice_measurable": 0.835,
    "hd95_measurable": 17.26,
}

#: Cleared predicate device, for context in the measurable-disease scope.
REFERENCE_PREDICATE = {
    "name": "Neosoma Brain Mets (K252922)",
    "lesion_sensitivity_measurable": 0.90,
    "lesion_dice_measurable": 0.86,
    "hd95_measurable": 1.78,
}


def _fmt(value: float, places: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        if value != value:  # NaN
            return "n/a"
    except TypeError:
        return "n/a"
    return f"{value:.{places}f}"


def _ci(entry: dict[str, float], places: int = 3) -> str:
    if not entry:
        return "n/a"
    mean = entry.get("mean")
    lo, hi = entry.get("ci_low"), entry.get("ci_high")
    if mean is None or mean != mean:
        return "n/a"
    if lo is None or lo != lo:
        return _fmt(mean, places)
    return f"{_fmt(mean, places)} ({_fmt(lo, places)}–{_fmt(hi, places)})"  # noqa: RUF001 - en dash is intentional in report tables


def render_text_summary(result: CohortResult) -> str:
    """Compact terminal summary shown at the end of `jannus evaluate`."""
    head = result.headline()
    lines = [
        f"Cases evaluated : {result.n_cases}" + (f"  ({result.n_failed} failed)" if result.n_failed else ""),
        "",
        "All lesions:",
        f"  voxel Dice           {_ci(result.voxel_dice_all, 4)}",
        f"  lesion sensitivity   {_ci(result.aggregate_all.get('lesion_wise_sensitivity', {}))}",
        f"  false positives/case {_ci(result.aggregate_all.get('fp_per_case', {}), 2)}",
        "",
        f"RANO-BM measurable disease (>= {MEASURABLE_DISEASE_MM:.0f} mm longest axis):",
        f"  lesion sensitivity   {_ci(result.aggregate_measurable.get('lesion_wise_sensitivity', {}))}",
        f"  lesion Dice(matched) {_ci(result.aggregate_measurable.get('lesion_wise_dice_matched', {}))}",
        f"  HD95 (mm)            {_ci(result.aggregate_measurable.get('hd95_mm', {}), 2)}",
        "",
        "Reference (internal 84-case held-out):",
        f"  voxel Dice {REFERENCE_INTERNAL['voxel_dice']:.4f}   "
        f"measurable sensitivity {REFERENCE_INTERNAL['lesion_sensitivity_measurable']:.3f}",
    ]
    delta = head["voxel_dice"] - REFERENCE_INTERNAL["voxel_dice"]
    if delta == delta:
        lines.append(f"  this cohort differs by {delta:+.4f} voxel Dice")
    return "\n".join(lines)


def render_markdown_report(
    result: CohortResult,
    *,
    site_name: str = "(unnamed site)",
    provenance: dict[str, Any] | None = None,
    validation_summary: dict[str, Any] | None = None,
) -> str:
    """Full Markdown report for return to the coordinating site."""
    prov = provenance or {}
    lines: list[str] = []
    add = lines.append

    add(f"# JANNUS external validation report — {site_name}")
    add("")
    add(f"- JANNUS version: `{prov.get('jannus_version', 'unknown')}`")
    add(f"- Pipeline revision: `{prov.get('pipeline_revision', 'unknown')}`")
    git = prov.get("git", {})
    if git.get("available"):
        dirty = " **(uncommitted changes present)**" if git.get("dirty") else ""
        add(f"- Source commit: `{str(git.get('commit', ''))[:12]}`{dirty}")
    det = prov.get("determinism", {})
    if det:
        add(
            f"- Deterministic execution: "
            f"{'yes' if det.get('fully_deterministic') else 'PARTIAL — see provenance.json'}"
        )
    threshold = (prov.get("config", {}) or {}).get("inference", {}).get("threshold")
    if threshold is not None:
        add(f"- Operating threshold: `{threshold}`")
    add(f"- Cases evaluated: **{result.n_cases}**"
        + (f" ({result.n_failed} failed)" if result.n_failed else ""))
    add("")
    add("All case identifiers below are salted pseudonyms. This report contains "
        "no protected health information.")
    add("")

    add("## Headline metrics")
    add("")
    add("| Metric | This site | Internal reference (84 cases) |")
    add("|---|---|---|")
    add(f"| Voxel Dice | {_ci(result.voxel_dice_all, 4)} | {REFERENCE_INTERNAL['voxel_dice']:.4f} |")
    add(f"| Lesion sensitivity (all) | {_ci(result.aggregate_all.get('lesion_wise_sensitivity', {}))} "
        f"| {REFERENCE_INTERNAL['lesion_sensitivity_all']:.3f} |")
    add(f"| False positives / case | {_ci(result.aggregate_all.get('fp_per_case', {}), 2)} "
        f"| {REFERENCE_INTERNAL['fp_per_case_all']:.2f} |")
    add("")

    add(f"## RANO-BM measurable disease (longest axis >= {MEASURABLE_DISEASE_MM:.0f} mm)")
    add("")
    add("This is the scope of the proposed Indication for Use.")
    add("")
    add("| Metric | This site | Internal reference | Predicate |")
    add("|---|---|---|---|")
    add(f"| Lesion sensitivity | {_ci(result.aggregate_measurable.get('lesion_wise_sensitivity', {}))} "
        f"| {REFERENCE_INTERNAL['lesion_sensitivity_measurable']:.3f} "
        f"| {REFERENCE_PREDICATE['lesion_sensitivity_measurable']:.2f} |")
    add(f"| Lesion Dice (matched) | {_ci(result.aggregate_measurable.get('lesion_wise_dice_matched', {}))} "
        f"| {REFERENCE_INTERNAL['lesion_dice_measurable']:.3f} "
        f"| {REFERENCE_PREDICATE['lesion_dice_measurable']:.2f} |")
    add(f"| HD95 (mm) | {_ci(result.aggregate_measurable.get('hd95_mm', {}), 2)} "
        f"| {REFERENCE_INTERNAL['hd95_measurable']:.2f} "
        f"| {REFERENCE_PREDICATE['hd95_measurable']:.2f} |")
    add("")
    add(f"Predicate values are from {REFERENCE_PREDICATE['name']} as published, on that "
        "device's own cohort. They are context, not a paired comparison.")
    add("")

    if result.stratified:
        add("## Stratified by lesion size (longest axis)")
        add("")
        add("| Bin | GT lesions | Sensitivity | FP/case | Mean DSC |")
        add("|---|---|---|---|---|")
        for label, stats in result.stratified.items():
            add(
                f"| {label} | {int(stats.get('n_gt_total', 0))} "
                f"| {_fmt(stats.get('sensitivity'))} "
                f"| {_fmt(stats.get('fp_per_case'), 2)} "
                f"| {_fmt(stats.get('mean_dsc'))} |"
            )
        add("")
        add("Sensitivity below 3 mm sits at the floor of MRI resolution and annotator "
            "agreement; it is reported for completeness, not as a performance claim.")
        add("")

    if validation_summary:
        add("## Dataset QC")
        add("")
        for key, value in validation_summary.items():
            add(f"- {key}: {value}")
        add("")

    if result.failures:
        add("## Per-case failures")
        add("")
        for failure in result.failures:
            add(f"- `{failure['case']}`: {failure['reason']}")
        add("")

    add("## Per-case metrics")
    add("")
    add("| Case | Voxel Dice | Sens (all) | FP | Sens (measurable) |")
    add("|---|---|---|---|---|")
    for case in result.per_case:
        add(
            f"| `{case.case_token}` | {_fmt(case.voxel_dice, 4)} "
            f"| {_fmt(case.all_lesions.get('lesion_wise_sensitivity'))} "
            f"| {_fmt(case.all_lesions.get('fp_per_case'), 0)} "
            f"| {_fmt(case.measurable.get('lesion_wise_sensitivity'))} |"
        )
    add("")
    add("---")
    add("")
    add("Generated by JANNUS. Not a medical device. Research use only — see "
        "`docs/INTENDED_USE.md`.")
    return "\n".join(lines)
