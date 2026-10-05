"""Render evaluation results into shareable reports.

The Markdown report is the artefact a validation site sends back. It is
deliberately self-contained and PHI-free: pseudonymised case tokens, aggregate
metrics, the provenance block, and the site's QC warnings. A coordinating site
can read it without access to the imaging.
"""

from __future__ import annotations

from typing import Any

from .harness import MEASURABLE_DISEASE_MM, CohortResult

#: v1.50.2: the published 1.50 figures were removed as the "reference" a site
#: compares against. They are not valid performance estimates (tag
#: audit/1.50-defects), so no internal reference is shown at all.
NO_REFERENCE_LINE = (
    "No validated reference performance exists for JANNUS 1.50; see audit/1.50-defects."
)

#: Placed verbatim at the top of every generated report.md.
REPORT_DEPRECATION_BANNER = (
    "> ⚠️ **DEPRECATED — DO NOT CITE (added 2026-10-05).** The performance figures in this\n"
    "> document are not valid estimates of JANNUS performance. The evaluation cohort\n"
    "> overlapped training data, the shipped pipeline does not reproduce these results,\n"
    "> and mm-based metrics used incorrect voxel spacing. See tag `audit/1.50-defects`\n"
    "> (experiments/lesion_detection/CLOSEOUT.md) for details. Successor: JANNUS 1.60."
)

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
        NO_REFERENCE_LINE,
    ]
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

    add(REPORT_DEPRECATION_BANNER)
    add("")
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
    add(NO_REFERENCE_LINE)
    add("")
    add("| Metric | This site |")
    add("|---|---|")
    add(f"| Voxel Dice | {_ci(result.voxel_dice_all, 4)} |")
    add(f"| Lesion sensitivity (all) | {_ci(result.aggregate_all.get('lesion_wise_sensitivity', {}))} |")
    add(f"| False positives / case | {_ci(result.aggregate_all.get('fp_per_case', {}), 2)} |")
    add("")

    add(f"## RANO-BM measurable disease (longest axis >= {MEASURABLE_DISEASE_MM:.0f} mm)")
    add("")
    add("This is the scope of the proposed Indication for Use.")
    add("")
    add("| Metric | This site | Predicate |")
    add("|---|---|---|")
    add(f"| Lesion sensitivity | {_ci(result.aggregate_measurable.get('lesion_wise_sensitivity', {}))} "
        f"| {REFERENCE_PREDICATE['lesion_sensitivity_measurable']:.2f} |")
    add(f"| Lesion Dice (matched) | {_ci(result.aggregate_measurable.get('lesion_wise_dice_matched', {}))} "
        f"| {REFERENCE_PREDICATE['lesion_dice_measurable']:.2f} |")
    add(f"| HD95 (mm) | {_ci(result.aggregate_measurable.get('hd95_mm', {}), 2)} "
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
