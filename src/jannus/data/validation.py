"""Pre-flight dataset QC for external validation sites.

This is the module that makes cross-site validation practical. Before a site
spends GPU-days on inference — and long before anyone tries to interpret the
resulting Dice — `jannus validate-data` answers:

* Are all four sequences present and readable for every case?
* Are they co-registered onto a common grid?
* Is the voxel spacing and field of view inside the range the model saw in
  training, or is this extrapolation?
* Does the intensity distribution look like brain MRI, or like something that
  has already been normalised, clipped, or skull-stripped differently?
* If ground truth is supplied, is it plausibly a brain-metastasis annotation?

Findings are graded. ``ERROR`` blocks inference. ``WARNING`` means results
remain interpretable but the deviation must be reported alongside them —
JANNUS records every warning in the provenance manifest so it travels with the
numbers rather than being lost.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..core.config import DataConfig
from ..core.logging import get_logger
from .layout import DatasetIndex, ResolvedCase
from .loading import (
    PLAUSIBLE_SHAPE_RANGE,
    PLAUSIBLE_SPACING_RANGE,
    load_case,
    load_mask,
)

logger = get_logger("data.validation")


class Severity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class Finding:
    """One QC observation about one case (or the cohort, when `case` is None)."""

    severity: Severity
    code: str
    message: str
    case: str | None = None  #: pseudonymised token, never a raw identifier

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


@dataclass
class CaseReport:
    """Per-case QC result."""

    case_token: str
    shape: Sequence[int] | None = None
    spacing: Sequence[float] | None = None
    matched_aliases: dict[str, str] = field(default_factory=dict)
    has_ground_truth: bool = False
    gt_lesion_voxels: int | None = None
    findings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(f.severity is Severity.ERROR for f in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case_token,
            "shape": list(self.shape) if self.shape else None,
            "spacing": [round(s, 4) for s in self.spacing] if self.spacing else None,
            "matched_aliases": self.matched_aliases,
            "has_ground_truth": self.has_ground_truth,
            "gt_lesion_voxels": self.gt_lesion_voxels,
            "ok": self.ok,
            "findings": [f.to_dict() for f in self.findings],
        }


@dataclass
class ValidationReport:
    """Whole-cohort QC result. This is what a site returns to the coordinator."""

    dataset_root: str
    n_resolved: int
    n_unresolved: int
    n_excluded: int
    cases: list[CaseReport] = field(default_factory=list)
    cohort_findings: list[Finding] = field(default_factory=list)

    @property
    def n_ok(self) -> int:
        return sum(1 for c in self.cases if c.ok)

    @property
    def errors(self) -> list[Finding]:
        out = [f for f in self.cohort_findings if f.severity is Severity.ERROR]
        for case in self.cases:
            out.extend(f for f in case.findings if f.severity is Severity.ERROR)
        return out

    @property
    def warnings(self) -> list[Finding]:
        out = [f for f in self.cohort_findings if f.severity is Severity.WARNING]
        for case in self.cases:
            out.extend(f for f in case.findings if f.severity is Severity.WARNING)
        return out

    @property
    def passed(self) -> bool:
        return not self.errors and self.n_ok > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": self.dataset_root,
            "summary": {
                "resolved": self.n_resolved,
                "unresolved": self.n_unresolved,
                "excluded": self.n_excluded,
                "passed_qc": self.n_ok,
                "errors": len(self.errors),
                "warnings": len(self.warnings),
                "passed": self.passed,
            },
            "cohort_findings": [f.to_dict() for f in self.cohort_findings],
            "cases": [c.to_dict() for c in self.cases],
        }

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    def render(self, *, max_listed: int = 10) -> str:
        """Human-readable summary for the terminal."""
        lines = [
            f"Dataset: {self.dataset_root}",
            f"  cases resolved   : {self.n_resolved}",
            f"  cases passing QC : {self.n_ok}",
        ]
        if self.n_unresolved:
            lines.append(f"  cases unresolved : {self.n_unresolved}  (missing sequences)")
        if self.n_excluded:
            lines.append(f"  dirs excluded    : {self.n_excluded}  (case_pattern)")

        for label, items in (("ERRORS", self.errors), ("WARNINGS", self.warnings)):
            if not items:
                continue
            lines.append("")
            lines.append(f"{label} ({len(items)}):")
            for finding in items[:max_listed]:
                where = f"[{finding.case}] " if finding.case else ""
                lines.append(f"  {where}{finding.code}: {finding.message}")
            if len(items) > max_listed:
                lines.append(f"  ... and {len(items) - max_listed} more (see the JSON report)")

        lines.append("")
        lines.append("RESULT: PASS — dataset is ready for inference." if self.passed
                     else "RESULT: FAIL — resolve the errors above before running inference.")
        return "\n".join(lines)


def _check_geometry(case: ResolvedCase, report: CaseReport, volume) -> None:
    shape = volume.shape
    spacing = volume.voxel_spacing
    report.shape = shape
    report.spacing = spacing

    lo, hi = PLAUSIBLE_SHAPE_RANGE
    if any(d < lo for d in shape):
        report.findings.append(Finding(
            Severity.ERROR, "SHAPE_TOO_SMALL",
            f"volume {shape} has a dimension below {lo} voxels; this does not look "
            f"like a 3D brain MRI",
            case.token,
        ))
    elif any(d > hi for d in shape):
        report.findings.append(Finding(
            Severity.WARNING, "SHAPE_LARGE",
            f"volume {shape} exceeds {hi} voxels on some axis; inference will be slow "
            f"and memory-hungry",
            case.token,
        ))

    slo, shi = PLAUSIBLE_SPACING_RANGE
    if any(s <= 0 for s in spacing):
        report.findings.append(Finding(
            Severity.ERROR, "SPACING_INVALID",
            f"voxel spacing {spacing} contains a non-positive value; the NIfTI header "
            f"is malformed and physical-distance metrics (HD95, MSD) would be meaningless",
            case.token,
        ))
    elif any(s < slo or s > shi for s in spacing):
        report.findings.append(Finding(
            Severity.WARNING, "SPACING_OUT_OF_RANGE",
            f"voxel spacing {tuple(round(s, 2) for s in spacing)} mm falls outside the "
            f"{slo}-{shi} mm range represented in training; results are extrapolation",
            case.token,
        ))

    # Strongly anisotropic acquisitions degrade 3D context and inflate HD95.
    if min(spacing) > 0 and max(spacing) / min(spacing) > 4.0:
        report.findings.append(Finding(
            Severity.WARNING, "HIGHLY_ANISOTROPIC",
            f"voxel spacing {tuple(round(s, 2) for s in spacing)} mm is strongly "
            f"anisotropic (ratio {max(spacing) / min(spacing):.1f}); small-lesion "
            f"sensitivity is expected to drop",
            case.token,
        ))


def _check_intensities(case: ResolvedCase, report: CaseReport, volume) -> None:
    for channel, stats in volume.channel_stats.items():
        if stats["std"] <= 0:
            report.findings.append(Finding(
                Severity.ERROR, "CONSTANT_CHANNEL",
                f"channel {channel!r} is constant (std=0) — the acquisition is empty "
                f"or the file is a placeholder",
                case.token,
            ))
            continue
        if stats["min"] == stats["max"]:
            report.findings.append(Finding(
                Severity.ERROR, "DEGENERATE_CHANNEL",
                f"channel {channel!r} has min == max ({stats['min']})",
                case.token,
            ))
        # Already-z-scored input is the most common silent preprocessing clash:
        # JANNUS z-scores internally, so double-normalisation is harmless, but
        # it usually signals the site also applied clipping we cannot see.
        if abs(stats["mean"]) < 0.01 and abs(stats["std"] - 1.0) < 0.05:
            report.findings.append(Finding(
                Severity.INFO, "PRE_NORMALISED",
                f"channel {channel!r} appears already z-scored (mean~0, std~1); JANNUS "
                f"normalises internally, so confirm no intensity clipping was applied",
                case.token,
            ))
        if stats["min"] < 0 and stats["max"] > 0 and abs(stats["min"]) > 3 * stats["std"]:
            report.findings.append(Finding(
                Severity.WARNING, "UNEXPECTED_NEGATIVES",
                f"channel {channel!r} has large negative values (min={stats['min']:.1f}); "
                f"MRI magnitude images are non-negative before normalisation",
                case.token,
            ))


def _check_ground_truth(case: ResolvedCase, report: CaseReport, volume) -> None:
    if case.ground_truth is None:
        return
    report.has_ground_truth = True
    try:
        mask = load_mask(case.ground_truth)
    except Exception as exc:
        report.findings.append(Finding(
            Severity.ERROR, "GT_UNREADABLE",
            f"ground truth {case.ground_truth.name} could not be read: {exc}",
            case.token,
        ))
        return

    if mask.shape != tuple(volume.shape):
        report.findings.append(Finding(
            Severity.ERROR, "GT_SHAPE_MISMATCH",
            f"ground truth shape {mask.shape} does not match the imaging "
            f"{tuple(volume.shape)}; metrics cannot be computed",
            case.token,
        ))
        return

    n_positive = int(mask.sum())
    report.gt_lesion_voxels = n_positive
    if n_positive == 0:
        report.findings.append(Finding(
            Severity.WARNING, "GT_EMPTY",
            "ground truth contains no foreground voxels; this case contributes only "
            "false positives to cohort metrics",
            case.token,
        ))
        return

    fraction = n_positive / mask.size
    # Brain metastases occupy a very small fraction of the volume. Anything
    # near a percent means whole-brain, oedema, or an inverted mask.
    if fraction > 0.05:
        report.findings.append(Finding(
            Severity.WARNING, "GT_IMPLAUSIBLY_LARGE",
            f"ground truth covers {fraction:.1%} of the volume; brain-metastasis "
            f"annotations are typically well under 1%. Check the mask is not inverted "
            f"or labelling a different structure",
            case.token,
        ))


def validate_dataset(
    index: DatasetIndex,
    data_config: DataConfig | None = None,
    *,
    strict: bool = False,
    check_intensities: bool = True,
) -> ValidationReport:
    """Run full QC over a scanned dataset.

    Args:
        index: output of :func:`jannus.data.layout.scan_dataset`.
        data_config: sequence naming rules.
        strict: promote every WARNING to an ERROR. Appropriate for a formal
            validation cohort where any protocol deviation should block.
        check_intensities: load pixel data for distribution checks. Disable for
            a fast structural-only pass over a very large cohort.
    """
    cfg = data_config or DataConfig()
    report = ValidationReport(
        dataset_root=str(index.root),
        n_resolved=len(index.cases),
        n_unresolved=len(index.unresolved),
        n_excluded=len(index.excluded),
    )

    for unresolved in index.unresolved:
        report.cohort_findings.append(Finding(
            Severity.ERROR, "MISSING_SEQUENCES",
            f"case is missing required channel(s): {', '.join(unresolved.missing)}",
            unresolved.token,
        ))

    if index.excluded:
        report.cohort_findings.append(Finding(
            Severity.WARNING, "EXCLUDED_BY_PATTERN",
            f"{len(index.excluded)} director(ies) were skipped by "
            f"data.case_pattern={cfg.case_pattern!r}; clear that setting to include them",
        ))

    shapes: list[Sequence[int]] = []
    for case in index.cases:
        case_report = CaseReport(
            case_token=case.token, matched_aliases=dict(case.matched_aliases)
        )
        try:
            volume = load_case(case, cfg.sequences, normalise=False)
        except Exception as exc:
            case_report.findings.append(Finding(
                Severity.ERROR, "UNREADABLE", str(exc), case.token
            ))
            report.cases.append(case_report)
            continue

        _check_geometry(case, case_report, volume)
        if check_intensities:
            _check_intensities(case, case_report, volume)
        _check_ground_truth(case, case_report, volume)

        shapes.append(volume.shape)
        report.cases.append(case_report)

    # Heterogeneous geometry across a cohort is legal but worth surfacing: it
    # usually means more than one acquisition protocol is present, which shows
    # up as bimodal per-case Dice later.
    if len({tuple(s) for s in shapes}) > 1:
        report.cohort_findings.append(Finding(
            Severity.INFO, "HETEROGENEOUS_GEOMETRY",
            f"cohort contains {len({tuple(s) for s in shapes})} distinct volume shapes; "
            f"consider stratifying reported metrics by acquisition protocol",
        ))

    n_with_gt = sum(1 for c in report.cases if c.has_ground_truth)
    if 0 < n_with_gt < len(report.cases):
        report.cohort_findings.append(Finding(
            Severity.WARNING, "PARTIAL_GROUND_TRUTH",
            f"only {n_with_gt}/{len(report.cases)} cases have ground truth; "
            f"`jannus evaluate` will score just those",
        ))

    if strict:
        for finding in report.cohort_findings:
            if finding.severity is Severity.WARNING:
                finding.severity = Severity.ERROR
        for case_report in report.cases:
            for finding in case_report.findings:
                if finding.severity is Severity.WARNING:
                    finding.severity = Severity.ERROR

    return report
