"""Metrics, confidence intervals, stratification and reporting.

`jannus.evaluation.metrics` is the module that produces the numbers a
validation site reports back, including the FDA-facing lesion-wise and RANO-BM
measurable-disease figures. It carried zero test coverage through v1.40; see
`tests/unit/test_metrics.py`.
"""

from __future__ import annotations

from .harness import CaseMetrics, CohortResult, evaluate_cohort
from .report import render_markdown_report, render_text_summary

__all__ = [
    "CaseMetrics",
    "CohortResult",
    "evaluate_cohort",
    "render_markdown_report",
    "render_text_summary",
]
