"""v1.50.2: no published JANNUS 1.50 figure may appear in generated reports or the demo UI.

The published figures are not valid performance estimates (tag audit/1.50-defects).
`jannus evaluate` used to print them as the "internal reference", and the demo
displayed them as badges. These tests pin their removal and the deprecation banner.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jannus.evaluation.harness import evaluate_cohort
from jannus.evaluation.report import (
    NO_REFERENCE_LINE,
    REPORT_DEPRECATION_BANNER,
    render_markdown_report,
    render_text_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Published 1.50 figures, in every form they were displayed in (decimal and percent).
PUBLISHED_FIGURES = (
    "0.7858", "0.786", "78.6%", "0.810", "81.0%", "1.68", "0.791", "0.943", "94.3%",
    "0.835", "83.5%", "1.61", "17.26", "3.89", "0.061", "0.867", "1.84%", "18.36", "3.53",
)


@pytest.fixture(scope="module")
def cohort():
    """Perfect predictions, so the site's own numbers (1.0000, 0.00, ...) cannot collide
    with any published figure."""
    pairs = []
    for i in range(3):
        gt = np.zeros((24, 24, 24), dtype=np.uint8)
        gt[4 + i:12 + i, 4:12, 4:12] = 1
        gt[16:20, 16:20, 16:20] = 1
        pairs.append((f"tok{i}", gt.copy(), gt, (1.0, 1.0, 1.0)))
    return evaluate_cohort(pairs, n_bootstrap=50)


def test_markdown_report_has_banner_first_and_no_reference_figures(cohort):
    md = render_markdown_report(cohort, site_name="Test Site", provenance={})
    assert md.startswith(REPORT_DEPRECATION_BANNER)
    assert "DEPRECATED — DO NOT CITE" in md and "audit/1.50-defects" in md
    assert NO_REFERENCE_LINE in md
    assert "Internal reference" not in md
    leaked = [f for f in PUBLISHED_FIGURES if f in md]
    assert not leaked, f"published 1.50 figures still in report.md: {leaked}"


def test_text_summary_has_no_reference_figures(cohort):
    text = render_text_summary(cohort)
    assert NO_REFERENCE_LINE in text
    assert "Reference (internal" not in text
    assert "differs by" not in text
    leaked = [f for f in PUBLISHED_FIGURES if f in text]
    assert not leaked, f"published 1.50 figures still in the terminal summary: {leaked}"
    NO_REFERENCE_LINE.encode("ascii")  # new stdout text must survive a non-UTF-8 console


def test_report_module_holds_no_published_figures():
    src = (REPO_ROOT / "src/jannus/evaluation/report.py").read_text(encoding="utf-8")
    assert "REFERENCE_INTERNAL" not in src
    leaked = [f for f in ("0.7858", "0.810", "1.68", "0.943", "0.835", "17.26") if f in src]
    assert not leaked, leaked


def test_demo_source_displays_no_published_figures():
    src = (REPO_ROOT / "demo/app.py").read_text(encoding="utf-8")
    leaked = [f for f in PUBLISHED_FIGURES if f in src]
    assert not leaked, f"published 1.50 figures still in demo/app.py: {leaked}"
    assert "Validation Dice" not in src
    assert "DEPRECATION_NOTICE" in src and "audit/1.50-defects" in src
    # The notice replaces the figures in the header badge, the footer and the report footer.
    assert src.count("DEPRECATION_NOTICE_SHORT") >= 3
    assert '""" + DEPRECATION_NOTICE + """' in src
