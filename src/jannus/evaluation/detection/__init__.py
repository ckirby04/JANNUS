"""Lesion-level detection (CADe-style) evaluation.

Experimental and self-contained: this subpackage imports from the rest of
``jannus`` but nothing outside it imports from here, and it changes no
existing metric, report or provenance format. Methodology and rationale live
in ``experiments/lesion_detection/DECISIONS.md``.
"""

from __future__ import annotations

from .geometry import (
    assign_size_bin,
    equivalent_spherical_diameter,
    resampled_spacing,
    validate_size_bins,
)
from .matching import (
    ComponentTable,
    MatchResult,
    extract_components,
    match,
    match_by_centroid,
    match_by_overlap,
)

__all__ = [
    "ComponentTable",
    "MatchResult",
    "assign_size_bin",
    "equivalent_spherical_diameter",
    "extract_components",
    "match",
    "match_by_centroid",
    "match_by_overlap",
    "resampled_spacing",
    "validate_size_bins",
]
