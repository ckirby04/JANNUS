"""Dataset discovery, QC and volume loading."""

from __future__ import annotations

from .layout import (
    DatasetIndex,
    ResolvedCase,
    UnresolvedCase,
    format_alias_table,
    resolve_case,
    scan_dataset,
)
from .loading import LoadedVolume, load_case, load_mask, save_mask, zscore
from .validation import (
    CaseReport,
    Finding,
    Severity,
    ValidationReport,
    validate_dataset,
)

__all__ = [
    "CaseReport",
    "DatasetIndex",
    "Finding",
    "LoadedVolume",
    "ResolvedCase",
    "Severity",
    "UnresolvedCase",
    "ValidationReport",
    "format_alias_table",
    "load_case",
    "load_mask",
    "resolve_case",
    "save_mask",
    "scan_dataset",
    "validate_dataset",
    "zscore",
]
