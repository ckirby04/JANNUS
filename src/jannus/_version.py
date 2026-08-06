"""Single source of truth for the JANNUS version.

`pyproject.toml` reads this dynamically, so the version is declared exactly
once. Every artefact JANNUS writes (segmentation NIfTI sidecars, evaluation
reports, provenance manifests) stamps this string, which is what lets a
coordinating site tie a result file back to the code that produced it.
"""

from __future__ import annotations

__version__ = "1.50.0"

# Bumped whenever a change could move a metric. Sites comparing results across
# runs must compare this, not `__version__` alone: a packaging-only release
# leaves it untouched, so identical values mean the numbers are comparable.
PIPELINE_REVISION = "7base-stackerv2-r1"
