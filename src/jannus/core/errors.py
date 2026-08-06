"""Typed exception hierarchy.

Every failure a site can plausibly hit gets its own class, so the CLI can map
it to a stable exit code and an actionable message instead of dumping a
traceback at a research coordinator who has no way to interpret it.

Exit codes are part of the published interface — sites wrap JANNUS in their own
batch schedulers and branch on them. Do not renumber without a major bump.
"""

from __future__ import annotations


class JannusError(Exception):
    """Base class for every error JANNUS raises deliberately."""

    exit_code: int = 1
    #: Shown to the user beneath the message. Override with something the site
    #: can actually act on.
    remedy: str | None = None

    def __init__(self, message: str, *, remedy: str | None = None):
        super().__init__(message)
        if remedy is not None:
            self.remedy = remedy


class ConfigError(JannusError):
    """Malformed, missing, or internally inconsistent configuration."""

    exit_code = 2
    remedy = "Check the --config file against configs/models.yaml in the repo."


class DataLayoutError(JannusError):
    """The input dataset does not match the expected on-disk layout."""

    exit_code = 3
    remedy = "Run `jannus validate-data` for a per-case breakdown of what is missing."


class DataIntegrityError(JannusError):
    """Files are present but unusable — unreadable, empty, or inconsistent."""

    exit_code = 4
    remedy = "Run `jannus validate-data --strict` to see which cases fail and why."


class WeightsError(JannusError):
    """A model checkpoint is missing, corrupt, or fails its checksum."""

    exit_code = 5
    remedy = "Run `jannus fetch-weights` to download and verify all checkpoints."


class PipelineError(JannusError):
    """Inference failed in a way that invalidates the result."""

    exit_code = 6


class EvaluationError(JannusError):
    """Metric computation could not proceed."""

    exit_code = 7


class UnverifiedPipelineError(WeightsError):
    """The pipeline would run with stub or partially-loaded weights.

    This is fatal by design. A stubbed adapter is random-initialised: it
    produces a well-formed probability map that is pure noise. In a validation
    study that silently becomes a published number, so JANNUS refuses to
    proceed rather than letting it through.
    """

    remedy = (
        "Run `jannus doctor` to see which checkpoints are missing, then "
        "`jannus fetch-weights`. Use --allow-stub only for plumbing smoke tests "
        "whose numbers you will discard."
    )
