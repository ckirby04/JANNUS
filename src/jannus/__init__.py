"""JANNUS — brain metastasis segmentation for multi-site validation.

Public entry points
-------------------
    jannus.load_pipeline(config, ...)   build the production inference pipeline
    jannus.__version__                  package version
    jannus.PIPELINE_REVISION            metric-affecting revision identifier

Everything else lives in a subpackage:

    jannus.core         config, logging, provenance, determinism, errors
    jannus.data         dataset layout discovery, validation, volume loading
    jannus.segmentation base models, adapters, the stacker, postprocessing
    jannus.evaluation   metrics, bootstrap CIs, stratification, reporting
    jannus.api          FastAPI service
    jannus.rag          literature retrieval

Heavy third-party imports (torch, monai, nnunetv2) are deliberately *not*
pulled in at package-import time, so `import jannus` and `jannus --help` stay
fast and work in environments where only a subset of extras is installed.
"""

from __future__ import annotations

from ._version import PIPELINE_REVISION, __version__

__all__ = ["PIPELINE_REVISION", "__version__", "load_pipeline"]


def load_pipeline(config_path, device=None, *, stub: bool = False, verify: bool = True):
    """Build the production 7-base + StackingClassifierV2 pipeline.

    Thin re-export of :meth:`jannus.segmentation.pipeline.BrainMetPipeline.from_config`
    kept at the top level so external sites have one obvious import.

    Note the default: ``verify=True``. The underlying classmethod defaults to
    ``False`` for backwards compatibility with existing research callers, but
    for validation work a silent fall-through to random-weight stubs would
    produce plausible-looking, meaningless numbers. Opt out explicitly if you
    genuinely want an unverified pipeline.
    """
    from .segmentation.pipeline import BrainMetPipeline

    return BrainMetPipeline.from_config(
        config_path, device=device, stub=stub, verify=verify
    )
