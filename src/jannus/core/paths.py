"""Repository and resource path resolution.

Why this exists
---------------
v1.40 had eight separate modules that located the repository root by counting
``Path(__file__).parent`` hops. Every one of them encoded the module's depth in
the tree, so moving a file — as the v1.50 package restructure did — silently
broke them all, each in a different way and none at import time. Two scripts
were *already* wrong in v1.40 for exactly this reason, resolving to
``<repo>/scripts`` instead of ``<repo>``.

Counting hops is the wrong mechanism. Here the root is found by searching
upward for marker files that only ever exist at the root, so the answer stays
correct however deep a module sits, and an explicit override is available for
installed or containerised layouts where the source tree is not present.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

#: Overrides discovery entirely. Set this when JANNUS is installed as a wheel
#: and configs/model weights live somewhere unrelated to the source.
HOME_ENV = "JANNUS_HOME"

#: Files that exist only at a repository root.
_ROOT_MARKERS = ("pyproject.toml", "weights.lock.json")


def package_root() -> Path:
    """Directory of the installed `jannus` package (``.../src/jannus``)."""
    return Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Best-effort repository root.

    Resolution order:
      1. ``$JANNUS_HOME`` if set and it exists.
      2. The nearest ancestor of this file containing a root marker.
      3. Two levels above the package (``src/jannus`` -> repo), which is the
         correct answer for an editable install even with no markers present.

    Never raises: a caller asking for a resource gets a definite path, and the
    subsequent "file not found" is far more actionable than a failure here.
    """
    override = os.environ.get(HOME_ENV)
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_dir():
            return candidate.resolve()

    for parent in package_root().parents:
        if any((parent / marker).is_file() for marker in _ROOT_MARKERS):
            return parent

    return package_root().parent.parent


def resource(*parts: str) -> Path:
    """Path to a repo-relative resource, e.g. ``resource("configs", "rag.yaml")``."""
    return repo_root().joinpath(*parts)


def config_path(name: str = "models.yaml", explicit: str | Path | None = None) -> Path:
    """Locate a file under ``configs/``.

    Resolution order:
      1. ``explicit``, if given.
      2. ``<repo_root>/configs/<name>`` — what a git clone uses, and what a
         site edits when tuning for their data.
      3. ``<package>/configs/<name>`` — the copy shipped inside the wheel, for
         installations with no source tree alongside.

    The repo copy wins so that a site's edits take effect. The packaged copy is
    the fallback that makes `pip install jannus` usable at all: v1.40 declared
    no package data, so an installed wheel had no models.yaml and could not
    build a pipeline.

    Args:
        name: filename inside ``configs/``.
        explicit: caller-supplied path; returned unchanged when given, so this
            can be dropped into an existing ``config_path=None`` parameter.
    """
    if explicit is not None:
        return Path(explicit)

    repo_copy = resource("configs", name)
    if repo_copy.is_file():
        return repo_copy

    packaged = package_root() / "configs" / name
    return packaged if packaged.is_file() else repo_copy


def model_dir() -> Path:
    """Directory holding model checkpoints."""
    return resource("model")


def data_root(explicit: str | Path | None = None) -> Path:
    """Site dataset root.

    v1.40 shipped ``data/`` as a symlink to an absolute path on one developer's
    machine, which at any other site was either absent or dangling. There is no
    default any more: a site states where its imaging lives, via the argument
    or ``$JANNUS_DATA_ROOT``.
    """
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get("JANNUS_DATA_ROOT")
    if env:
        return Path(env).expanduser()
    return resource("data")
