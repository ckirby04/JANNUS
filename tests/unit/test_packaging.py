"""Packaging invariants.

These guard the class of defect that made v1.40 unusable when pip-installed:
resources the code needs at runtime that were never actually shipped, and
duplicated files that silently drift apart.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "src" / "jannus"

#: Config files that must exist in both the repo root and inside the package.
#: The repo copy is what a git clone uses; the package copy is what a wheel
#: install uses. They must be byte-identical.
SHIPPED_CONFIGS = ("models.yaml", "rag.yaml")


@pytest.mark.parametrize("name", SHIPPED_CONFIGS)
def test_config_is_shipped_inside_the_package(name):
    """`configs/` must be package data.

    In v1.40 it was not, so `pip install` produced an installation with no
    models.yaml, and the pipeline could not be constructed at all from a wheel.
    CI never caught it because it never installed the built wheel.
    """
    assert (PACKAGE_ROOT / "configs" / name).is_file(), (
        f"{name} is missing from src/jannus/configs/ and will not ship in the wheel"
    )


@pytest.mark.parametrize("name", SHIPPED_CONFIGS)
def test_repo_and_package_configs_are_identical(name):
    """The two copies must not drift.

    A site editing `configs/models.yaml` in a clone and a site running an
    installed wheel must get the same operating point.
    """
    repo_copy = (REPO_ROOT / "configs" / name).read_bytes()
    package_copy = (PACKAGE_ROOT / "configs" / name).read_bytes()
    assert repo_copy == package_copy, (
        f"configs/{name} and src/jannus/configs/{name} have diverged. "
        f"Copy the repo-root version into the package: "
        f"cp configs/{name} src/jannus/configs/{name}"
    )


def test_py_typed_marker_is_present():
    """PEP 561 marker, so downstream type checkers see our annotations."""
    assert (PACKAGE_ROOT / "py.typed").is_file()


def test_dead_configs_are_not_in_the_live_config_dir():
    """augmentation.yaml / training_strategy.yaml had zero consumers.

    They were retired to docs/research/dead_configs/. Shipping configuration
    that looks live but is inert misleads an external site into tuning values
    that do nothing.
    """
    live = {p.name for p in (REPO_ROOT / "configs").glob("*.yaml")}
    assert "augmentation.yaml" not in live
    assert "training_strategy.yaml" not in live


def test_version_is_single_sourced():
    """The version must come from jannus._version, not be duplicated.

    v1.40 hardcoded "1.23.0" in five places, including the SoftwareVersions tag
    written into every emitted DICOM-SEG and the footer of clinician-facing
    PDFs — a traceability defect in a regulated context.
    """
    from jannus import __version__

    offenders = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        if path.name == "_version.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        # A version-shaped literal assigned to a version-ish field.
        for marker in ('version: str = "1.', 'version="1.', 'version = "1.'):
            if marker in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {marker!r}")

    assert not offenders, (
        "hardcoded version literal(s) found; import from jannus._version instead:\n  "
        + "\n  ".join(offenders)
    )
    assert __version__.count(".") == 2


def test_weights_manifest_covers_every_configured_checkpoint():
    """Every checkpoint the config references must be pinned by hash.

    Otherwise a site could load an unverified model and report numbers that
    cannot be traced back to a known set of weights.
    """
    from jannus.core.checksums import load_manifest
    from jannus.core.config import load_config

    config = load_config(REPO_ROOT / "configs" / "models.yaml")
    manifest_names = {e.name for e in load_manifest(REPO_ROOT / "weights.lock.json")}

    for name in config.model_names:
        assert name in manifest_names, f"base model {name!r} is not pinned in weights.lock.json"
    assert "stacker" in manifest_names
