"""Run provenance: what code, what weights, what machine, what settings.

Every artefact JANNUS writes is accompanied by a provenance record. This is the
mechanism that makes multi-site validation auditable: when a site returns
results, the coordinating site can confirm they came from the expected code and
the expected weights, rather than taking it on trust.

The record is deliberately PHI-free. It captures identifiers of *software and
configuration*, never of patients, so it can be emailed, attached to a ticket,
or included in a regulatory submission without further review.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .._version import PIPELINE_REVISION, __version__
from .checksums import hash_loaded_checkpoints

#: Packages whose version can move a metric. Recorded exactly.
_TRACKED_PACKAGES = (
    "torch",
    "numpy",
    "scipy",
    "monai",
    "nibabel",
    "SimpleITK",
    "scikit-image",
    "nnunetv2",
)


def _package_versions() -> dict[str, str]:
    """Installed versions of the packages that can change results."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "<not installed>"
    return out


def _git_revision(repo_root: Path | None = None) -> dict[str, Any]:
    """Best-effort git description of the working tree.

    A site running from a release tarball has no `.git`, which is normal and
    not an error — the field simply reports that.
    """
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    info: dict[str, Any] = {"available": False}
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if commit.returncode != 0:
            return info
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        info = {
            "available": True,
            "commit": commit.stdout.strip(),
            # A dirty tree means the code does not match any published commit.
            # Recorded so a reviewer can discount the run.
            "dirty": bool(dirty.stdout.strip()),
        }
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def _hardware() -> dict[str, Any]:
    """CPU/GPU description. Affects float behaviour, so worth recording."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or "<unknown>",
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["torch_cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpus"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        info["torch_cuda_available"] = False
    return info


@dataclass
class RunProvenance:
    """A complete, PHI-free description of one JANNUS run."""

    command: str
    jannus_version: str = __version__
    pipeline_revision: str = PIPELINE_REVISION
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    finished_at: str | None = None

    git: dict[str, Any] = field(default_factory=_git_revision)
    packages: dict[str, str] = field(default_factory=_package_versions)
    hardware: dict[str, Any] = field(default_factory=_hardware)

    #: Fully-resolved effective configuration, including the operating point.
    config: dict[str, Any] = field(default_factory=dict)
    #: SHA-256 of every checkpoint actually loaded.
    checkpoints: dict[str, str] = field(default_factory=dict)
    #: Output of `configure_determinism`.
    determinism: dict[str, Any] = field(default_factory=dict)

    n_cases_attempted: int = 0
    n_cases_succeeded: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def record_checkpoints(self, paths: dict[str, str | Path]) -> None:
        self.checkpoints = hash_loaded_checkpoints(paths)

    def record_failure(self, case_token: str, reason: str) -> None:
        """Log a per-case failure.

        `case_token` must already be pseudonymised — see
        :func:`jannus.core.logging.pseudonymise`. Provenance records are shared
        artefacts and must never carry a raw identifier.
        """
        self.failures.append({"case": case_token, "reason": reason})

    def finish(self) -> RunProvenance:
        self.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> Path:
        """Serialise to JSON beside the run's outputs."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str) + "\n", encoding="utf-8")
        return path

    def summary_lines(self) -> list[str]:
        """Short human-readable digest for the end of a CLI run."""
        git = self.git.get("commit", "<no git>")
        git_short = git[:12] if git != "<no git>" else git
        dirty = " (DIRTY TREE)" if self.git.get("dirty") else ""
        lines = [
            f"jannus {self.jannus_version}  pipeline {self.pipeline_revision}",
            f"commit {git_short}{dirty}",
            f"cases  {self.n_cases_succeeded}/{self.n_cases_attempted} succeeded",
        ]
        if not self.determinism.get("fully_deterministic", False):
            lines.append("determinism: PARTIAL — results may vary between runs")
        if self.failures:
            lines.append(f"failures: {len(self.failures)} (see provenance.json)")
        return lines
