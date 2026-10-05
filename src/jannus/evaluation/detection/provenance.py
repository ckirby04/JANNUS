"""Provenance records for the detection experiment.

A thin layer over :mod:`jannus.core.provenance` and :mod:`jannus.core.checksums`
that adds what this experiment needs:

- a canonical config hash;
- the split identity (file hash plus the full case list);
- the hash of every input file consumed;
- checkpoint hashes for **every** entry in ``weights.lock.json``.

The last point matters because ``JannusConfig.checkpoint_paths()``, which
``jannus predict`` records, omits the nnU-Net checkpoints, which are configured
by trainer and fold rather than path (PRIOR_DEFECTS.md #7).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..._version import PIPELINE_REVISION, __version__
from ...core.checksums import sha256_file, verify_manifest
from ...core.provenance import _git_revision, _hardware, _package_versions

#: The tag the experiment branch is based on (owner decision Q1).
BASE_TAG = "v1.50.0"
BASE_COMMIT = "d43dc9731667909c4e63459317f7d4f4d5ace2e8"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: Mapping[str, Any]) -> str:
    """SHA-256 of the config's canonical JSON form. Insensitive to key order and YAML layout."""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def checkpoint_hashes(repo_root: str | Path) -> dict[str, dict[str, Any]]:
    """Verify every weights.lock entry and return ``{name: {sha256, role, ok}}``.

    Raises (via ``verify_manifest``) if any checkpoint is missing or differs,
    because a run on unverified weights must not produce results.
    """
    root = Path(repo_root)
    results = verify_manifest(root / "weights.lock.json", root, raise_on_failure=True)
    return {
        r.entry.name: {
            "path": r.entry.path,
            "role": r.entry.role,
            "sha256": r.actual_sha256 or r.entry.sha256,
            "matches_weights_lock": r.ok,
        }
        for r in results
    }


def split_record(name: str, split_file: str | Path, case_ids: Iterable[str]) -> dict[str, Any]:
    ids = sorted(case_ids)
    return {
        "name": name,
        "file": str(split_file),
        "file_sha256": sha256_file(split_file),
        "n_cases": len(ids),
        "case_ids": ids,
    }


def build_provenance(
    *,
    command: str,
    repo_root: str | Path,
    config: Mapping[str, Any],
    config_path: str | Path,
    splits: list[dict[str, Any]],
    inputs: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the provenance.json for one experiment run.

    ``inputs`` maps each consumed file (e.g. a cached probability map) to its
    SHA-256, so a later reader can tell exactly which intermediates fed a result.
    """
    git = _git_revision(Path(repo_root))
    return {
        "command": command,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "jannus_version": __version__,
        "pipeline_revision": PIPELINE_REVISION,
        "git": {**git, "base_tag": BASE_TAG, "base_commit": BASE_COMMIT},
        "config_path": str(config_path),
        "config_sha256": config_hash(config),
        "config": dict(config),
        "checkpoints": checkpoint_hashes(repo_root),
        "splits": splits,
        "inputs_sha256": dict(sorted((inputs or {}).items())),
        "packages": _package_versions(),
        "hardware": _hardware(),
        **(dict(extra) if extra else {}),
    }


def write_json(path: str | Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
    return path
