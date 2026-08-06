"""Checkpoint integrity: hashing, manifests, and verification.

Why this module exists
----------------------
In a multi-site validation study the coordinating site has to be able to prove
that every participating site ran *the same weights*. Before v1.50 nothing in
the codebase hashed a `.pth` file, so a site could quietly run a stale or
substituted checkpoint and report numbers that could never be traced back.

The contract:

* ``weights.lock.json`` at the repo root lists every checkpoint with its
  SHA-256, byte size and role.
* ``jannus fetch-weights`` downloads and verifies against it.
* ``verify_manifest()`` runs at pipeline construction, so a mismatch fails the
  run instead of contaminating a study.

Hashes are computed over raw file bytes — not over the deserialised tensors —
so verification never needs to unpickle an untrusted file.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from .errors import WeightsError

#: Read size for streaming hashes. Checkpoints run to hundreds of MB; never
#: read one fully into memory.
_CHUNK = 1024 * 1024

MANIFEST_FILENAME = "weights.lock.json"
MANIFEST_SCHEMA_VERSION = 1


def sha256_file(path: str | Path, *, chunk_size: int = _CHUNK) -> str:
    """Return the lowercase hex SHA-256 of a file, streamed."""
    path = Path(path)
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise WeightsError(f"Cannot read checkpoint {path}: {exc}") from exc
    return digest.hexdigest()


def short_hash(full: str, length: int = 12) -> str:
    """Abbreviated hash for logs and report headers."""
    return full[:length]


@dataclass(frozen=True)
class WeightEntry:
    """One checkpoint in the manifest."""

    name: str  #: logical name, matches configs/models.yaml `name`
    path: str  #: repo-relative path, e.g. "model/patch_8.pth"
    sha256: str
    bytes: int
    role: str = "base"  #: "base" | "stacker" | "pretrained"
    url: str | None = None  #: where fetch-weights retrieves it
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {k: v for k, v in asdict(self).items() if v not in (None, "")}


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of checking one entry against disk."""

    entry: WeightEntry
    present: bool
    size_ok: bool
    hash_ok: bool
    actual_sha256: str | None = None
    actual_bytes: int | None = None

    @property
    def ok(self) -> bool:
        return self.present and self.size_ok and self.hash_ok

    def describe(self) -> str:
        if self.ok:
            return f"OK       {self.entry.name}  ({short_hash(self.entry.sha256)})"
        if not self.present:
            return f"MISSING  {self.entry.name}  expected at {self.entry.path}"
        if not self.size_ok:
            return (
                f"SIZE     {self.entry.name}  expected {self.entry.bytes} bytes, "
                f"found {self.actual_bytes}"
            )
        return (
            f"HASH     {self.entry.name}  expected {short_hash(self.entry.sha256)}, "
            f"found {short_hash(self.actual_sha256 or '')}"
        )


def load_manifest(manifest_path: str | Path) -> list[WeightEntry]:
    """Parse ``weights.lock.json``."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise WeightsError(
            f"Weight manifest not found at {manifest_path}.",
            remedy="Expected `weights.lock.json` at the repository root.",
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WeightsError(f"Weight manifest {manifest_path} is not valid JSON: {exc}") from exc

    version = payload.get("schema_version")
    if version != MANIFEST_SCHEMA_VERSION:
        raise WeightsError(
            f"Weight manifest schema version {version!r} is not supported "
            f"(this build understands {MANIFEST_SCHEMA_VERSION})."
        )

    entries: list[WeightEntry] = []
    for raw in payload.get("weights", []):
        missing = {"name", "path", "sha256", "bytes"} - set(raw)
        if missing:
            raise WeightsError(
                f"Weight manifest entry {raw.get('name', '<unnamed>')!r} is missing "
                f"required field(s): {', '.join(sorted(missing))}"
            )
        entries.append(
            WeightEntry(
                name=raw["name"],
                path=raw["path"],
                sha256=raw["sha256"].lower(),
                bytes=int(raw["bytes"]),
                role=raw.get("role", "base"),
                url=raw.get("url"),
                notes=raw.get("notes", ""),
            )
        )
    return entries


def write_manifest(
    entries: Iterable[WeightEntry],
    manifest_path: str | Path,
    *,
    pipeline_revision: str,
) -> Path:
    """Serialise a manifest. Used by the maintainer tooling, not by sites."""
    manifest_path = Path(manifest_path)
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "pipeline_revision": pipeline_revision,
        "weights": [entry.to_dict() for entry in entries],
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def verify_entry(entry: WeightEntry, root: str | Path) -> VerificationResult:
    """Check one manifest entry against the filesystem.

    Size is checked before the hash: it is effectively free and catches the
    common failure (a truncated or LFS-pointer-only download) without spending
    time hashing hundreds of megabytes.
    """
    target = Path(root) / entry.path
    if not target.is_file():
        return VerificationResult(entry=entry, present=False, size_ok=False, hash_ok=False)

    actual_bytes = target.stat().st_size
    if actual_bytes != entry.bytes:
        return VerificationResult(
            entry=entry,
            present=True,
            size_ok=False,
            hash_ok=False,
            actual_bytes=actual_bytes,
        )

    actual = sha256_file(target)
    return VerificationResult(
        entry=entry,
        present=True,
        size_ok=True,
        hash_ok=actual == entry.sha256,
        actual_sha256=actual,
        actual_bytes=actual_bytes,
    )


def verify_manifest(
    manifest_path: str | Path,
    root: str | Path,
    *,
    roles: Iterable[str] | None = None,
    raise_on_failure: bool = True,
) -> list[VerificationResult]:
    """Verify every checkpoint in the manifest.

    Args:
        manifest_path: path to ``weights.lock.json``.
        root: directory the manifest's relative paths resolve against.
        roles: restrict to these roles (e.g. ``{"base", "stacker"}`` to skip
            optional pretrained encoders that are not needed for inference).
        raise_on_failure: raise :class:`WeightsError` if anything fails.

    Returns:
        One result per entry, in manifest order, so a caller can render a full
        table rather than stopping at the first failure.
    """
    entries = load_manifest(manifest_path)
    if roles is not None:
        wanted = set(roles)
        entries = [e for e in entries if e.role in wanted]

    results = [verify_entry(entry, root) for entry in entries]

    if raise_on_failure:
        failures = [r for r in results if not r.ok]
        if failures:
            detail = "\n  ".join(r.describe() for r in failures)
            raise WeightsError(
                f"{len(failures)} of {len(results)} checkpoint(s) failed verification:\n  {detail}"
            )
    return results


def hash_loaded_checkpoints(paths: dict[str, str | Path]) -> dict[str, str]:
    """Hash the checkpoints actually loaded, for the provenance manifest.

    Distinct from :func:`verify_manifest`: that answers "are the expected
    weights on disk", this answers "what did this specific run actually load".
    Missing files are recorded as ``"<missing>"`` instead of raising, because a
    provenance record of a partly-failed run is still worth writing.
    """
    out: dict[str, str] = {}
    for name, path in paths.items():
        p = Path(path)
        out[name] = sha256_file(p) if p.is_file() else "<missing>"
    return out
