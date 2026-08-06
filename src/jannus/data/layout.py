"""Dataset discovery and sequence resolution.

The problem this solves
-----------------------
v1.40 resolved the four input channels four different ways:

===========================  ========================================
`scripts/inference/...`      defaulted to `bravo.nii.gz` (GE-specific)
`src/api/server.py`          hardcoded `t2.nii.gz`, rejected `bravo`
`demo/app.py`                probed `t2`, fell back to `bravo`
`src/segmentation/dataset`   its own default list
===========================  ========================================

and gated case discovery on a hardcoded folder-name allowlist
(``Mets_``, ``UCSF_``, ``Yale_``, ``BraTS_``, ``BMS_``, ...). A site whose
cases were named ``PT_001/`` got ``Loaded 0 cases`` and no error at all.

Here there is exactly one resolver, driven by
:class:`jannus.core.config.DataConfig`, and a case is a case because it
*contains the required imaging* — not because of what its folder is called.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..core.config import DataConfig
from ..core.errors import DataLayoutError
from ..core.logging import get_logger, pseudonymise

logger = get_logger("data.layout")

#: Extensions accepted for a volume, in preference order. `.nii.gz` first
#: because it is what every documented example uses.
VOLUME_SUFFIXES: tuple[str, ...] = (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd")


@dataclass
class ResolvedCase:
    """One case whose required sequences were all located on disk."""

    case_id: str
    directory: Path
    #: canonical channel name -> file that supplies it
    channels: dict[str, Path]
    #: canonical channel name -> the alias that actually matched, for the
    #: validation report ("t2 <- bravo.nii.gz" is worth showing a site).
    matched_aliases: dict[str, str] = field(default_factory=dict)
    ground_truth: Path | None = None

    @property
    def token(self) -> str:
        """Pseudonymised identifier, safe for logs and shared reports."""
        return pseudonymise(self.case_id)

    def ordered_paths(self, sequences: Sequence[str]) -> list[Path]:
        """Channel files in canonical model-input order."""
        return [self.channels[name] for name in sequences]


@dataclass
class UnresolvedCase:
    """A directory that looked like a case but is missing required imaging."""

    case_id: str
    directory: Path
    missing: list[str]
    found: dict[str, Path] = field(default_factory=dict)

    @property
    def token(self) -> str:
        return pseudonymise(self.case_id)

    def describe(self, aliases: dict[str, tuple[str, ...]]) -> str:
        parts = []
        for channel in self.missing:
            accepted = ", ".join(f"{a}{VOLUME_SUFFIXES[0]}" for a in aliases.get(channel, ()))
            parts.append(f"    {channel}: none of [{accepted}]")
        return f"{self.case_id}\n" + "\n".join(parts)


@dataclass
class DatasetIndex:
    """Result of scanning a dataset root."""

    root: Path
    cases: list[ResolvedCase] = field(default_factory=list)
    unresolved: list[UnresolvedCase] = field(default_factory=list)
    #: Directories skipped because `case_pattern` excluded them.
    excluded: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cases)

    @property
    def case_ids(self) -> list[str]:
        return [c.case_id for c in self.cases]


def _candidate_files(directory: Path) -> dict[str, Path]:
    """Map lowercase stem -> path for every volume-like file in `directory`.

    Matching is case-insensitive because sites vary (`T1c.nii.gz` vs
    `t1c.nii.gz`) and because Windows and Linux disagree about whether that
    distinction even exists.
    """
    found: dict[str, Path] = {}
    if not directory.is_dir():
        return found
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        name = path.name.lower()
        for suffix in VOLUME_SUFFIXES:
            if name.endswith(suffix):
                stem = name[: -len(suffix)]
                # First match wins, and VOLUME_SUFFIXES is preference-ordered,
                # so `t2.nii.gz` beats a stray `t2.nii` in the same folder.
                found.setdefault(stem, path)
                break
    return found


def resolve_case(
    directory: Path,
    data_config: DataConfig,
    *,
    ground_truth_names: Sequence[str] = ("seg", "gt", "label", "mask", "truth"),
) -> ResolvedCase | UnresolvedCase:
    """Locate every required channel inside one case directory."""
    available = _candidate_files(directory)
    case_id = directory.name

    channels: dict[str, Path] = {}
    matched: dict[str, str] = {}
    missing: list[str] = []

    for channel in data_config.sequences:
        aliases = data_config.sequence_aliases.get(channel, (channel,))
        for alias in aliases:
            hit = available.get(alias.lower())
            if hit is not None:
                channels[channel] = hit
                matched[channel] = hit.name
                break
        else:
            missing.append(channel)

    if missing:
        return UnresolvedCase(
            case_id=case_id, directory=directory, missing=missing, found=dict(channels)
        )

    ground_truth = next(
        (available[n] for n in ground_truth_names if n in available), None
    )
    return ResolvedCase(
        case_id=case_id,
        directory=directory,
        channels=channels,
        matched_aliases=matched,
        ground_truth=ground_truth,
    )


def scan_dataset(
    root: str | Path,
    data_config: DataConfig | None = None,
    *,
    require_ground_truth: bool = False,
) -> DatasetIndex:
    """Discover every case under `root`.

    Accepts either a single case directory (one containing the sequences
    directly) or a parent of many case directories, so a site does not have to
    reason about which shape the CLI wants.

    Args:
        root: dataset root, or a single case directory.
        data_config: sequence naming rules. Defaults to the shipped convention.
        require_ground_truth: treat cases without a segmentation as unresolved.
            Set by `jannus evaluate`, which cannot score a case without one.

    Raises:
        DataLayoutError: if `root` does not exist or contains no directories.
    """
    root = Path(root)
    cfg = data_config or DataConfig()

    if not root.exists():
        raise DataLayoutError(
            f"Dataset root does not exist: {root}",
            remedy="Pass --input pointing at the directory containing your case folders.",
        )
    if not root.is_dir():
        raise DataLayoutError(f"Dataset root is not a directory: {root}")

    index = DatasetIndex(root=root)

    # Case 1: `root` is itself a single case.
    single = resolve_case(root, cfg)
    if isinstance(single, ResolvedCase):
        if require_ground_truth and single.ground_truth is None:
            index.unresolved.append(
                UnresolvedCase(
                    case_id=single.case_id, directory=root, missing=["ground truth"]
                )
            )
        else:
            index.cases.append(single)
        return index

    # Case 2: `root` is a parent of case directories.
    pattern = re.compile(cfg.case_pattern) if cfg.case_pattern else None
    subdirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not subdirs:
        raise DataLayoutError(
            f"No case directories found under {root}, and {root} does not itself "
            f"contain the required sequences.",
            remedy=(
                "Expected either <root>/<case_id>/<sequence>.nii.gz or a single "
                "case directory. See docs/DATA_REQUIREMENTS.md."
            ),
        )

    for directory in subdirs:
        if pattern is not None and not pattern.search(directory.name):
            index.excluded.append(directory.name)
            continue
        resolved = resolve_case(directory, cfg)
        if isinstance(resolved, UnresolvedCase):
            index.unresolved.append(resolved)
        elif require_ground_truth and resolved.ground_truth is None:
            index.unresolved.append(
                UnresolvedCase(
                    case_id=resolved.case_id,
                    directory=directory,
                    missing=["ground truth"],
                    found=resolved.channels,
                )
            )
        else:
            index.cases.append(resolved)

    logger.info(
        "scanned dataset: %d resolved, %d unresolved, %d excluded",
        len(index.cases),
        len(index.unresolved),
        len(index.excluded),
    )

    # A scan that resolves nothing is always an error, never a quiet empty run.
    # This is the exact v1.40 failure mode being closed.
    if not index.cases:
        detail = ""
        if index.unresolved:
            sample = index.unresolved[: min(5, len(index.unresolved))]
            detail = "\n\nDirectories found, but missing required sequences:\n" + "\n".join(
                c.describe(cfg.sequence_aliases) for c in sample
            )
            if len(index.unresolved) > len(sample):
                detail += f"\n    ... and {len(index.unresolved) - len(sample)} more"
        if index.excluded:
            detail += (
                f"\n\n{len(index.excluded)} director(ies) were excluded by "
                f"data.case_pattern={cfg.case_pattern!r}."
            )
        raise DataLayoutError(
            f"No usable cases found under {root}.{detail}",
            remedy="Run `jannus validate-data --input <root>` for a full per-case report.",
        )

    return index


def format_alias_table(data_config: DataConfig) -> str:
    """Render the accepted filenames, for `--help` and the validation report."""
    lines = ["Required sequences (first matching filename wins):"]
    for channel in data_config.sequences:
        aliases = data_config.sequence_aliases.get(channel, ())
        rendered = ", ".join(f"{a}.nii.gz" for a in aliases)
        lines.append(f"  {channel:<8} {rendered}")
    return "\n".join(lines)


def iter_case_dirs(root: str | Path) -> Iterable[Path]:
    """Yield immediate subdirectories of `root`. Convenience for tooling."""
    root = Path(root)
    if root.is_dir():
        yield from (d for d in sorted(root.iterdir()) if d.is_dir())
