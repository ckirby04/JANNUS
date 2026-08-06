"""Configuration loading, validation and path resolution.

This module fixes a class of defect that silently invalidated cross-site
comparison in v1.40: configuration keys that were declared in
``configs/models.yaml`` but never read by the code, so a site could edit a
value, observe no effect, and never be told.

Three guarantees:

1. **Every path resolves relative to the config file**, never to the process
   working directory. In v1.40 base-model paths resolved against `cwd` while
   the stacker resolved against the config's parent — two mechanisms in one
   loader, and both broke when a site ran from anywhere but the repo root.

2. **Unknown and unread keys are reported.** :func:`audit_config_keys` returns
   keys present in the file that no consumer reads, so dead config is visible
   instead of misleading.

3. **The operating point is explicit.** ``inference.threshold`` is required and
   validated. See :data:`PUBLISHED_THRESHOLD`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError

#: The binarisation threshold used to produce every published JANNUS metric
#: (Dice 0.7858, lesion sensitivity 0.810, RANO-BM >=10mm sensitivity 0.943).
#:
#: HISTORICAL DEFECT — v1.40 and earlier: `configs/models.yaml` never defined
#: `inference.threshold`, so `BrainMetPipeline` fell through to a hardcoded
#: 0.5, while the evaluation scripts that generated the published numbers used
#: 0.55. Any site running the documented inference command reproduced neither
#: the config nor the paper. v1.50 makes the key required and defaults it here.
#: Changing this value invalidates comparison against published results.
PUBLISHED_THRESHOLD = 0.55

#: Base-model names, in the exact order StackingClassifierV2 was trained to
#: expect. Channel order is a hard contract: permuting it produces a valid
#: shaped output that is quietly wrong.
CANONICAL_MODEL_ORDER: tuple[str, ...] = (
    "nnunet_3d",
    "nnunet_2d",
    "patch_8",
    "patch_12",
    "patch_24",
    "patch_36",
    "swin_unetr",
)

#: Keys the production inference path actually consumes. Anything in the config
#: outside this set is reported by `audit_config_keys` as inert.
CONSUMED_KEYS: frozenset[str] = frozenset(
    {
        "models",
        "stacking.classifier_path",
        "stacking.in_channels",
        "stacking.mid_channels",
        "stacking.n_blocks",
        "stacking.n_blocks_lowres",
        "stacking.patch_size",
        "stacking.overlap",
        "inference.threshold",
        "inference.postprocessing.min_size",
        "inference.postprocessing.opening_size",
        "inference.postprocessing.closing_size",
        "data.sequences",
        "data.sequence_aliases",
        "data.case_pattern",
    }
)


@dataclass
class PostprocessingConfig:
    """Morphological cleanup applied after thresholding."""

    #: v1.40 shipped 0 deliberately: a sweep showed min_size=20 cost 0.06 Dice
    #: on the tiny-lesion bin. Raising it trades small-lesion recall for fewer
    #: false positives — a regulatory decision, not a tuning knob.
    min_size: int = 0
    opening_size: int = 1
    closing_size: int = 1


@dataclass
class InferenceConfig:
    """The operating point."""

    threshold: float = PUBLISHED_THRESHOLD
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise ConfigError(
                f"inference.threshold must lie in (0, 1); got {self.threshold}."
            )


@dataclass
class StackingConfig:
    """StackingClassifierV2 hyperparameters and checkpoint location."""

    classifier_path: str = "model/stacking_classifier_production.pth"
    in_channels: int = 9
    mid_channels: int = 64
    n_blocks: int = 4
    n_blocks_lowres: int = 2
    patch_size: int = 32
    overlap: float = 0.5

    def __post_init__(self) -> None:
        # 7 base predictions + voxel-wise variance + (max - min) range.
        expected = len(CANONICAL_MODEL_ORDER) + 2
        if self.in_channels != expected:
            raise ConfigError(
                f"stacking.in_channels must be {expected} "
                f"({len(CANONICAL_MODEL_ORDER)} base predictions + variance + range); "
                f"got {self.in_channels}."
            )
        if not 0.0 <= self.overlap < 1.0:
            raise ConfigError(f"stacking.overlap must lie in [0, 1); got {self.overlap}.")


@dataclass
class DataConfig:
    """How this site's imaging is laid out on disk.

    v1.40 resolved sequence naming four different ways across four entry points
    (the CLI defaulted to `bravo`, the API hardcoded `t2`, the demo probed for
    one then the other, the dataset loader had its own default). A site whose
    files were named either way hit a different failure depending on which
    entry point it used. This is now the single source of truth.
    """

    #: Canonical channel order fed to the models: T1 pre, T1 post-Gd, FLAIR, T2.
    sequences: tuple[str, ...] = ("t1_pre", "t1_gd", "flair", "t2")

    #: Accepted on-disk filenames per canonical channel, tried in order. This is
    #: what lets one config serve BrainMetShare (`bravo`), TCIA-style
    #: (`T1c`/`T2w`), and BIDS-ish naming without any code change.
    sequence_aliases: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "t1_pre": ("t1_pre", "t1", "T1", "t1_native", "T1w"),
            "t1_gd": ("t1_gd", "t1c", "T1c", "t1_post", "t1ce", "T1wCE", "t1_gd_reg"),
            "flair": ("flair", "FLAIR", "t2_flair", "T2wFLAIR"),
            "t2": ("t2", "T2", "bravo", "BRAVO", "T2w"),
        }
    )

    #: Optional regex constraining which subdirectories count as cases. Left
    #: empty by default: v1.40 hardcoded a prefix allowlist
    #: (`Mets_`, `UCSF_`, `Yale_`, ...) so a site with `PT_001/` loaded zero
    #: cases and was told nothing. Presence of the required sequences is the
    #: correct test, not the folder name.
    case_pattern: str = ""


@dataclass
class JannusConfig:
    """Fully-resolved, validated configuration."""

    models: list[dict[str, Any]]
    stacking: StackingConfig
    inference: InferenceConfig
    data: DataConfig
    #: Directory all relative checkpoint paths resolve against.
    root: Path
    source_path: Path
    #: Config keys present in the file that nothing reads.
    inert_keys: list[str] = field(default_factory=list)

    @property
    def model_names(self) -> list[str]:
        return [m["name"] for m in self.models]

    def checkpoint_paths(self) -> dict[str, Path]:
        """Absolute path for every checkpoint this config references."""
        paths: dict[str, Path] = {}
        for entry in self.models:
            if "path" in entry:
                paths[entry["name"]] = self.resolve(entry["path"])
        paths["stacker"] = self.resolve(self.stacking.classifier_path)
        return paths

    def resolve(self, relative: str | Path) -> Path:
        """Resolve a config-relative path against the config root."""
        p = Path(relative)
        return p if p.is_absolute() else (self.root / p)


def _flatten_keys(node: Any, prefix: str = "") -> list[str]:
    """Dotted key paths for every leaf in a nested mapping."""
    keys: list[str] = []
    if isinstance(node, Mapping):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, Mapping):
                keys.extend(_flatten_keys(value, path))
            else:
                keys.append(path)
    return keys


def audit_config_keys(raw: Mapping[str, Any]) -> list[str]:
    """Return config keys that no consumer reads.

    Reported by `jannus doctor` so a site never edits a value that does nothing.
    `models` is a list of heterogeneous entries consumed wholesale by the
    adapter factory, so its subkeys are exempt.
    """
    inert: list[str] = []
    for key in _flatten_keys(raw):
        if key.startswith("models"):
            continue
        if key in CONSUMED_KEYS:
            continue
        # A parent being consumed covers its children.
        if any(key.startswith(f"{consumed}.") for consumed in CONSUMED_KEYS):
            continue
        inert.append(key)
    return sorted(inert)


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"Config section {name!r} must be a mapping; got {type(value).__name__}.")
    return dict(value)


def load_config(
    config_path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
) -> JannusConfig:
    """Load, validate and resolve ``models.yaml``.

    Args:
        config_path: path to the YAML config.
        overrides: flat dotted-key overrides applied after parsing, e.g.
            ``{"inference.threshold": 0.5}`` from a CLI flag.
        root: directory relative checkpoint paths resolve against. Defaults to
            the config file's parent's parent (``configs/`` sits one level below
            the repo root), which is what makes ``model/patch_8.pth`` work from
            any working directory.

    Raises:
        ConfigError: on missing files, malformed YAML, or a config that could
            not produce valid inference.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise ConfigError(
            f"Config file not found: {config_path}",
            remedy="Pass --config pointing at configs/models.yaml.",
        )

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {config_path}: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise ConfigError(f"{config_path} must contain a YAML mapping at the top level.")

    raw = dict(raw)
    for dotted, value in (overrides or {}).items():
        _apply_override(raw, dotted, value)

    models = raw.get("models")
    if not isinstance(models, Sequence) or isinstance(models, (str, bytes)) or not models:
        raise ConfigError(f"{config_path} must define a non-empty `models` list.")
    models = [dict(m) for m in models]

    declared = [m.get("name") for m in models]
    missing = [n for n in CANONICAL_MODEL_ORDER if n not in declared]
    if missing:
        raise ConfigError(
            f"{config_path} is missing required base model entr(ies): {', '.join(missing)}. "
            f"The production stacker expects all of {list(CANONICAL_MODEL_ORDER)}.",
            remedy="Restore the missing entries or use the shipped configs/models.yaml.",
        )
    # Order the list canonically so downstream code never depends on file order.
    by_name = {m["name"]: m for m in models}
    ordered = [by_name[name] for name in CANONICAL_MODEL_ORDER]

    stacking_raw = _require_mapping(raw.get("stacking"), "stacking")
    inference_raw = _require_mapping(raw.get("inference"), "inference")
    pp_raw = _require_mapping(inference_raw.get("postprocessing"), "inference.postprocessing")
    data_raw = _require_mapping(raw.get("data"), "data")

    stacking = StackingConfig(
        **{k: v for k, v in stacking_raw.items() if k in StackingConfig.__dataclass_fields__}
    )
    inference = InferenceConfig(
        threshold=float(inference_raw.get("threshold", PUBLISHED_THRESHOLD)),
        postprocessing=PostprocessingConfig(
            min_size=int(pp_raw.get("min_size", 0)),
            opening_size=int(pp_raw.get("opening_size", 1)),
            closing_size=int(pp_raw.get("closing_size", 1)),
        ),
    )

    data = DataConfig()
    if "sequences" in data_raw:
        data.sequences = tuple(data_raw["sequences"])
    if "sequence_aliases" in data_raw:
        data.sequence_aliases = {
            k: tuple(v) for k, v in dict(data_raw["sequence_aliases"]).items()
        }
    if "case_pattern" in data_raw:
        data.case_pattern = str(data_raw["case_pattern"])

    for channel in data.sequences:
        if channel not in data.sequence_aliases:
            raise ConfigError(
                f"data.sequences lists {channel!r} but data.sequence_aliases has no "
                f"entry for it, so JANNUS cannot tell which file on disk supplies "
                f"that channel."
            )

    resolved_root = Path(root) if root is not None else config_path.resolve().parent.parent

    return JannusConfig(
        models=ordered,
        stacking=stacking,
        inference=inference,
        data=data,
        root=resolved_root,
        source_path=config_path.resolve(),
        inert_keys=audit_config_keys(raw),
    )


def _apply_override(target: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``a.b.c = value`` inside a nested dict, creating levels as needed."""
    parts = dotted.split(".")
    node: dict[str, Any] = target
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value
