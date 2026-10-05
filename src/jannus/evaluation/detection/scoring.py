"""Per-component confidence scores for FROC analysis.

Each scorer maps every predicted component to one number in [0, 1], higher = more
confident. A FROC curve is drawn by sweeping a threshold over that number.
Scorers are registered by name in :data:`SCORERS` and selected in an
experiment config, so adding one never touches the metrics code.

``max_prob``       maximum stacker probability inside the component
``mean_prob``      mean stacker probability inside the component
``agreement``      fraction of base models (k/7) whose map reaches
                   ``base_threshold`` at one or more voxels inside the component
``inv_variance``   ``1 - mean_voxel_variance / 0.25``. The variance is the
                   population variance across base models, as in the
                   stacker's own variance channel
                   (``segmentation.stacking.build_stacking_features_from_preds``).
                   0.25 is the largest variance values in [0, 1] can have.

Caveat for ``inv_variance``: variance is also low where every base model is
confidently *negative*. A component that only the stacker switches on can
therefore score as "confident". ``agreement`` is the cross-check.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .matching import ComponentTable

#: Largest possible variance of values confined to [0, 1].
MAX_UNIT_VARIANCE = 0.25


@dataclass(frozen=True)
class ScoringInputs:
    """The probability maps a case's components are scored against.

    ``base`` holds one map per base model in ``base_names`` order. Pass ``None``
    when only stacker-based scorers are needed.
    """

    stacker: np.ndarray                      # (H, W, D) in [0, 1]
    base: np.ndarray | None = None           # (n_models, H, W, D) in [0, 1]
    base_names: tuple[str, ...] = ()
    _variance: dict[str, np.ndarray] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.stacker.ndim != 3:
            raise ValueError(f"stacker map must be 3D, got {self.stacker.shape}")
        _check_unit_range("stacker", self.stacker)
        if self.base is not None:
            if self.base.ndim != 4 or self.base.shape[1:] != self.stacker.shape:
                raise ValueError(
                    f"base maps must be (n_models, *{self.stacker.shape}), got {self.base.shape}"
                )
            if self.base_names and len(self.base_names) != self.base.shape[0]:
                raise ValueError("base_names does not match the number of base maps")
            _check_unit_range("base", self.base)

    @property
    def n_models(self) -> int:
        return 0 if self.base is None else int(self.base.shape[0])

    def require_base(self, scorer: str) -> np.ndarray:
        if self.base is None or self.n_models == 0:
            raise ValueError(f"scorer {scorer!r} needs base-model maps; none were provided")
        return self.base

    def variance(self) -> np.ndarray:
        """Voxel-wise population variance across base models, computed once per case."""
        if "v" not in self._variance:
            base = self.require_base("inv_variance")
            self._variance["v"] = base.var(axis=0, dtype=np.float64).astype(np.float32)
        return self._variance["v"]


def _check_unit_range(name: str, arr: np.ndarray) -> None:
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if lo < 0.0 or hi > 1.0 or not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(f"{name} map must lie in [0, 1]; got range [{lo}, {hi}]")


def dequantize_uint8(maps: np.ndarray) -> np.ndarray:
    """Inverse of the cache's uint8 storage, ``round(p * 255)``. Max error is 1/510."""
    if maps.dtype != np.uint8:
        raise ValueError(f"expected uint8, got {maps.dtype}")
    return maps.astype(np.float32) / 255.0


def quantize_uint8(maps: np.ndarray) -> np.ndarray:
    _check_unit_range("map", maps)
    return np.rint(np.asarray(maps, dtype=np.float64) * 255.0).astype(np.uint8)


Scorer = Callable[[ScoringInputs, ComponentTable], np.ndarray]


def _ids(table: ComponentTable) -> np.ndarray:
    return np.arange(1, table.n + 1)


def _check_grid(inputs: ScoringInputs, table: ComponentTable) -> None:
    if table.labels.shape != inputs.stacker.shape:
        raise ValueError(
            f"component grid {table.labels.shape} does not match maps {inputs.stacker.shape}"
        )


def score_max_prob(inputs: ScoringInputs, table: ComponentTable) -> np.ndarray:
    _check_grid(inputs, table)
    if table.n == 0:
        return np.zeros(0)
    return np.asarray(ndimage.maximum(inputs.stacker, table.labels, _ids(table)), dtype=np.float64)


def score_mean_prob(inputs: ScoringInputs, table: ComponentTable) -> np.ndarray:
    _check_grid(inputs, table)
    if table.n == 0:
        return np.zeros(0)
    return np.asarray(ndimage.mean(inputs.stacker, table.labels, _ids(table)), dtype=np.float64)


def make_agreement_scorer(base_threshold: float = 0.5) -> Scorer:
    """Fraction of base models with a voxel at or above ``base_threshold`` in the component.

    ``base_threshold`` defaults to 0.5, the per-model threshold in
    ``configs/models.yaml``. Only k/n_models distinct values are possible, so a
    FROC curve over this scorer has at most n_models + 1 points per voxel
    threshold. Ties are not broken.
    """
    if not 0.0 < base_threshold <= 1.0:
        raise ValueError("base_threshold must be in (0, 1]")

    def score_agreement(inputs: ScoringInputs, table: ComponentTable) -> np.ndarray:
        _check_grid(inputs, table)
        base = inputs.require_base("agreement")
        if table.n == 0:
            return np.zeros(0)
        ids = _ids(table)
        votes = np.zeros(table.n)
        for m in range(base.shape[0]):
            peak = np.asarray(ndimage.maximum(base[m], table.labels, ids), dtype=np.float64)
            votes += peak >= base_threshold
        return votes / base.shape[0]

    return score_agreement


def score_inv_variance(inputs: ScoringInputs, table: ComponentTable) -> np.ndarray:
    _check_grid(inputs, table)
    if table.n == 0:
        inputs.require_base("inv_variance")
        return np.zeros(0)
    mean_var = np.asarray(ndimage.mean(inputs.variance(), table.labels, _ids(table)), dtype=np.float64)
    return np.clip(1.0 - mean_var / MAX_UNIT_VARIANCE, 0.0, 1.0)


def build_scorers(config: Mapping[str, Mapping[str, float]] | Sequence[str]) -> dict[str, Scorer]:
    """Instantiate scorers from config: a list of names, or ``{name: {param: value}}``."""
    items = config.items() if isinstance(config, Mapping) else ((n, {}) for n in config)
    out: dict[str, Scorer] = {}
    for name, params in items:
        if name not in SCORER_FACTORIES:
            raise ValueError(f"unknown scorer {name!r}; available: {sorted(SCORER_FACTORIES)}")
        out[name] = SCORER_FACTORIES[name](**dict(params or {}))
    if not out:
        raise ValueError("at least one scorer must be configured")
    return out


def score_components(
    scorers: Mapping[str, Scorer], inputs: ScoringInputs, table: ComponentTable
) -> dict[str, np.ndarray]:
    """Every configured score for every component, keyed by scorer name."""
    out = {}
    for name, fn in scorers.items():
        scores = fn(inputs, table)
        if scores.shape != (table.n,) or not np.all((scores >= 0) & (scores <= 1)):
            raise RuntimeError(f"scorer {name!r} returned out-of-contract scores")
        out[name] = scores
    return out


def _fixed(fn: Scorer) -> Callable[..., Scorer]:
    def factory(**params: float) -> Scorer:
        if params:
            raise ValueError(f"scorer takes no parameters, got {sorted(params)}")
        return fn

    return factory


#: name -> factory(**params) -> scorer. Config names refer to these keys.
SCORER_FACTORIES: dict[str, Callable[..., Scorer]] = {
    "max_prob": _fixed(score_max_prob),
    "mean_prob": _fixed(score_mean_prob),
    "agreement": make_agreement_scorer,
    "inv_variance": _fixed(score_inv_variance),
}

#: Default-parameter instances, for callers that do not need configuration.
SCORERS: dict[str, Scorer] = build_scorers(list(SCORER_FACTORIES))
