"""Patient-level bootstrap for detection metrics.

Each resample draws **patients** with replacement, not lesions and not scans.
Every scan of a drawn patient enters with that patient's multiplicity, and
the statistic is recomputed from scratch through the case weights that every
function in :mod:`.metrics` accepts. Lesions within one patient are
correlated, so resampling lesions would give CIs that are too narrow.
Resampling scans would do the same for the UCSF patients who have several
timepoints.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .metrics import DetectionData

Statistic = Callable[[DetectionData, np.ndarray], dict[str, float]]


@dataclass(frozen=True)
class BootstrapResult:
    point: dict[str, float]
    ci_low: dict[str, float]
    ci_high: dict[str, float]
    n_valid: dict[str, int]          # resamples in which the statistic was defined
    n_resamples: int
    seed: int
    n_patients: int

    def as_rows(self) -> list[dict[str, float | str | int]]:
        return [
            {"metric": k, "point": v, "ci_low": self.ci_low[k], "ci_high": self.ci_high[k],
             "n_valid_resamples": self.n_valid[k]}
            for k, v in self.point.items()
        ]


def patient_case_weights(
    data: DetectionData, n_resamples: int, seed: int
) -> tuple[np.ndarray, int]:
    """``(weights, n_patients)``. ``weights`` has shape (n_resamples, n_cases) and holds
    each case's multiplicity under patient-level resampling."""
    patients, case_to_patient = np.unique(np.asarray(data.patient_ids), return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(patients), size=(n_resamples, len(patients)))
    counts = np.zeros((n_resamples, len(patients)))
    np.add.at(counts, (np.arange(n_resamples)[:, None], draws), 1.0)
    return counts[:, case_to_patient], len(patients)


def bootstrap(
    data: DetectionData,
    statistic: Statistic,
    *,
    n_resamples: int = 2000,
    seed: int = 20261004,
    ci: float = 0.95,
) -> BootstrapResult:
    """Percentile CIs for every key returned by ``statistic(data, case_weight)``.

    Resamples in which a value is NaN (for example, a size bin with no lesions
    drawn) are dropped for that key only. ``n_valid`` records how many remain,
    so a CI resting on few resamples is visible.
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    point = statistic(data, np.ones(data.n_cases))
    weights, n_patients = patient_case_weights(data, n_resamples, seed)
    samples: dict[str, list[float]] = {k: [] for k in point}
    for w in weights:
        stat = statistic(data, w)
        for k in point:
            samples[k].append(stat.get(k, float("nan")))
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    ci_low, ci_high, n_valid = {}, {}, {}
    for k, vals in samples.items():
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        n_valid[k] = int(arr.size)
        ci_low[k] = float(np.percentile(arr, lo_q)) if arr.size else float("nan")
        ci_high[k] = float(np.percentile(arr, hi_q)) if arr.size else float("nan")
    return BootstrapResult(point=point, ci_low=ci_low, ci_high=ci_high, n_valid=n_valid,
                           n_resamples=n_resamples, seed=seed, n_patients=n_patients)
