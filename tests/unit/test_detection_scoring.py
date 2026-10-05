"""Tests for `jannus.evaluation.detection.scoring`.

Maps are built so that every expected score can be worked out by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from jannus.evaluation.detection.matching import extract_components
from jannus.evaluation.detection.scoring import (
    SCORERS,
    ScoringInputs,
    build_scorers,
    dequantize_uint8,
    make_agreement_scorer,
    quantize_uint8,
    score_components,
)

ISO = (1.0, 1.0, 1.0)
SHAPE = (20, 20, 20)
A = (slice(2, 5), slice(2, 5), slice(2, 5))       # component 1: 27 voxels
B = (slice(12, 14), slice(12, 14), slice(12, 14))  # component 2: 8 voxels


def setup():
    stacker = np.zeros(SHAPE, dtype=np.float32)
    stacker[A] = 0.6
    stacker[3, 3, 3] = 0.9            # peak inside A
    stacker[B] = 0.7
    base = np.zeros((7, *SHAPE), dtype=np.float32)
    base[:5][(slice(None), *A)] = 0.8  # 5/7 models fire on A
    base[0][B] = 0.55                  # 1/7 fires on B
    base[1][B] = 0.45                  # just below the base threshold
    table = extract_components(stacker > 0.5, ISO)
    return ScoringInputs(stacker=stacker, base=base), table


def test_max_and_mean_prob():
    inputs, table = setup()
    assert table.n == 2
    assert SCORERS["max_prob"](inputs, table) == pytest.approx([0.9, 0.7])
    assert SCORERS["mean_prob"](inputs, table) == pytest.approx([(26 * 0.6 + 0.9) / 27, 0.7])


def test_agreement_counts_models_reaching_base_threshold():
    inputs, table = setup()
    assert SCORERS["agreement"](inputs, table) == pytest.approx([5 / 7, 1 / 7])
    assert make_agreement_scorer(0.4)(inputs, table) == pytest.approx([5 / 7, 2 / 7])


def test_agreement_uses_any_voxel_in_component():
    inputs, table = setup()
    base = inputs.base.copy()
    base[6][4, 4, 4] = 0.99  # a single voxel of model 7 inside A
    s = SCORERS["agreement"](ScoringInputs(stacker=inputs.stacker, base=base), table)
    assert s[0] == pytest.approx(6 / 7)


def test_inv_variance_matches_population_variance():
    inputs, table = setup()
    # In A: five models at 0.8, two at 0.0 -> population variance = p(1-p) * 0.8^2, p = 5/7.
    var_a = (5 / 7) * (2 / 7) * 0.8**2
    vals_b = np.array([0.55, 0.45, 0, 0, 0, 0, 0])
    var_b = vals_b.var()
    s = SCORERS["inv_variance"](inputs, table)
    assert s == pytest.approx([1 - var_a / 0.25, 1 - var_b / 0.25], rel=1e-5)


def test_inv_variance_is_high_when_all_models_agree_negative():
    # Documented caveat: stacker-only components look "confident" under inv_variance.
    stacker = np.zeros(SHAPE, dtype=np.float32)
    stacker[A] = 0.8
    inputs = ScoringInputs(stacker=stacker, base=np.zeros((7, *SHAPE), dtype=np.float32))
    table = extract_components(stacker > 0.5, ISO)
    assert SCORERS["inv_variance"](inputs, table) == pytest.approx([1.0])
    assert SCORERS["agreement"](inputs, table) == pytest.approx([0.0])


def test_empty_table_gives_empty_scores():
    inputs, _ = setup()
    empty = extract_components(np.zeros(SHAPE, dtype=bool), ISO)
    for name, fn in SCORERS.items():
        assert fn(inputs, empty).shape == (0,), name


def test_base_scorers_need_base_maps():
    inputs, table = setup()
    stacker_only = ScoringInputs(stacker=inputs.stacker)
    assert SCORERS["max_prob"](stacker_only, table).shape == (2,)
    for name in ("agreement", "inv_variance"):
        with pytest.raises(ValueError, match="needs base-model maps"):
            SCORERS[name](stacker_only, table)


def test_input_validation():
    inputs, _ = setup()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ScoringInputs(stacker=inputs.stacker * 2)
    with pytest.raises(ValueError, match="base maps must be"):
        ScoringInputs(stacker=inputs.stacker, base=inputs.base[:, :10])
    with pytest.raises(ValueError, match="base_names"):
        ScoringInputs(stacker=inputs.stacker, base=inputs.base, base_names=("a",))
    other = extract_components(np.ones((5, 5, 5), dtype=bool), ISO)
    with pytest.raises(ValueError, match="does not match maps"):
        SCORERS["max_prob"](inputs, other)


def test_build_scorers_from_config():
    s = build_scorers({"max_prob": {}, "agreement": {"base_threshold": 0.4}})
    assert list(s) == ["max_prob", "agreement"]
    inputs, table = setup()
    assert s["agreement"](inputs, table) == pytest.approx([5 / 7, 2 / 7])
    assert list(build_scorers(["mean_prob"])) == ["mean_prob"]
    with pytest.raises(ValueError, match="unknown scorer"):
        build_scorers(["dice"])
    with pytest.raises(ValueError, match="no parameters"):
        build_scorers({"max_prob": {"x": 1.0}})
    with pytest.raises(ValueError, match="at least one"):
        build_scorers([])
    with pytest.raises(ValueError):
        make_agreement_scorer(0.0)


def test_score_components_returns_all_and_checks_contract():
    inputs, table = setup()
    out = score_components(SCORERS, inputs, table)
    assert set(out) == {"max_prob", "mean_prob", "agreement", "inv_variance"}
    assert all(v.shape == (2,) for v in out.values())
    with pytest.raises(RuntimeError, match="out-of-contract"):
        score_components({"bad": lambda i, t: np.full(t.n, 2.0)}, inputs, table)


def test_uint8_round_trip_error_bound():
    rng = np.random.default_rng(0)
    p = rng.random((7, 8, 8, 8)).astype(np.float32)
    q = dequantize_uint8(quantize_uint8(p))
    assert np.abs(q - p).max() <= 1 / 510 + 1e-7
    assert quantize_uint8(np.array([0.0, 1.0])).tolist() == [0, 255]
    with pytest.raises(ValueError):
        dequantize_uint8(p)
