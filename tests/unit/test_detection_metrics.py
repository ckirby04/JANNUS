"""Tests for `jannus.evaluation.detection.metrics`, `.bootstrap` and `.provenance`.

The cohorts are tiny and hand-built, so every FROC point can be checked by
counting.
"""

from __future__ import annotations

import numpy as np
import pytest

from jannus.evaluation.detection.bootstrap import bootstrap, patient_case_weights
from jannus.evaluation.detection.metrics import (
    DetectionData,
    concat_cases,
    cpm,
    froc,
    operating_point,
    records_from_match,
    summarize_curve,
)
from jannus.evaluation.detection.provenance import config_hash

NEG = -np.inf
BINS = ("<3", "3-5", "5-10", ">10")


def cohort() -> DetectionData:
    """Three scans, two patients (P1 has two timepoints).

    s1 (P1): lesions scored 0.9 (<3) and 0.4 (3-5); one FP at 0.8
    s2 (P1): one missed lesion (5-10); FPs at 0.3 and 0.2
    s3 (P2): no lesions; one FP at 0.6
    """
    return concat_cases([
        ("s1", "P1", np.array([0.9, 0.4]), np.array(["<3", "3-5"]), np.array([0.8])),
        ("s2", "P1", np.array([NEG]), np.array(["5-10"]), np.array([0.3, 0.2])),
        ("s3", "P2", np.array([]), np.array([]), np.array([0.6])),
    ])


def test_froc_points_by_hand():
    c = froc(cohort())
    # tau:   inf  0.9  0.8  0.6  0.4  0.3  0.2
    assert c.thresholds[0] == np.inf
    assert c.thresholds[1:].tolist() == [0.9, 0.8, 0.6, 0.4, 0.3, 0.2]
    assert c.sensitivity.tolist() == pytest.approx([0, 1 / 3, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 2 / 3])
    assert c.fp_per_scan.tolist() == pytest.approx([0, 0, 1 / 3, 2 / 3, 2 / 3, 1, 4 / 3])
    assert (c.n_lesions, c.n_scans) == (3, 3)


def test_sensitivity_at_fp_is_conservative_step():
    c = froc(cohort())
    assert c.sensitivity_at_fp(0.5) == (pytest.approx(1 / 3), 0.8, True)
    assert c.sensitivity_at_fp(1.0) == (pytest.approx(2 / 3), 0.3, True)
    sens, tau, reached = c.sensitivity_at_fp(3.0)
    assert (sens, tau, reached) == (pytest.approx(2 / 3), 0.2, False)  # curve ends at 4/3 FP/scan
    assert cpm(c, (0.5, 1.0)) == pytest.approx(0.5)


def test_froc_restricted_to_size_bin_keeps_all_fps():
    d = cohort()
    c = froc(d, gt_mask=d.gt_bin == "3-5")
    assert c.n_lesions == 1
    assert c.sensitivity_at_fp(1.0)[0] == 1.0
    assert c.fp_per_scan[-1] == pytest.approx(4 / 3)


def test_operating_point_counts_and_scan_rates():
    op = operating_point(cohort(), 0.35, bins=BINS)
    assert op["n_lesions"] == 3 and op["n_detected"] == 2
    assert op["sensitivity"] == pytest.approx(2 / 3)
    assert op["n_fp"] == 2 and op["fp_per_scan"] == pytest.approx(2 / 3)
    assert op["sensitivity[<3]"] == 1.0 and op["n_lesions[<3]"] == 1
    assert op["sensitivity[5-10]"] == 0.0
    assert np.isnan(op["sensitivity[>10]"]) and op["n_lesions[>10]"] == 0
    assert op["n_scans_with_lesions"] == 2
    assert op["scan_any_flagged"] == pytest.approx(0.5)   # s1 yes, s2 no
    assert op["scan_all_flagged"] == pytest.approx(0.5)   # s1 all, s2 none
    assert op["n_scans_without_lesions"] == 1
    assert op["fp_per_scan_lesion_free"] == 1.0
    # A stricter tau loses the 0.4 lesion: s1 still has one flag but not all.
    op2 = operating_point(cohort(), 0.5, bins=BINS)
    assert op2["scan_any_flagged"] == pytest.approx(0.5)
    assert op2["scan_all_flagged"] == 0.0


def test_case_weights_equal_duplicated_records():
    d = cohort()
    w = np.array([2.0, 0.0, 1.0])
    dup = concat_cases([
        ("s1a", "P1", np.array([0.9, 0.4]), np.array(["<3", "3-5"]), np.array([0.8])),
        ("s1b", "P1", np.array([0.9, 0.4]), np.array(["<3", "3-5"]), np.array([0.8])),
        ("s3", "P2", np.array([]), np.array([]), np.array([0.6])),
    ])
    for tau in (0.3, 0.5, 0.85):
        a = operating_point(d, tau, bins=BINS, case_weight=w)
        b = operating_point(dup, tau, bins=BINS)
        for k in a:
            if k != "tau":
                assert a[k] == pytest.approx(b[k], nan_ok=True), k
    ca, cb = froc(d, case_weight=w), froc(dup)
    for t in (0.5, 1.0, 2.0):
        assert ca.sensitivity_at_fp(t)[0] == pytest.approx(cb.sensitivity_at_fp(t)[0])


def test_summarize_uses_overall_threshold_for_bins():
    s = summarize_curve(cohort(), bins=BINS, targets=(0.5, 1.0))
    assert s["sens@0.5"] == pytest.approx(1 / 3) and s["tau@0.5"] == 0.8
    assert s["sens@0.5[<3]"] == 1.0 and s["sens@0.5[3-5]"] == 0.0
    assert s["sens@1[3-5]"] == 1.0
    assert s["cpm"] == pytest.approx(0.5)


def test_subset_reindexes():
    d = cohort().subset(["s3", "s1"])
    assert d.case_ids == ("s3", "s1")
    assert d.gt_case.tolist() == [1, 1]
    assert sorted(d.fp_score.tolist()) == [0.6, 0.8]


def test_validation_errors():
    with pytest.raises(ValueError, match="duplicate"):
        concat_cases([("a", "P", np.array([]), np.array([]), np.array([]))] * 2)
    with pytest.raises(ValueError, match="finite"):
        concat_cases([("a", "P", np.array([np.nan]), np.array(["<3"]), np.array([]))])
    with pytest.raises(ValueError, match="case_weight"):
        operating_point(cohort(), 0.5, case_weight=np.array([1.0]))


def test_empty_cohort_pieces():
    d = concat_cases([("a", "P", np.array([]), np.array([]), np.array([]))])
    c = froc(d)
    assert c.thresholds.tolist() == [np.inf]
    assert np.isnan(c.sensitivity[0]) and c.fp_per_scan[0] == 0


def test_records_from_match_uses_fp_flags():
    from jannus.evaluation.detection.matching import extract_components, match_by_overlap

    gt = np.zeros((10, 10, 10), np.uint8)
    gt[1:3, 1:3, 1:3] = 1
    pred = gt.copy()
    pred[6:8, 6:8, 6:8] = 1
    m = match_by_overlap(extract_components(pred, (1, 1, 1)), extract_components(gt, (1, 1, 1)),
                         scores=np.array([0.7, 0.2]))
    gs, gb, fs = records_from_match(m, ["<3"])
    assert gs.tolist() == [0.7] and gb.tolist() == ["<3"] and fs.tolist() == [0.2]
    with pytest.raises(ValueError):
        records_from_match(m, [])


# --------------------------------------------------------------------------- bootstrap


def test_patient_weights_keep_timepoints_together():
    d = cohort()
    w, n_pat = patient_case_weights(d, 500, seed=1)
    assert n_pat == 2 and w.shape == (500, 3)
    assert np.array_equal(w[:, 0], w[:, 1])        # s1 and s2 belong to P1
    assert np.all(w.sum(axis=1) == w[:, 0] * 2 + w[:, 2])
    assert np.all(w[:, 0] + w[:, 2] == 2)          # 2 patients drawn per resample


def test_bootstrap_is_seeded_and_brackets_point():
    d = cohort()

    def stat(data, w):
        return {k: operating_point(data, 0.35, bins=BINS, case_weight=w)[k]
                for k in ("sensitivity", "fp_per_scan", "sensitivity[>10]")}

    a = bootstrap(d, stat, n_resamples=300, seed=7)
    b = bootstrap(d, stat, n_resamples=300, seed=7)
    for ra, rb in zip(a.as_rows(), b.as_rows(), strict=True):  # NaN-aware equality
        assert ra == pytest.approx(rb, nan_ok=True)
    assert a.ci_low["fp_per_scan"] <= a.point["fp_per_scan"] <= a.ci_high["fp_per_scan"]
    assert a.n_valid["sensitivity[>10]"] == 0         # never defined: no >10 mm lesions
    assert np.isnan(a.ci_low["sensitivity[>10]"])
    assert a.n_valid["sensitivity"] < 300              # resamples drawing only P2 have no lesions
    with pytest.raises(ValueError):
        bootstrap(d, stat, n_resamples=0)


# --------------------------------------------------------------------------- provenance


def test_config_hash_ignores_key_order():
    assert config_hash({"a": 1, "b": {"c": 2, "d": 3}}) == config_hash({"b": {"d": 3, "c": 2}, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})
