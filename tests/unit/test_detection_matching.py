"""Tests for `jannus.evaluation.detection.matching` and `.geometry`.

Every expectation is hand-derived from the synthetic geometry. The edge cases
(merge, split, tiny fragment, anisotropic spacing, empty volumes) are the ones
DECISIONS.md commits to, so a change in counting semantics fails here first.
"""

from __future__ import annotations

import numpy as np
import pytest

from jannus.evaluation.detection.geometry import (
    assign_size_bin,
    equivalent_spherical_diameter,
    resampled_spacing,
    validate_size_bins,
)
from jannus.evaluation.detection.matching import (
    extract_components,
    match,
    match_by_centroid,
    match_by_overlap,
)

ISO = (1.0, 1.0, 1.0)
SHAPE = (40, 40, 40)
BINS = [("<3", 0.0, 3.0), ("3-5", 3.0, 5.0), ("5-10", 5.0, 10.0), (">10", 10.0, float("inf"))]


def box(vol: np.ndarray, lo, hi) -> np.ndarray:
    vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 1
    return vol


def empty() -> np.ndarray:
    return np.zeros(SHAPE, dtype=np.uint8)


def tables(pred, gt, spacing=ISO, connectivity=26):
    return (
        extract_components(pred, spacing, connectivity=connectivity),
        extract_components(gt, spacing, connectivity=connectivity),
    )


# --------------------------------------------------------------------------- geometry


def test_resampled_spacing_recovers_true_voxel_size():
    # BMS native grid 256x256x150 @ 0.9375x0.9375x1.0 resized to 256^3.
    sp = resampled_spacing((256, 256, 150), (0.9375, 0.9375, 1.0), (256, 256, 256))
    assert sp == pytest.approx((0.9375, 0.9375, 150 / 256))


def test_resampled_spacing_rejects_bad_input():
    with pytest.raises(ValueError):
        resampled_spacing((256, 256), (1.0, 1.0), (256, 256))
    with pytest.raises(ValueError):
        resampled_spacing((256, 256, 150), (1.0, 0.0, 1.0), (256, 256, 256))


def test_esd_of_unit_sphere_volume():
    assert equivalent_spherical_diameter(4 / 3 * np.pi) == pytest.approx(2.0)


def test_size_bins_half_open_and_validated():
    assert assign_size_bin(2.999, BINS) == "<3"
    assert assign_size_bin(3.0, BINS) == "3-5"
    assert assign_size_bin(10.0, BINS) == ">10"
    with pytest.raises(ValueError, match="contiguous"):
        validate_size_bins([("a", 0.0, 3.0), ("b", 4.0, float("inf"))])
    with pytest.raises(ValueError, match="start at 0"):
        validate_size_bins([("a", 1.0, float("inf"))])


# --------------------------------------------------------------------------- components


def test_component_measurements_use_anisotropic_spacing():
    gt = box(empty(), (10, 10, 10), (12, 13, 14))  # 2 x 3 x 4 = 24 voxels
    spacing = (3.0, 1.0, 0.5)
    t = extract_components(gt, spacing)
    assert t.n == 1
    assert t.n_vox[0] == 24
    assert t.volume_mm3[0] == pytest.approx(24 * 1.5)
    assert t.esd_mm[0] == pytest.approx((6 * 36 / np.pi) ** (1 / 3))
    # Extents in mm: 2*3=6, 3*1=3, 4*0.5=2 -> longest is the z axis despite fewest voxels.
    assert t.longest_axis_mm[0] == pytest.approx(6.0)
    assert t.centroid_mm[0] == pytest.approx([10.5 * 3.0, 11.0 * 1.0, 11.5 * 0.5])


def test_connectivity_6_vs_26():
    m = empty()
    m[5, 5, 5] = 1
    m[6, 6, 6] = 1  # corner-adjacent only
    assert extract_components(m, ISO, connectivity=26).n == 1
    assert extract_components(m, ISO, connectivity=6).n == 2
    with pytest.raises(ValueError):
        extract_components(m, ISO, connectivity=8)


def test_scalar_spacing_rejected():
    with pytest.raises((ValueError, TypeError)):
        extract_components(empty(), 1.0)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- overlap rule


def test_simple_hit_miss_and_fp():
    gt = box(empty(), (5, 5, 5), (9, 9, 9))
    box(gt, (25, 25, 25), (29, 29, 29))         # GT 2: missed
    pred = box(empty(), (6, 6, 6), (8, 8, 8))   # hits GT 1
    box(pred, (15, 30, 5), (17, 32, 7))         # FP far from everything
    m = match_by_overlap(*tables(pred, gt))
    assert m.detected().tolist() == [True, False]
    assert m.pred_is_fp.tolist() == [False, True]
    assert m.n_fp() == 1


def test_split_counts_lesion_once_and_never_as_fp():
    gt = box(empty(), (10, 10, 10), (20, 20, 20))
    pred = box(empty(), (10, 10, 10), (14, 20, 20))
    box(pred, (16, 10, 10), (20, 20, 20))       # two disjoint pieces on one lesion
    p, g = tables(pred, gt)
    m = match_by_overlap(p, g, scores=np.array([0.9, 0.4]))
    assert m.n_gt == 1 and m.n_pred == 2
    assert m.detected().sum() == 1             # one TP, not two
    assert m.n_fp() == 0
    assert m.n_split_lesions() == 1
    assert m.n_redundant() == 1                # the lower-scoring piece
    assert m.gt_score[0] == pytest.approx(0.9)  # lesion score = max over its hits
    # Raising tau above the weaker piece removes the split but keeps the detection.
    assert m.n_split_lesions(0.5) == 0 and m.n_redundant(0.5) == 0
    assert m.detected(0.5).tolist() == [True]


def test_merge_detects_every_covered_lesion_with_no_fp():
    gt = box(empty(), (10, 10, 10), (13, 13, 13))
    box(gt, (10, 10, 16), (13, 13, 19))         # two lesions, 3 voxels apart
    pred = box(empty(), (10, 10, 10), (13, 13, 19))  # one blob over both
    m = match_by_overlap(*tables(pred, gt))
    assert m.n_pred == 1 and m.n_gt == 2
    assert m.detected().tolist() == [True, True]
    assert m.n_fp() == 0
    assert m.n_merge_predictions() == 1
    assert m.n_redundant() == 0


def test_tiny_fragment_is_hit_by_default_and_reported():
    gt = box(empty(), (10, 10, 10), (20, 20, 20))  # 1000 voxels
    pred = empty()
    pred[19, 19, 19] = 1                            # one-voxel touch
    m = match_by_overlap(*tables(pred, gt))
    assert m.detected().tolist() == [True]
    assert m.gt_overlap_vox[0] == 1
    assert m.gt_overlap_frac[0] == pytest.approx(0.001)


def test_strict_rule_fragment_neither_detects_nor_counts_as_fp():
    gt = box(empty(), (10, 10, 10), (20, 20, 20))
    pred = empty()
    pred[19, 19, 19] = 1
    m = match_by_overlap(*tables(pred, gt), min_overlap_vox=3, min_overlap_frac=0.10)
    assert m.detected().tolist() == [False]
    assert m.n_fp() == 0
    assert m.pred_is_fragment.tolist() == [True]


def test_strict_voxel_criterion_capped_at_lesion_size():
    gt = empty()
    gt[10, 10, 10] = 1                              # 1-voxel lesion
    pred = box(empty(), (9, 9, 9), (12, 12, 12))
    m = match_by_overlap(*tables(pred, gt), min_overlap_vox=3)
    assert m.detected().tolist() == [True]


def test_score_threshold_removes_fp_and_detection_independently():
    gt = box(empty(), (5, 5, 5), (9, 9, 9))
    pred = box(empty(), (5, 5, 5), (9, 9, 9))
    box(pred, (30, 30, 30), (32, 32, 32))
    m = match_by_overlap(*tables(pred, gt), scores=np.array([0.3, 0.8]))
    assert (m.detected(0.5).sum(), m.n_fp(0.5)) == (0, 1)
    assert (m.detected(0.2).sum(), m.n_fp(0.2)) == (1, 1)
    assert (m.detected(0.9).sum(), m.n_fp(0.9)) == (0, 0)


def test_scores_validated():
    gt = box(empty(), (5, 5, 5), (9, 9, 9))
    p, g = tables(gt.copy(), gt)
    with pytest.raises(ValueError):
        match_by_overlap(p, g, scores=np.array([0.1, 0.2]))
    with pytest.raises(ValueError):
        match_by_overlap(p, g, scores=np.array([np.nan]))


# --------------------------------------------------------------------------- empty cases


def test_empty_gt_all_predictions_are_fp():
    pred = box(empty(), (5, 5, 5), (7, 7, 7))
    box(pred, (20, 20, 20), (22, 22, 22))
    for m in (match_by_overlap(*tables(pred, empty())),
              match_by_centroid(*tables(pred, empty()), max_distance_mm=5)):
        assert m.n_gt == 0
        assert m.n_fp() == 2
        assert m.detected().shape == (0,)


def test_empty_prediction_misses_everything():
    gt = box(empty(), (5, 5, 5), (7, 7, 7))
    for m in (match_by_overlap(*tables(empty(), gt)),
              match_by_centroid(*tables(empty(), gt), max_distance_mm=5)):
        assert m.detected().tolist() == [False]
        assert m.n_fp() == 0
        assert np.isneginf(m.gt_score[0])


def test_both_empty():
    m = match_by_overlap(*tables(empty(), empty()))
    assert (m.n_gt, m.n_pred, m.n_fp()) == (0, 0, 0)


# --------------------------------------------------------------------------- centroid rule


def test_centroid_rule_respects_anisotropic_spacing():
    gt = box(empty(), (10, 10, 10), (13, 13, 13))    # centroid voxel (11, 11, 11)
    pred = box(empty(), (12, 10, 10), (15, 13, 13))  # centroid voxel (13, 11, 11): 2 voxels in z
    iso = match_by_centroid(*tables(pred, gt, ISO), max_distance_mm=3)
    assert iso.detected().tolist() == [True]
    aniso = match_by_centroid(*tables(pred, gt, (3.0, 1.0, 1.0)), max_distance_mm=5)
    assert aniso.detected().tolist() == [False]       # 2 voxels x 3 mm = 6 mm > 5 mm
    assert aniso.n_fp() == 1                          # overlaps, but the centroid is too far


def test_centroid_rule_merge_can_miss_both_lesions():
    gt = box(empty(), (10, 10, 2), (13, 13, 5))
    box(gt, (10, 10, 30), (13, 13, 33))
    pred = box(empty(), (10, 10, 2), (13, 13, 33))    # one long blob over both
    m = match_by_centroid(*tables(pred, gt), max_distance_mm=5)
    assert m.detected().tolist() == [False, False]    # documented weakness of the rule
    assert m.n_fp() == 1
    # The overlap rule credits both.
    assert match_by_overlap(*tables(pred, gt)).detected().tolist() == [True, True]


def test_centroid_split_counts_once():
    gt = box(empty(), (10, 10, 10), (16, 16, 16))
    pred = box(empty(), (10, 10, 10), (16, 16, 12))
    box(pred, (10, 10, 14), (16, 16, 16))
    m = match_by_centroid(*tables(pred, gt), max_distance_mm=5)
    assert m.detected().sum() == 1 and m.n_fp() == 0 and m.n_split_lesions() == 1


# --------------------------------------------------------------------------- misc


def test_bbox_iou_is_supplementary_column():
    gt = box(empty(), (10, 10, 10), (14, 14, 14))
    pred = box(empty(), (12, 10, 10), (16, 14, 14))
    m = match_by_overlap(*tables(pred, gt))
    assert m.gt_best_bbox_iou[0] == pytest.approx(32 / 96)


def test_dispatch_and_unknown_rule():
    gt = box(empty(), (5, 5, 5), (9, 9, 9))
    p, g = tables(gt.copy(), gt)
    assert match(p, g, rule="centroid", max_distance_mm=3).rule == "centroid"
    with pytest.raises(ValueError, match="unknown matching rule"):
        match(p, g, rule="bbox_iou")


def test_shape_mismatch_rejected():
    p = extract_components(np.zeros((4, 4, 4)), ISO)
    g = extract_components(np.zeros((4, 4, 5)), ISO)
    with pytest.raises(ValueError, match="shape mismatch"):
        match_by_overlap(p, g)
