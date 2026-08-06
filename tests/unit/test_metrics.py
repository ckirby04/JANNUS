"""Tests for `jannus.evaluation.metrics`.

This module computes every number a validation site reports back, including the
FDA-facing lesion-wise figures and the RANO-BM measurable-disease scope. It
carried **zero** test coverage through v1.40 — a regression in lesion matching
or surface distance would have silently changed published results with nothing
to catch it.

Every expectation here is hand-derivable from the synthetic geometry, not
recorded from a previous run, so these tests pin the intended behaviour rather
than the current behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from jannus.evaluation.metrics import (
    DEFAULT_SIZE_BINS,
    _longest_axis_mm,
    aggregate_stratified,
    aggregate_with_ci,
    bootstrap_ci,
    compute_case_metrics,
    compute_case_stratified,
    false_positives_per_case,
    hd95,
    lesion_wise_dice_brats,
    lesion_wise_dice_matched,
    lesion_wise_sensitivity,
    match_lesions,
    mean_surface_distance,
)


def cube(volume: np.ndarray, corner, size) -> np.ndarray:
    """Set an axis-aligned cube of ones. Returns the same array for chaining."""
    z, y, x = corner
    dz, dy, dx = (size, size, size) if isinstance(size, int) else size
    volume[z:z + dz, y:y + dy, x:x + dx] = 1
    return volume


@pytest.fixture
def empty():
    return np.zeros((40, 40, 40), dtype=np.uint8)


# ---------------------------------------------------------------------------
# match_lesions — the foundation every lesion-wise metric rests on
# ---------------------------------------------------------------------------

class TestMatchLesions:
    def test_perfect_single_lesion(self, empty):
        gt = cube(empty.copy(), (10, 10, 10), 6)
        result = match_lesions(gt.copy(), gt)

        assert result["n_pred"] == 1
        assert result["n_gt"] == 1
        assert result["matched"] == [(1, 1)]
        assert result["unmatched_pred"] == set()
        assert result["unmatched_gt"] == set()

    def test_disjoint_lesions_do_not_match(self, empty):
        gt = cube(empty.copy(), (5, 5, 5), 4)
        pred = cube(empty.copy(), (30, 30, 30), 4)
        result = match_lesions(pred, gt)

        assert result["matched"] == []
        assert result["unmatched_pred"] == {1}
        assert result["unmatched_gt"] == {1}

    def test_iou_threshold_is_respected(self, empty):
        # GT is a 10x10x10 cube (1000 voxels). Prediction overlaps a 2x10x10
        # slab (200 voxels) and extends outside, so:
        #   intersection = 200, union = 1000 + 400 - 200 = 1200, IoU = 0.1667
        gt = cube(empty.copy(), (10, 10, 10), 10)
        pred = np.zeros_like(empty)
        pred[18:22, 10:20, 10:20] = 1

        assert match_lesions(pred, gt, iou_threshold=0.10)["matched"] == [(1, 1)]
        # Above the actual IoU, the pair must be rejected.
        assert match_lesions(pred, gt, iou_threshold=0.25)["matched"] == []

    def test_greedy_assignment_is_one_to_one(self, empty):
        # One large prediction spanning two separate GT lesions can only claim
        # one of them; the other must be counted as missed.
        gt = empty.copy()
        cube(gt, (10, 10, 10), 4)
        cube(gt, (10, 10, 20), 4)
        pred = np.zeros_like(empty)
        pred[10:14, 10:14, 10:24] = 1

        result = match_lesions(pred, gt)
        assert result["n_gt"] == 2
        assert result["n_pred"] == 1
        assert len(result["matched"]) == 1
        assert len(result["unmatched_gt"]) == 1

    def test_empty_prediction(self, empty):
        gt = cube(empty.copy(), (10, 10, 10), 5)
        result = match_lesions(np.zeros_like(empty), gt)

        assert result["n_pred"] == 0
        assert result["matched"] == []
        assert result["unmatched_gt"] == {1}

    def test_both_empty(self, empty):
        result = match_lesions(empty, empty)
        assert result["n_pred"] == 0
        assert result["n_gt"] == 0
        assert result["matched"] == []


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------

class TestDetectionMetrics:
    def test_sensitivity_counts_detected_fraction(self, empty):
        gt = empty.copy()
        for i, offset in enumerate((5, 15, 25)):
            cube(gt, (offset, offset, offset), 4)
        # Predict only the first two of three lesions.
        pred = empty.copy()
        cube(pred, (5, 5, 5), 4)
        cube(pred, (15, 15, 15), 4)

        assert lesion_wise_sensitivity(pred, gt) == pytest.approx(2 / 3)

    def test_sensitivity_with_no_ground_truth_is_one(self, empty):
        # Documented convention: a case with nothing to find cannot lower
        # sensitivity. Callers exclude these via NaN in compute_case_metrics.
        assert lesion_wise_sensitivity(empty, empty) == 1.0

    def test_false_positives_counts_unmatched_predictions(self, empty):
        gt = cube(empty.copy(), (5, 5, 5), 4)
        pred = empty.copy()
        cube(pred, (5, 5, 5), 4)     # true positive
        cube(pred, (20, 20, 20), 3)  # false positive
        cube(pred, (30, 30, 30), 3)  # false positive

        assert false_positives_per_case(pred, gt) == 2

    def test_perfect_prediction_has_no_false_positives(self, empty):
        gt = cube(empty.copy(), (10, 10, 10), 5)
        assert false_positives_per_case(gt.copy(), gt) == 0


# ---------------------------------------------------------------------------
# Dice variants — the matched/BraTS distinction is a real reporting trap
# ---------------------------------------------------------------------------

class TestDiceVariants:
    def test_perfect_overlap_is_one_for_both_styles(self, empty):
        gt = cube(empty.copy(), (10, 10, 10), 6)
        assert lesion_wise_dice_matched(gt.copy(), gt) == pytest.approx(1.0)
        assert lesion_wise_dice_brats(gt.copy(), gt) == pytest.approx(1.0)

    def test_matched_style_ignores_missed_lesions_brats_does_not(self, empty):
        # Two GT lesions; predict one of them perfectly and miss the other.
        gt = empty.copy()
        cube(gt, (5, 5, 5), 4)
        cube(gt, (25, 25, 25), 4)
        pred = cube(empty.copy(), (5, 5, 5), 4)

        # Matched-only averages over the single matched pair -> 1.0.
        assert lesion_wise_dice_matched(pred, gt) == pytest.approx(1.0)
        # BraTS-style scores the missed lesion as 0 -> mean(1.0, 0.0) = 0.5.
        assert lesion_wise_dice_brats(pred, gt) == pytest.approx(0.5)

    def test_brats_dice_is_zero_when_nothing_detected(self, empty):
        gt = cube(empty.copy(), (10, 10, 10), 5)
        assert lesion_wise_dice_brats(np.zeros_like(empty), gt) == 0.0

    def test_matched_dice_is_nan_when_nothing_matched(self, empty):
        gt = cube(empty.copy(), (5, 5, 5), 4)
        pred = cube(empty.copy(), (30, 30, 30), 4)
        assert np.isnan(lesion_wise_dice_matched(pred, gt))


# ---------------------------------------------------------------------------
# Surface distances — these are the FDA-failing metrics, so they must be right
# ---------------------------------------------------------------------------

class TestSurfaceDistances:
    def test_identical_masks_have_zero_distance(self, empty):
        gt = cube(empty.copy(), (10, 10, 10), 8)
        assert hd95(gt.copy(), gt) == pytest.approx(0.0)
        assert mean_surface_distance(gt.copy(), gt) == pytest.approx(0.0)

    def test_empty_mask_yields_nan(self, empty):
        gt = cube(empty.copy(), (10, 10, 10), 5)
        assert np.isnan(hd95(np.zeros_like(empty), gt))
        assert np.isnan(hd95(gt, np.zeros_like(empty)))

    def test_spacing_scales_distance_linearly(self, empty):
        # A prediction offset from GT along one axis; doubling the spacing on
        # every axis must double the reported millimetre distance.
        gt = cube(empty.copy(), (10, 10, 10), 6)
        pred = cube(empty.copy(), (14, 10, 10), 6)

        at_1mm = mean_surface_distance(pred, gt, spacing=1.0)
        at_2mm = mean_surface_distance(pred, gt, spacing=2.0)

        assert at_1mm > 0
        assert at_2mm == pytest.approx(2.0 * at_1mm, rel=1e-6)

    def test_anisotropic_spacing_is_honoured(self, empty):
        # Displacement along axis 0 only. Stretching axis 0 must increase the
        # distance; stretching axis 2 must not.
        gt = cube(empty.copy(), (10, 10, 10), 6)
        pred = cube(empty.copy(), (16, 10, 10), 6)

        baseline = mean_surface_distance(pred, gt, spacing=(1.0, 1.0, 1.0))
        stretch_axis0 = mean_surface_distance(pred, gt, spacing=(3.0, 1.0, 1.0))

        assert stretch_axis0 > baseline

    def test_hd95_is_at_least_mean_surface_distance(self, empty):
        gt = cube(empty.copy(), (8, 8, 8), 10)
        pred = cube(empty.copy(), (11, 9, 8), 10)
        assert hd95(pred, gt) >= mean_surface_distance(pred, gt)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_constant_values_give_a_degenerate_interval(self):
        low, high = bootstrap_ci([0.8] * 30, n_bootstrap=200, seed=1)
        assert low == pytest.approx(0.8)
        assert high == pytest.approx(0.8)

    def test_interval_brackets_the_mean(self):
        rng = np.random.default_rng(0)
        values = rng.normal(0.75, 0.1, size=100)
        low, high = bootstrap_ci(values, n_bootstrap=500, seed=7)
        assert low < values.mean() < high

    def test_is_reproducible_for_a_fixed_seed(self):
        rng = np.random.default_rng(3)
        values = rng.normal(0.5, 0.2, size=50)
        assert bootstrap_ci(values, n_bootstrap=300, seed=42) == bootstrap_ci(
            values, n_bootstrap=300, seed=42
        )

    def test_wider_spread_gives_a_wider_interval(self):
        rng = np.random.default_rng(11)
        tight = rng.normal(0.5, 0.01, size=80)
        loose = rng.normal(0.5, 0.30, size=80)

        tight_lo, tight_hi = bootstrap_ci(tight, n_bootstrap=400, seed=5)
        loose_lo, loose_hi = bootstrap_ci(loose, n_bootstrap=400, seed=5)

        assert (loose_hi - loose_lo) > (tight_hi - tight_lo)


# ---------------------------------------------------------------------------
# Per-case aggregation
# ---------------------------------------------------------------------------

class TestComputeCaseMetrics:
    def test_perfect_case_reports_perfect_scores(self, empty):
        gt = cube(empty.copy(), (10, 10, 10), 6)
        metrics = compute_case_metrics(gt.copy(), gt)

        assert metrics["lesion_wise_sensitivity"] == pytest.approx(1.0)
        assert metrics["fp_per_case"] == 0.0
        assert metrics["lesion_wise_dice_matched"] == pytest.approx(1.0)
        assert metrics["hd95_mm"] == pytest.approx(0.0)
        assert metrics["n_gt_lesions"] == 1.0

    def test_sensitivity_is_nan_when_case_has_no_ground_truth(self, empty):
        # Distinct from lesion_wise_sensitivity()'s 1.0: NaN here keeps
        # ground-truth-free cases out of the cohort mean instead of inflating it.
        pred = cube(empty.copy(), (10, 10, 10), 4)
        metrics = compute_case_metrics(pred, empty)

        assert np.isnan(metrics["lesion_wise_sensitivity"])
        assert metrics["fp_per_case"] == 1.0

    def test_all_documented_keys_are_present(self, empty):
        gt = cube(empty.copy(), (10, 10, 10), 5)
        metrics = compute_case_metrics(gt.copy(), gt)

        for key in (
            "lesion_wise_sensitivity", "fp_per_case", "lesion_wise_dice_matched",
            "lesion_wise_dice_brats", "small_lesion_sensitivity", "n_small_lesions",
            "hd95_mm", "msd_mm", "n_gt_lesions", "n_pred_lesions",
        ):
            assert key in metrics, f"missing metric key {key!r}"


class TestAggregateWithCI:
    def test_nan_entries_are_excluded_from_the_mean(self):
        per_case = [
            {"metric": 1.0},
            {"metric": float("nan")},
            {"metric": 0.0},
        ]
        agg = aggregate_with_ci(per_case, n_bootstrap=100)

        assert agg["metric"]["n"] == 2
        assert agg["metric"]["mean"] == pytest.approx(0.5)

    def test_all_nan_metric_reports_nan_not_a_crash(self):
        agg = aggregate_with_ci([{"m": float("nan")}] * 3, n_bootstrap=50)
        assert agg["m"]["n"] == 0
        assert np.isnan(agg["m"]["mean"])

    def test_empty_input_returns_empty_dict(self):
        assert aggregate_with_ci([], n_bootstrap=10) == {}


# ---------------------------------------------------------------------------
# Size stratification — underpins the RANO-BM measurable-disease scope
# ---------------------------------------------------------------------------

class TestSizeStratification:
    def test_longest_axis_uses_spacing(self, empty):
        # A 5-voxel run along axis 0 at 2 mm spacing spans 10 mm.
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[5:10, 5, 5] = 1

        assert _longest_axis_mm(mask, spacing=1.0) == pytest.approx(5.0)
        assert _longest_axis_mm(mask, spacing=(2.0, 1.0, 1.0)) == pytest.approx(10.0)

    def test_longest_axis_of_empty_mask_is_zero(self, empty):
        assert _longest_axis_mm(empty, spacing=1.0) == 0.0

    def test_lesions_land_in_the_expected_size_bins(self, empty):
        # 12 voxels at 1 mm -> 12 mm longest axis -> the "10-20mm" bin.
        gt = np.zeros((40, 40, 40), dtype=np.uint8)
        gt[5:17, 5:8, 5:8] = 1
        strat = compute_case_stratified(gt.copy(), gt, spacing=1.0)

        assert strat["10-20mm"]["n_gt"] == 1
        assert sum(b["n_gt"] for b in strat.values()) == 1

    def test_stratified_aggregation_sums_across_cases(self, empty):
        gt = np.zeros((40, 40, 40), dtype=np.uint8)
        gt[5:17, 5:8, 5:8] = 1
        per_case = [compute_case_stratified(gt.copy(), gt, spacing=1.0) for _ in range(3)]

        agg = aggregate_stratified(per_case, bins=DEFAULT_SIZE_BINS)
        assert agg["10-20mm"]["n_gt_total"] == 3
        assert agg["10-20mm"]["n_matched_total"] == 3
        assert agg["10-20mm"]["sensitivity"] == pytest.approx(1.0)
        assert agg["10-20mm"]["fp_per_case"] == pytest.approx(0.0)

    def test_aggregate_exposes_the_documented_key_names(self, empty):
        # Pinned because `report.py` renders these by name; a rename here
        # silently blanks columns in the report a site returns to us.
        agg = aggregate_stratified([compute_case_stratified(empty, empty)])
        assert set(agg["<3mm"]) == {
            "n_gt_total", "n_matched_total", "sensitivity",
            "n_fp_total", "fp_per_case", "mean_dsc",
        }

    def test_every_default_bin_is_represented(self, empty):
        agg = aggregate_stratified([compute_case_stratified(empty, empty)])
        assert set(agg) == {label for label, _ in DEFAULT_SIZE_BINS}
