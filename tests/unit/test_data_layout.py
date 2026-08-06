"""Tests for dataset discovery, QC and loading.

These cover the failure modes that made v1.40 hard to deploy at a new site:
folder-name allowlists that silently matched nothing, four inconsistent
sequence-naming conventions, and cohort scans that returned zero cases without
raising.
"""

from __future__ import annotations

import numpy as np
import pytest

nib = pytest.importorskip("nibabel", reason="nibabel is required for the data layer")

from jannus.core.config import DataConfig
from jannus.core.errors import DataIntegrityError, DataLayoutError
from jannus.data.layout import ResolvedCase, UnresolvedCase, resolve_case, scan_dataset
from jannus.data.loading import load_case, load_mask, save_mask, zscore
from jannus.data.validation import Severity, validate_dataset

SEQUENCES = ("t1_pre", "t1_gd", "flair", "t2")


# 40^3 keeps synthetic volumes above PLAUSIBLE_SHAPE_RANGE's 32-voxel floor, so
# fixtures do not trip the "this is not a brain MRI" check that QC correctly
# raises on anything smaller.
def write_volume(path, shape=(40, 40, 40), spacing=(1.0, 1.0, 1.0), fill=None, seed=0):
    """Write a synthetic NIfTI. No patient data is involved anywhere in tests."""
    rng = np.random.default_rng(seed)
    data = rng.random(shape).astype(np.float32) * 100 if fill is None else np.full(shape, fill, np.float32)
    affine = np.diag([*spacing, 1.0])
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


def make_case(root, case_id, names=("t1_pre", "t1_gd", "flair", "t2"), **kwargs):
    directory = root / case_id
    for i, name in enumerate(names):
        write_volume(directory / f"{name}.nii.gz", seed=i, **kwargs)
    return directory


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

class TestResolveCase:
    def test_resolves_standard_names(self, tmp_path):
        directory = make_case(tmp_path, "PT_001")
        resolved = resolve_case(directory, DataConfig())

        assert isinstance(resolved, ResolvedCase)
        assert set(resolved.channels) == set(SEQUENCES)

    def test_resolves_bravo_as_the_t2_channel(self, tmp_path):
        # BrainMetShare uses the GE BRAVO sequence in place of T2. In v1.40 the
        # API rejected this while the CLI required it.
        directory = make_case(tmp_path, "BMS_1", names=("t1_pre", "t1_gd", "flair", "bravo"))
        resolved = resolve_case(directory, DataConfig())

        assert isinstance(resolved, ResolvedCase)
        assert resolved.matched_aliases["t2"] == "bravo.nii.gz"

    def test_resolves_tcia_style_names(self, tmp_path):
        directory = make_case(tmp_path, "TCIA_1", names=("T1w", "T1c", "FLAIR", "T2w"))
        assert isinstance(resolve_case(directory, DataConfig()), ResolvedCase)

    def test_reports_which_channel_is_missing(self, tmp_path):
        directory = make_case(tmp_path, "PT_002", names=("t1_pre", "t1_gd", "flair"))
        resolved = resolve_case(directory, DataConfig())

        assert isinstance(resolved, UnresolvedCase)
        assert resolved.missing == ["t2"]

    def test_detects_ground_truth(self, tmp_path):
        directory = make_case(tmp_path, "PT_003")
        write_volume(directory / "seg.nii.gz", fill=0)
        assert resolve_case(directory, DataConfig()).ground_truth is not None

    def test_case_id_is_never_exposed_raw_in_the_token(self, tmp_path):
        directory = make_case(tmp_path, "MRN12345678")
        assert "MRN12345678" not in resolve_case(directory, DataConfig()).token


class TestScanDataset:
    def test_finds_cases_regardless_of_folder_naming(self, tmp_path):
        # The v1.40 regression: a prefix allowlist meant sites using their own
        # naming loaded zero cases and were told nothing.
        for case_id in ("PT_001", "MGH_0042", "site-b-007", "Mets_011"):
            make_case(tmp_path, case_id)

        index = scan_dataset(tmp_path, DataConfig())
        assert len(index) == 4

    def test_accepts_a_single_case_directory(self, tmp_path):
        directory = make_case(tmp_path, "PT_001")
        index = scan_dataset(directory, DataConfig())
        assert len(index) == 1

    def test_raises_rather_than_returning_an_empty_cohort(self, tmp_path):
        (tmp_path / "not_a_case").mkdir()
        with pytest.raises(DataLayoutError, match="No usable cases"):
            scan_dataset(tmp_path, DataConfig())

    def test_error_names_the_missing_sequences(self, tmp_path):
        make_case(tmp_path, "PT_001", names=("t1_pre", "t1_gd"))
        with pytest.raises(DataLayoutError) as excinfo:
            scan_dataset(tmp_path, DataConfig())
        assert "flair" in str(excinfo.value)

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(DataLayoutError, match="does not exist"):
            scan_dataset(tmp_path / "absent", DataConfig())

    def test_partial_cohort_separates_resolved_from_unresolved(self, tmp_path):
        make_case(tmp_path, "good_1")
        make_case(tmp_path, "bad_1", names=("t1_pre",))

        index = scan_dataset(tmp_path, DataConfig())
        assert len(index.cases) == 1
        assert len(index.unresolved) == 1

    def test_case_pattern_excludes_directories(self, tmp_path):
        make_case(tmp_path, "KEEP_1")
        make_case(tmp_path, "DROP_1")

        index = scan_dataset(tmp_path, DataConfig(case_pattern=r"^KEEP_"))
        assert index.case_ids == ["KEEP_1"]
        assert index.excluded == ["DROP_1"]

    def test_require_ground_truth_moves_unannotated_cases_aside(self, tmp_path):
        make_case(tmp_path, "with_gt")
        write_volume(tmp_path / "with_gt" / "seg.nii.gz", fill=0)
        make_case(tmp_path, "without_gt")

        index = scan_dataset(tmp_path, DataConfig(), require_ground_truth=True)
        assert index.case_ids == ["with_gt"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class TestLoading:
    def test_stacks_channels_in_canonical_order(self, tmp_path):
        directory = make_case(tmp_path, "PT_001")
        case = resolve_case(directory, DataConfig())
        volume = load_case(case, SEQUENCES)

        assert volume.array.shape == (4, 40, 40, 40)
        assert volume.array.dtype == np.float32

    def test_normalisation_is_per_channel_zscore(self, tmp_path):
        directory = make_case(tmp_path, "PT_001")
        volume = load_case(resolve_case(directory, DataConfig()), SEQUENCES)

        for channel in volume.array:
            assert channel.mean() == pytest.approx(0.0, abs=1e-4)
            assert channel.std() == pytest.approx(1.0, abs=1e-4)

    def test_reads_voxel_spacing_from_the_header(self, tmp_path):
        directory = make_case(tmp_path, "PT_001", spacing=(0.5, 0.5, 3.0))
        volume = load_case(resolve_case(directory, DataConfig()), SEQUENCES)
        assert volume.voxel_spacing == pytest.approx((0.5, 0.5, 3.0))

    def test_mismatched_channel_shapes_are_rejected(self, tmp_path):
        # Silently resampling here would hide a co-registration failure that
        # materially changes the result.
        directory = tmp_path / "PT_001"
        for name in ("t1_pre", "t1_gd", "flair"):
            write_volume(directory / f"{name}.nii.gz", shape=(40, 40, 40))
        write_volume(directory / "t2.nii.gz", shape=(48, 48, 48))

        case = resolve_case(directory, DataConfig())
        with pytest.raises(DataIntegrityError, match="co-registered"):
            load_case(case, SEQUENCES)

    def test_constant_channel_normalises_to_zeros_not_nan(self):
        normalised, stats = zscore(np.full((4, 4, 4), 7.0, dtype=np.float32))
        assert np.isfinite(normalised).all()
        assert (normalised == 0).all()
        assert stats["std"] == 0.0

    def test_mask_round_trip_preserves_geometry(self, tmp_path):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[2:5, 2:5, 2:5] = 1
        affine = np.diag([2.0, 2.0, 2.0, 1.0])

        path = save_mask(mask, affine, tmp_path / "out" / "case_seg.nii.gz")
        reloaded = load_mask(path)

        assert reloaded.shape == mask.shape
        assert int(reloaded.sum()) == int(mask.sum())

    def test_multilabel_ground_truth_collapses_to_binary(self, tmp_path):
        data = np.zeros((8, 8, 8), dtype=np.float32)
        data[0:2, 0:2, 0:2] = 1   # metastasis
        data[4:6, 4:6, 4:6] = 3   # some other structure the site annotates
        path = tmp_path / "seg.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))

        mask = load_mask(path)
        assert set(np.unique(mask)) <= {0, 1}
        assert int(mask.sum()) == 16


# ---------------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------------

class TestValidation:
    def _report(self, root, **kwargs):
        return validate_dataset(scan_dataset(root, DataConfig()), DataConfig(), **kwargs)

    def test_clean_cohort_passes(self, tmp_path):
        make_case(tmp_path, "PT_001")
        make_case(tmp_path, "PT_002")
        report = self._report(tmp_path)

        assert report.passed
        assert report.n_ok == 2

    def test_missing_sequence_is_an_error(self, tmp_path):
        make_case(tmp_path, "good")
        make_case(tmp_path, "bad", names=("t1_pre", "t1_gd"))
        report = self._report(tmp_path)

        assert not report.passed
        assert any(f.code == "MISSING_SEQUENCES" for f in report.errors)

    def test_constant_channel_is_flagged(self, tmp_path):
        directory = tmp_path / "PT_001"
        for name in ("t1_pre", "t1_gd", "flair"):
            write_volume(directory / f"{name}.nii.gz")
        write_volume(directory / "t2.nii.gz", fill=5.0)

        report = self._report(tmp_path)
        assert any(f.code == "CONSTANT_CHANNEL" for f in report.errors)

    def test_unusual_spacing_warns_but_does_not_block(self, tmp_path):
        make_case(tmp_path, "PT_001", spacing=(0.05, 0.05, 0.05))
        report = self._report(tmp_path)

        assert report.passed
        assert any(f.code == "SPACING_OUT_OF_RANGE" for f in report.warnings)

    def test_anisotropy_is_flagged(self, tmp_path):
        make_case(tmp_path, "PT_001", spacing=(1.0, 1.0, 4.5))
        report = self._report(tmp_path)
        assert any(f.code == "HIGHLY_ANISOTROPIC" for f in report.warnings)

    def test_strict_mode_promotes_warnings_to_errors(self, tmp_path):
        make_case(tmp_path, "PT_001", spacing=(1.0, 1.0, 4.5))

        assert self._report(tmp_path).passed
        assert not self._report(tmp_path, strict=True).passed

    def test_implausibly_large_ground_truth_is_flagged(self, tmp_path):
        directory = make_case(tmp_path, "PT_001")
        write_volume(directory / "seg.nii.gz", fill=1.0)  # entire volume labelled
        report = self._report(tmp_path)

        assert any(f.code == "GT_IMPLAUSIBLY_LARGE" for f in report.warnings)

    def test_ground_truth_shape_mismatch_is_an_error(self, tmp_path):
        directory = make_case(tmp_path, "PT_001", shape=(40, 40, 40))
        write_volume(directory / "seg.nii.gz", shape=(48, 48, 48), fill=0)
        report = self._report(tmp_path)

        assert any(f.code == "GT_SHAPE_MISMATCH" for f in report.errors)

    def test_report_contains_no_raw_case_identifiers(self, tmp_path):
        make_case(tmp_path, "MRN98765432", names=("t1_pre",))
        make_case(tmp_path, "PT_001")
        rendered = self._report(tmp_path).render()

        assert "MRN98765432" not in rendered

    def test_report_serialises_to_json(self, tmp_path):
        make_case(tmp_path, "PT_001")
        report = self._report(tmp_path)
        path = report.write(tmp_path / "out" / "validation_report.json")

        import json
        payload = json.loads(path.read_text())
        assert payload["summary"]["resolved"] == 1
        assert payload["summary"]["passed"] is True

    def test_findings_carry_a_severity(self, tmp_path):
        make_case(tmp_path, "PT_001")
        report = self._report(tmp_path)
        for case in report.cases:
            for finding in case.findings:
                assert isinstance(finding.severity, Severity)
