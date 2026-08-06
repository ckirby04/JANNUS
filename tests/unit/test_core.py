"""Tests for the v1.50 core runtime: config, checksums, logging, determinism."""

from __future__ import annotations

import json

import pytest
import yaml

from jannus.core.checksums import (
    WeightEntry,
    load_manifest,
    sha256_file,
    verify_entry,
    verify_manifest,
    write_manifest,
)
from jannus.core.config import (
    CANONICAL_MODEL_ORDER,
    PUBLISHED_THRESHOLD,
    audit_config_keys,
    load_config,
)
from jannus.core.determinism import configure_determinism
from jannus.core.errors import ConfigError, WeightsError
from jannus.core.logging import pseudonymise, redact
from jannus.core.provenance import RunProvenance


def minimal_config(**overrides) -> dict:
    cfg = {
        "models": [
            {"name": name, "architecture": name, "path": f"model/{name}.pth"}
            for name in CANONICAL_MODEL_ORDER
        ],
        "stacking": {"classifier_path": "model/stacker.pth", "in_channels": 9},
        "inference": {"threshold": 0.55, "postprocessing": {"min_size": 0}},
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture
def config_file(tmp_path):
    def _write(cfg: dict):
        configs = tmp_path / "configs"
        configs.mkdir(exist_ok=True)
        path = configs / "models.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return path
    return _write


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_the_operating_threshold(self, config_file):
        config = load_config(config_file(minimal_config()))
        assert config.inference.threshold == 0.55

    def test_defaults_to_the_published_threshold_when_absent(self, config_file):
        # Regression guard for the v1.40 defect: the key was missing entirely
        # and the pipeline silently fell back to 0.5, while every published
        # metric was produced at 0.55.
        cfg = minimal_config()
        del cfg["inference"]["threshold"]
        config = load_config(config_file(cfg))
        assert config.inference.threshold == PUBLISHED_THRESHOLD == 0.55

    def test_models_are_reordered_canonically(self, config_file):
        cfg = minimal_config()
        cfg["models"].reverse()
        config = load_config(config_file(cfg))
        # Channel order is a hard contract with the trained stacker; file order
        # must never leak through.
        assert config.model_names == list(CANONICAL_MODEL_ORDER)

    def test_missing_base_model_is_rejected(self, config_file):
        cfg = minimal_config()
        cfg["models"] = [m for m in cfg["models"] if m["name"] != "swin_unetr"]
        with pytest.raises(ConfigError, match="swin_unetr"):
            load_config(config_file(cfg))

    def test_out_of_range_threshold_is_rejected(self, config_file):
        cfg = minimal_config()
        cfg["inference"]["threshold"] = 1.5
        with pytest.raises(ConfigError, match="threshold"):
            load_config(config_file(cfg))

    def test_wrong_stacker_input_channels_rejected(self, config_file):
        cfg = minimal_config()
        cfg["stacking"]["in_channels"] = 7  # forgot variance + range
        with pytest.raises(ConfigError, match="in_channels"):
            load_config(config_file(cfg))

    def test_missing_file_raises_config_error(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "nope.yaml")

    def test_paths_resolve_against_config_not_cwd(self, config_file, tmp_path, monkeypatch):
        # v1.40 resolved base-model paths against the process working
        # directory, so running from anywhere but the repo root broke loading.
        path = config_file(minimal_config())
        monkeypatch.chdir(tmp_path.parent)
        config = load_config(path)
        assert config.checkpoint_paths()["patch_8"] == tmp_path / "model" / "patch_8.pth"
        assert config.checkpoint_paths()["patch_8"].is_absolute()

    def test_overrides_are_applied(self, config_file):
        config = load_config(
            config_file(minimal_config()), overrides={"inference.threshold": 0.42}
        )
        assert config.inference.threshold == 0.42

    def test_sequence_alias_without_a_channel_is_rejected(self, config_file):
        cfg = minimal_config()
        cfg["data"] = {"sequences": ["t1_pre", "dwi"], "sequence_aliases": {"t1_pre": ["t1"]}}
        with pytest.raises(ConfigError, match="dwi"):
            load_config(config_file(cfg))


class TestConfigKeyAudit:
    def test_inert_keys_are_reported(self):
        # This is what stops a site editing a value that does nothing — the
        # exact trap `ensemble.default_threshold` set in v1.40.
        inert = audit_config_keys({"ensemble": {"default_threshold": 0.5}})
        assert "ensemble.default_threshold" in inert

    def test_consumed_keys_are_not_reported(self):
        assert audit_config_keys({"inference": {"threshold": 0.55}}) == []

    def test_model_subkeys_are_exempt(self):
        assert audit_config_keys({"models": [{"name": "x", "path": "y"}]}) == []


# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------

class TestChecksums:
    def test_hash_is_stable_and_content_dependent(self, tmp_path):
        a = tmp_path / "a.bin"
        a.write_bytes(b"weights")
        first = sha256_file(a)

        assert first == sha256_file(a)
        assert len(first) == 64

        a.write_bytes(b"weight5")
        assert sha256_file(a) != first

    def test_verify_entry_accepts_matching_file(self, tmp_path):
        target = tmp_path / "model" / "x.pth"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"abc")

        entry = WeightEntry(
            name="x", path="model/x.pth", sha256=sha256_file(target), bytes=3
        )
        assert verify_entry(entry, tmp_path).ok

    def test_verify_entry_detects_a_missing_file(self, tmp_path):
        entry = WeightEntry(name="x", path="model/x.pth", sha256="0" * 64, bytes=3)
        result = verify_entry(entry, tmp_path)
        assert not result.ok and not result.present
        assert "MISSING" in result.describe()

    def test_verify_entry_detects_a_truncated_file(self, tmp_path):
        # The common real failure: an interrupted download or an LFS pointer.
        target = tmp_path / "x.pth"
        target.write_bytes(b"ab")
        entry = WeightEntry(name="x", path="x.pth", sha256=sha256_file(target), bytes=999)
        result = verify_entry(entry, tmp_path)
        assert not result.ok and not result.size_ok
        assert "SIZE" in result.describe()

    def test_verify_entry_detects_substituted_content(self, tmp_path):
        target = tmp_path / "x.pth"
        target.write_bytes(b"abc")
        entry = WeightEntry(name="x", path="x.pth", sha256="1" * 64, bytes=3)
        result = verify_entry(entry, tmp_path)
        assert not result.ok and result.size_ok and not result.hash_ok
        assert "HASH" in result.describe()

    def test_manifest_round_trip(self, tmp_path):
        target = tmp_path / "x.pth"
        target.write_bytes(b"abc")
        entry = WeightEntry(
            name="x", path="x.pth", sha256=sha256_file(target), bytes=3, role="stacker"
        )
        manifest = write_manifest([entry], tmp_path / "weights.lock.json",
                                  pipeline_revision="test-r1")
        loaded = load_manifest(manifest)

        assert len(loaded) == 1
        assert loaded[0].name == "x"
        assert loaded[0].role == "stacker"

    def test_verify_manifest_raises_on_failure(self, tmp_path):
        manifest = tmp_path / "weights.lock.json"
        manifest.write_text(json.dumps({
            "schema_version": 1,
            "weights": [{"name": "x", "path": "missing.pth", "sha256": "0" * 64, "bytes": 1}],
        }))
        with pytest.raises(WeightsError, match="failed verification"):
            verify_manifest(manifest, tmp_path)

    def test_verify_manifest_can_report_without_raising(self, tmp_path):
        manifest = tmp_path / "weights.lock.json"
        manifest.write_text(json.dumps({
            "schema_version": 1,
            "weights": [{"name": "x", "path": "missing.pth", "sha256": "0" * 64, "bytes": 1}],
        }))
        results = verify_manifest(manifest, tmp_path, raise_on_failure=False)
        assert len(results) == 1 and not results[0].ok

    def test_unsupported_schema_version_is_rejected(self, tmp_path):
        manifest = tmp_path / "weights.lock.json"
        manifest.write_text(json.dumps({"schema_version": 99, "weights": []}))
        with pytest.raises(WeightsError, match="schema version"):
            load_manifest(manifest)


def test_shipped_weights_manifest_is_valid():
    """The manifest committed to this repo must parse and be complete."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    entries = load_manifest(repo_root / "weights.lock.json")
    names = {e.name for e in entries}

    assert names >= set(CANONICAL_MODEL_ORDER), "manifest is missing a base model"
    assert "stacker" in names
    for entry in entries:
        assert len(entry.sha256) == 64, f"{entry.name} has a malformed hash"
        assert entry.bytes > 0


# ---------------------------------------------------------------------------
# PHI-safe logging
# ---------------------------------------------------------------------------

class TestLogRedaction:
    @pytest.mark.parametrize("secret", [
        "MRN 123456789",
        "accession 20240131001",
        "1.2.840.113619.2.55.3.604688.1",
    ])
    def test_identifier_shaped_text_is_redacted(self, secret):
        assert "redacted" in redact(secret)

    def test_api_keys_are_redacted(self):
        cleaned = redact("token sk-ant-api03-AAAAAAAAAAAAAAAAAAAA")
        assert "sk-ant-api03" not in cleaned

    def test_ordinary_text_is_untouched(self):
        message = "loaded 84 cases from cohort"
        assert redact(message) == message

    def test_pseudonyms_are_stable_and_non_reversible(self):
        token = pseudonymise("Mets_042")
        assert token == pseudonymise("Mets_042")
        assert token != pseudonymise("Mets_043")
        assert "Mets_042" not in token
        assert token.startswith("case-")

    def test_salt_changes_the_token(self):
        assert pseudonymise("X", salt="a") != pseudonymise("X", salt="b")


# ---------------------------------------------------------------------------
# Determinism and provenance
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_reports_what_it_actually_set(self):
        report = configure_determinism(seed=123)
        assert report.seed == 123
        assert report.numpy_seeded
        assert "fully_deterministic" in report.to_dict()

    def test_seeding_makes_numpy_reproducible(self):
        import numpy as np

        configure_determinism(seed=7)
        first = np.random.rand(5).tolist()
        configure_determinism(seed=7)
        assert np.random.rand(5).tolist() == first


class TestProvenance:
    def test_records_version_and_revision(self):
        from jannus import PIPELINE_REVISION, __version__

        prov = RunProvenance(command="predict")
        assert prov.jannus_version == __version__
        assert prov.pipeline_revision == PIPELINE_REVISION

    def test_writes_json(self, tmp_path):
        prov = RunProvenance(command="predict")
        prov.n_cases_attempted = 2
        prov.n_cases_succeeded = 1
        prov.record_failure("case-abc123", "unreadable")
        path = prov.finish().write(tmp_path / "provenance.json")

        payload = json.loads(path.read_text())
        assert payload["command"] == "predict"
        assert payload["finished_at"] is not None
        assert payload["failures"] == [{"case": "case-abc123", "reason": "unreadable"}]

    def test_checkpoint_hashes_are_recorded(self, tmp_path):
        ckpt = tmp_path / "m.pth"
        ckpt.write_bytes(b"weights")

        prov = RunProvenance(command="predict")
        prov.record_checkpoints({"stacker": ckpt, "absent": tmp_path / "no.pth"})

        assert prov.checkpoints["stacker"] == sha256_file(ckpt)
        assert prov.checkpoints["absent"] == "<missing>"

    def test_summary_flags_partial_determinism(self):
        prov = RunProvenance(command="predict")
        prov.determinism = {"fully_deterministic": False}
        assert any("PARTIAL" in line for line in prov.summary_lines())
