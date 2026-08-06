"""
Unit tests for the resume branching in scripts/training/train_swin_unetr.py.

We don't actually run any training here — run_phase is monkey-patched to
record its arguments, the dataset loader is stubbed, and we simulate
various LATEST_PATH checkpoint contents to verify the script:

  1. Runs both phases fresh when no checkpoint exists.
  2. Skips phase 1 entirely and resumes phase 2 at the correct epoch
     when LATEST_PATH has phase=2.
  3. Resumes phase 1 at the correct epoch when LATEST_PATH has phase=1,
     then proceeds to phase 2 fresh.
  4. Correctly handles an already-completed phase (epoch == max).
"""

from __future__ import annotations

import pytest

# Requires the optional `segmentation` extra. This guard must precede every other
# import: a bare `import torch` at module scope fails collection outright on a
# core install (`pip install -e ".[dev]"`), which is what CI verifies.
pytest.importorskip("torch", reason="install jannus[segmentation]")


import sys
from pathlib import Path
from unittest.mock import patch

import torch
from torch.amp import GradScaler

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

# Every test here builds a full SwinUNETR (feature_size 48, 96^3 ROI) and runs
# a backward pass on CPU, which costs minutes per test. Marked `slow` so CI can
# run `-m "not slow"` on pull requests and the full suite on a schedule.
pytestmark = pytest.mark.slow

# Must import the module (not `from ... import main`) so we can patch
# symbols inside it.
import importlib.util

spec = importlib.util.spec_from_file_location(
    "train_swin_unetr",
    PROJECT / "scripts" / "training" / "train_swin_unetr.py",
)
train_swin_unetr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_swin_unetr)

@pytest.fixture
def patched_environment(tmp_path, monkeypatch):
    """Replace the heavy parts of the training script with fakes so we
    can exercise main() without touching real data or GPUs."""
    calls: list[dict] = []

    # Accept **kwargs so this double does not have to track every optional
    # parameter the real `run_phase` grows. v1.40's version pinned the exact
    # signature and broke the moment `early_stop_patience` was added — the
    # tests then failed for a reason unrelated to the resume logic they exist
    # to cover.
    def fake_run_phase(model, optimizer, train_loader, val_dataset,
                       val_case_ids, epochs, phase, device, scaler,
                       best_val_dice, val_every, val_subset,
                       start_epoch=1, **kwargs):
        calls.append({
            "phase": phase,
            "start_epoch": start_epoch,
            "target_epochs": epochs,
            "best_in": best_val_dice,
        })
        return best_val_dice, epochs

    class FakeDataset:
        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return 1

    monkeypatch.setattr(train_swin_unetr, "run_phase", fake_run_phase)
    monkeypatch.setattr(train_swin_unetr, "NnUNetRawDataset", FakeDataset)
    monkeypatch.setattr(
        train_swin_unetr, "load_nnunet_split",
        lambda p, fold=0: (["case1"], ["case_val1"]),
    )

    # Use a tmp dir for checkpoints so we don't clobber real artifacts.
    monkeypatch.setattr(
        train_swin_unetr, "MODEL_OUT_DIR", tmp_path,
    )
    monkeypatch.setattr(
        train_swin_unetr, "MODEL_OUT_PATH",
        tmp_path / "swin_unetr_brainmets.pth",
    )
    monkeypatch.setattr(
        train_swin_unetr, "LATEST_PATH",
        tmp_path / "swin_unetr_latest.pth",
    )
    monkeypatch.setattr(
        train_swin_unetr, "STATE_PATH",
        tmp_path / "swin_unetr_training_state.json",
    )

    return calls, tmp_path

def _run_main(argv_extra: list[str]):
    argv = ["train_swin_unetr.py", *argv_extra]
    with patch("sys.argv", argv):
        train_swin_unetr.main()

def _stage_latest(path: Path, phase: int, epoch: int, best_val_dice: float):
    """Write a minimal LATEST checkpoint with just enough content for
    main()'s resume branch to pick it up. We don't need a real model —
    the resume path calls model.load_state_dict(strict=False) so empty
    state_dicts work."""
    model = train_swin_unetr.build_swin_unetr_from_dict(
        {"in_channels": 4, "out_channels": 2, "feature_size": 48,
         "img_size": [96, 96, 96]}
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # Give optimizer some state so load_state_dict has something non-empty.
    x = torch.randn(1, 4, 96, 96, 96)
    model(x).sum().backward()
    opt.step()
    scaler = GradScaler(device="cpu", enabled=False)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "phase": phase,
        "epoch": epoch,
        "best_val_dice": best_val_dice,
    }, path)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fresh_run_runs_both_phases(patched_environment):
    """No LATEST_PATH + no --resume: phase 1 from epoch 1, phase 2 from epoch 1."""
    calls, _ = patched_environment
    _run_main(["--epochs-phase1", "3", "--epochs-phase2", "5"])

    assert len(calls) == 2
    assert calls[0] == {"phase": 1, "start_epoch": 1, "target_epochs": 3, "best_in": 0.0}
    assert calls[1] == {"phase": 2, "start_epoch": 1, "target_epochs": 5, "best_in": 0.0}

def test_resume_mid_phase1(patched_environment):
    """LATEST has phase=1 epoch=2 best=0.3: phase 1 resumes at epoch 3,
    phase 2 starts fresh at epoch 1 with best carried through."""
    calls, tmp = patched_environment
    _stage_latest(tmp / "swin_unetr_latest.pth", phase=1, epoch=2, best_val_dice=0.3)

    _run_main(["--resume", "--epochs-phase1", "5", "--epochs-phase2", "10"])

    assert len(calls) == 2
    assert calls[0]["phase"] == 1
    assert calls[0]["start_epoch"] == 3
    assert calls[0]["target_epochs"] == 5
    assert calls[0]["best_in"] == 0.3
    assert calls[1]["phase"] == 2
    assert calls[1]["start_epoch"] == 1
    assert calls[1]["best_in"] == 0.3

def test_resume_mid_phase2(patched_environment):
    """LATEST has phase=2 epoch=47 best=0.5: phase 1 skipped entirely,
    phase 2 resumes at epoch 48."""
    calls, tmp = patched_environment
    _stage_latest(tmp / "swin_unetr_latest.pth", phase=2, epoch=47, best_val_dice=0.5)

    _run_main(["--resume", "--epochs-phase1", "20", "--epochs-phase2", "500"])

    assert len(calls) == 1, "phase 1 should have been skipped entirely"
    assert calls[0]["phase"] == 2
    assert calls[0]["start_epoch"] == 48
    assert calls[0]["target_epochs"] == 500
    assert calls[0]["best_in"] == 0.5

def test_resume_phase1_already_done(patched_environment):
    """LATEST phase=1 epoch=20 (all 20 epochs done). Script should skip
    phase 1 ('already finished') and run phase 2 from epoch 1."""
    calls, tmp = patched_environment
    _stage_latest(tmp / "swin_unetr_latest.pth", phase=1, epoch=20, best_val_dice=0.27)

    _run_main(["--resume", "--epochs-phase1", "20", "--epochs-phase2", "100"])

    # Only phase 2 should have been invoked — phase 1 start_epoch=21 > 20
    # triggers the 'already finished, skipping' branch in main().
    assert len(calls) == 1
    assert calls[0]["phase"] == 2
    assert calls[0]["start_epoch"] == 1
    assert calls[0]["best_in"] == 0.27

def test_resume_phase2_extend_epochs(patched_environment):
    """User resumes a finished phase-2 run with a larger --epochs-phase2
    to continue training. Should resume at epoch+1 of the new target."""
    calls, tmp = patched_environment
    _stage_latest(tmp / "swin_unetr_latest.pth", phase=2, epoch=150, best_val_dice=0.68)

    _run_main(["--resume", "--phase", "2",
               "--epochs-phase1", "0", "--epochs-phase2", "300"])

    assert len(calls) == 1
    assert calls[0]["phase"] == 2
    assert calls[0]["start_epoch"] == 151
    assert calls[0]["target_epochs"] == 300
    assert calls[0]["best_in"] == 0.68

def test_explicit_phase_2_without_resume_runs_fresh(patched_environment):
    """--phase 2 without --resume: phase 2 from epoch 1, phase 1 skipped."""
    calls, _ = patched_environment
    _run_main(["--phase", "2", "--epochs-phase2", "10"])

    assert len(calls) == 1
    assert calls[0]["phase"] == 2
    assert calls[0]["start_epoch"] == 1
    assert calls[0]["best_in"] == 0.0
