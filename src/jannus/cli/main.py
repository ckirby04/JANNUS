"""`jannus` command-line interface.

The intended path for an external validation site, in order:

    jannus doctor                          confirm the install is sound
    jannus fetch-weights                   download and verify checkpoints
    jannus validate-data  --input DATA     QC the cohort before spending GPU time
    jannus predict        --input DATA --output PRED
    jannus evaluate       --input DATA --predictions PRED --output RESULTS

Every command exits non-zero with a typed exit code on failure (see
:mod:`jannus.core.errors`), so the whole sequence can be driven from a site's
own batch scheduler.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .._version import PIPELINE_REVISION, __version__
from ..core.errors import JannusError
from ..core.logging import configure_logging

#: Default config location, relative to the repo root.
DEFAULT_CONFIG = "configs/models.yaml"


def _find_default_config() -> Path | None:
    """Locate `configs/models.yaml` for a user who did not pass --config.

    Tries the working directory first (the common case: a site runs from its
    clone), then walks up from this file so an editable install still works.
    Returns None rather than guessing wrong — the caller then requires --config
    explicitly, which is better than silently loading someone else's config.
    """
    cwd_candidate = Path.cwd() / DEFAULT_CONFIG
    if cwd_candidate.is_file():
        return cwd_candidate
    for parent in Path(__file__).resolve().parents:
        candidate = parent / DEFAULT_CONFIG
        if candidate.is_file():
            return candidate

    # Fall back to the copy shipped inside the wheel, so a pip-installed
    # JANNUS works with no source tree present.
    from ..core.paths import config_path

    packaged = config_path("models.yaml")
    return packaged if packaged.is_file() else None


def _add_common(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by every subcommand."""
    parser.add_argument(
        "--config", type=Path, default=None,
        help=f"Path to models.yaml (default: nearest {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log verbosity (default: INFO)",
    )
    parser.add_argument(
        "--log-format", default="console", choices=["console", "json"],
        help="Console log format; the --log-file copy is always JSON",
    )
    parser.add_argument(
        "--log-file", type=Path, default=None,
        help="Additionally write JSON-lines logs here (safe to share: PHI is redacted)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Master RNG seed (default: 20260805)",
    )
    parser.add_argument(
        "--strict-determinism", action="store_true",
        help="Error rather than fall back on any nondeterministic op. "
             "Recommended for a formal validation run.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jannus",
        description=(
            "JANNUS — brain metastasis segmentation for multi-site validation.\n\n"
            "RESEARCH USE ONLY. Not a medical device; not for diagnostic use."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Typical sequence:\n"
            "  jannus doctor\n"
            "  jannus validate-data --input /data/cohort\n"
            "  jannus predict       --input /data/cohort --output /out/pred\n"
            "  jannus evaluate      --input /data/cohort --predictions /out/pred "
            "--output /out/results\n"
        ),
    )
    parser.add_argument(
        "--version", action="version",
        version=f"jannus {__version__} (pipeline {PIPELINE_REVISION})",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # ---- doctor -----------------------------------------------------------
    doctor = subparsers.add_parser(
        "doctor",
        help="Check the installation, config and weights without touching data",
        description="Verify that this machine can run JANNUS correctly.",
    )
    _add_common(doctor)
    doctor.add_argument(
        "--weights-manifest", type=Path, default=None,
        help="Path to weights.lock.json (default: alongside the repo root)",
    )
    doctor.add_argument(
        "--skip-weights", action="store_true",
        help="Skip checkpoint hashing (fast; checks environment and config only)",
    )

    # ---- validate-data ----------------------------------------------------
    validate = subparsers.add_parser(
        "validate-data",
        help="QC a cohort before running inference",
        description=(
            "Check that every case has the required sequences, that they are "
            "co-registered, and that geometry and intensities fall inside the "
            "range JANNUS was trained on. Run this first."
        ),
    )
    _add_common(validate)
    validate.add_argument("--input", type=Path, required=True, help="Dataset root")
    validate.add_argument(
        "--output", type=Path, default=None,
        help="Write the full JSON report here",
    )
    validate.add_argument(
        "--strict", action="store_true",
        help="Treat every warning as an error (recommended for a formal cohort)",
    )
    validate.add_argument(
        "--require-ground-truth", action="store_true",
        help="Fail cases that have no segmentation (needed for `evaluate`)",
    )
    validate.add_argument(
        "--fast", action="store_true",
        help="Structural checks only; skip loading pixel data",
    )

    # ---- predict ----------------------------------------------------------
    predict = subparsers.add_parser(
        "predict",
        help="Run segmentation over a cohort",
        description=(
            "Run the 7-base + StackingClassifierV2 pipeline over every case and "
            "write one binary NIfTI mask per case, plus a provenance manifest."
        ),
    )
    _add_common(predict)
    predict.add_argument("--input", type=Path, required=True, help="Dataset root")
    predict.add_argument("--output", type=Path, required=True, help="Output directory")
    predict.add_argument(
        "--threshold", type=float, default=None,
        help="Override the operating threshold. Changing this invalidates "
             "comparison against published results.",
    )
    predict.add_argument(
        "--device", default=None,
        help="torch device, e.g. cuda, cuda:0, cpu (default: cuda if available)",
    )
    predict.add_argument(
        "--overwrite", action="store_true",
        help="Recompute cases whose output already exists",
    )
    predict.add_argument(
        "--skip-validation", action="store_true",
        help="Do not run dataset QC first (not recommended)",
    )
    predict.add_argument(
        "--allow-stub", action="store_true",
        help="DANGEROUS: permit random-weight stub models. Plumbing tests only — "
             "the resulting masks are meaningless.",
    )
    predict.add_argument(
        "--save-probabilities", action="store_true",
        help="Also write the float probability map per case (large)",
    )

    # ---- evaluate ---------------------------------------------------------
    evaluate = subparsers.add_parser(
        "evaluate",
        help="Score predictions against ground truth",
        description=(
            "Compute voxel and lesion-wise metrics with bootstrap confidence "
            "intervals, stratified by lesion size and scoped to RANO-BM "
            "measurable disease, then render a shareable report."
        ),
    )
    _add_common(evaluate)
    evaluate.add_argument("--input", type=Path, required=True,
                          help="Dataset root (supplies ground truth and spacing)")
    evaluate.add_argument("--predictions", type=Path, required=True,
                          help="Directory of {case_id}_seg.nii.gz masks")
    evaluate.add_argument("--output", type=Path, required=True,
                          help="Directory for results.json and report.md")
    evaluate.add_argument("--site-name", default="(unnamed site)",
                          help="Label for the report header")
    evaluate.add_argument("--bootstrap", type=int, default=1000,
                          help="Bootstrap resamples for CIs (default: 1000)")
    evaluate.add_argument("--iou-threshold", type=float, default=0.1,
                          help="IoU for lesion matching (default: 0.1)")
    evaluate.add_argument("--suffix", default="_seg.nii.gz",
                          help="Prediction filename suffix (default: _seg.nii.gz)")

    # ---- fetch-weights ----------------------------------------------------
    fetch = subparsers.add_parser(
        "fetch-weights",
        help="Download and verify model checkpoints",
        description=(
            "Checkpoints are not distributed in the git repository. This "
            "downloads each one and verifies it against weights.lock.json."
        ),
    )
    _add_common(fetch)
    fetch.add_argument("--dest", type=Path, default=None,
                       help="Destination directory (default: <repo>/model)")
    fetch.add_argument("--weights-manifest", type=Path, default=None,
                       help="Path to weights.lock.json")
    fetch.add_argument("--verify-only", action="store_true",
                       help="Check what is already on disk; download nothing")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    configure_logging(
        level=args.log_level, log_file=args.log_file, fmt=args.log_format
    )

    if getattr(args, "config", None) is None:
        args.config = _find_default_config()

    # Imported lazily so `jannus --help` does not pay for torch.
    from . import commands

    handlers = {
        "doctor": commands.cmd_doctor,
        "validate-data": commands.cmd_validate_data,
        "predict": commands.cmd_predict,
        "evaluate": commands.cmd_evaluate,
        "fetch-weights": commands.cmd_fetch_weights,
    }

    try:
        return handlers[args.command](args)
    except JannusError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        if exc.remedy:
            print(f"       {exc.remedy}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


def _entry() -> None:  # console_scripts target
    sys.exit(main())


if __name__ == "__main__":
    _entry()
