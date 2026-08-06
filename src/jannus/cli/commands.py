"""Implementations of the `jannus` subcommands.

Each handler takes the parsed argparse namespace and returns a process exit
code. Heavy imports (torch, nibabel, the pipeline) happen inside the handlers
so `jannus --help` and `jannus doctor --skip-weights` stay fast and work in a
partially-installed environment.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .._version import __version__
from ..core.checksums import MANIFEST_FILENAME, load_manifest, verify_manifest
from ..core.config import PUBLISHED_THRESHOLD, load_config
from ..core.determinism import DEFAULT_SEED, configure_determinism
from ..core.errors import (
    ConfigError,
    DataLayoutError,
    JannusError,
    PipelineError,
    UnverifiedPipelineError,
    WeightsError,
)
from ..core.logging import get_logger
from ..core.provenance import RunProvenance

logger = get_logger("cli")

RULE = "=" * 72


def _banner(title: str) -> None:
    print(RULE)
    print(f"  JANNUS {__version__}  —  {title}")
    print(RULE)


def _repo_root(args) -> Path:
    """Repo root, inferred from --config or from the installed package."""
    if getattr(args, "config", None):
        return Path(args.config).resolve().parent.parent
    return Path(__file__).resolve().parents[3]


def _require_config(args):
    if getattr(args, "config", None) is None:
        raise ConfigError(
            "Could not locate configs/models.yaml.",
            remedy="Pass --config /path/to/configs/models.yaml explicitly.",
        )
    return load_config(args.config)


def _setup_run(args, command: str) -> RunProvenance:
    """Seed RNGs and open a provenance record."""
    determinism = configure_determinism(
        seed=args.seed if args.seed is not None else DEFAULT_SEED,
        strict=bool(getattr(args, "strict_determinism", False)),
    )
    provenance = RunProvenance(command=command)
    provenance.determinism = determinism.to_dict()
    if not determinism.fully_deterministic:
        for warning in determinism.warnings:
            provenance.warnings.append(f"determinism: {warning}")
    return provenance


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

def cmd_doctor(args) -> int:
    """Check environment, config and weights without touching patient data."""
    _banner("environment check")
    problems: list[str] = []
    warnings: list[str] = []

    print("\n[1/4] Python environment")
    from ..core.provenance import _hardware, _package_versions

    hardware = _hardware()
    print(f"  python           {hardware['python']}")
    print(f"  platform         {hardware['platform']}")
    print(f"  cuda available   {hardware.get('torch_cuda_available', False)}")
    for gpu in hardware.get("gpus", []):
        print(f"    GPU            {gpu}")
    if not hardware.get("torch_cuda_available"):
        warnings.append(
            "No CUDA GPU detected — inference will run on CPU and take roughly "
            "an hour per case instead of a few minutes."
        )

    print("\n[2/4] Required packages")
    for name, ver in _package_versions().items():
        marker = " " if ver != "<not installed>" else "!"
        print(f"  {marker} {name:<16} {ver}")
        if ver == "<not installed>":
            if name in {"torch", "numpy", "scipy", "nibabel"}:
                problems.append(f"{name} is not installed — inference cannot run.")
            elif name == "nnunetv2":
                problems.append(
                    "nnunetv2 is not installed — 2 of the 7 base models "
                    "(nnunet_3d, nnunet_2d) cannot load."
                )
            else:
                warnings.append(f"{name} is not installed.")

    print("\n[3/4] Configuration")
    config = None
    try:
        config = _require_config(args)
        print(f"  config           {config.source_path}")
        print(f"  models           {', '.join(config.model_names)}")
        print(f"  threshold        {config.inference.threshold}")
        if abs(config.inference.threshold - PUBLISHED_THRESHOLD) > 1e-9:
            warnings.append(
                f"inference.threshold is {config.inference.threshold}, but published "
                f"JANNUS metrics were produced at {PUBLISHED_THRESHOLD}. Results will "
                f"not be comparable to the reference figures."
            )
        print(f"  min_size         {config.inference.postprocessing.min_size}")
        if config.inert_keys:
            warnings.append(
                "Config keys present but never read by the inference path: "
                + ", ".join(config.inert_keys)
            )
    except JannusError as exc:
        problems.append(f"config: {exc}")
        print(f"  ! {exc}")

    print("\n[4/4] Model checkpoints")
    if args.skip_weights:
        print("  skipped (--skip-weights)")
    else:
        root = _repo_root(args)
        manifest_path = Path(args.weights_manifest) if args.weights_manifest else root / MANIFEST_FILENAME
        try:
            results = verify_manifest(
                manifest_path, root, roles=("base", "stacker"), raise_on_failure=False
            )
            for result in results:
                print(f"  {result.describe()}")
            failed = [r for r in results if not r.ok]
            if failed:
                problems.append(
                    f"{len(failed)} of {len(results)} checkpoints failed verification. "
                    f"Run `jannus fetch-weights`."
                )
        except WeightsError as exc:
            problems.append(f"weights: {exc}")
            print(f"  ! {exc}")

    print("\n" + RULE)
    for warning in warnings:
        print(f"  WARNING  {warning}")
    if problems:
        for problem in problems:
            print(f"  PROBLEM  {problem}")
        print(RULE)
        print("\nRESULT: NOT READY — resolve the problems above.")
        return 1
    print(RULE)
    print("\nRESULT: READY — this installation can run JANNUS.")
    return 0


# ---------------------------------------------------------------------------
# validate-data
# ---------------------------------------------------------------------------

def cmd_validate_data(args) -> int:
    from ..data.layout import format_alias_table, scan_dataset
    from ..data.validation import validate_dataset

    _banner("dataset validation")
    config = _require_config(args)
    print()
    print(format_alias_table(config.data))
    print()

    index = scan_dataset(
        args.input, config.data, require_ground_truth=args.require_ground_truth
    )
    report = validate_dataset(
        index, config.data, strict=args.strict, check_intensities=not args.fast
    )

    print(report.render())

    if args.output:
        out = Path(args.output)
        if out.suffix != ".json":
            out = out / "validation_report.json"
        report.write(out)
        print(f"\nFull report: {out}")

    return 0 if report.passed else 3


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def cmd_predict(args) -> int:
    import numpy as np
    import torch

    from ..data.layout import scan_dataset
    from ..data.loading import load_case, save_mask
    from ..data.validation import validate_dataset
    from ..segmentation.pipeline import BrainMetPipeline

    _banner("inference")
    provenance = _setup_run(args, "predict")
    config = _require_config(args)

    threshold = args.threshold if args.threshold is not None else config.inference.threshold
    if args.threshold is not None and abs(args.threshold - config.inference.threshold) > 1e-9:
        provenance.warnings.append(
            f"threshold overridden on the command line: {args.threshold} "
            f"(config says {config.inference.threshold}); results are not comparable "
            f"to published figures"
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    index = scan_dataset(args.input, config.data)
    print(f"\nFound {len(index)} case(s) under {args.input}")

    if not args.skip_validation:
        report = validate_dataset(index, config.data, check_intensities=True)
        blocked = {f.case for f in report.errors if f.case}
        if report.errors:
            print("\nDataset QC found errors:")
            print(report.render())
            if not blocked:
                raise DataLayoutError(
                    "Dataset QC failed at cohort level; cannot proceed.",
                    remedy="Run `jannus validate-data` and resolve the errors.",
                )
            print(f"\nSkipping {len(blocked)} case(s) that failed QC; continuing with the rest.")
        for warning in report.warnings:
            provenance.warnings.append(f"qc: {warning.code} {warning.case or ''}".strip())
    else:
        blocked = set()
        provenance.warnings.append("dataset QC was skipped (--skip-validation)")

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    if args.allow_stub:
        provenance.warnings.append(
            "STUB MODE — base models are randomly initialised. Outputs are meaningless."
        )
        print("\n*** WARNING: --allow-stub is set. Outputs are NOT valid results. ***\n")

    print("Building pipeline...")
    try:
        pipeline = BrainMetPipeline.from_config(
            config.source_path, device=device, stub=args.allow_stub,
            verify=not args.allow_stub,
        )
    except RuntimeError as exc:
        # from_config(verify=True) raises plain RuntimeError on silent
        # fall-through; re-raise as the typed error so the CLI exit code and
        # the remedy text are right.
        raise UnverifiedPipelineError(str(exc)) from exc

    provenance.record_checkpoints(dict(config.checkpoint_paths()))
    provenance.config = {
        "source": str(config.source_path),
        "inference": {
            "threshold": threshold,
            "postprocessing": {
                "min_size": config.inference.postprocessing.min_size,
                "opening_size": config.inference.postprocessing.opening_size,
                "closing_size": config.inference.postprocessing.closing_size,
            },
        },
        "stacking": {
            "patch_size": config.stacking.patch_size,
            "overlap": config.stacking.overlap,
        },
        "sequences": list(config.data.sequences),
    }

    todo = [c for c in index.cases if c.token not in blocked]
    provenance.n_cases_attempted = len(todo)
    print(f"\nProcessing {len(todo)} case(s)...\n")

    started = time.time()
    for position, case in enumerate(todo, start=1):
        out_path = output_dir / f"{case.case_id}_seg.nii.gz"
        if out_path.exists() and not args.overwrite:
            print(f"  [{position}/{len(todo)}] {case.token}: exists, skipping")
            provenance.n_cases_succeeded += 1
            continue

        case_started = time.time()
        try:
            volume = load_case(case, config.data.sequences)
            result = pipeline.predict_volume(
                volume.array, voxel_spacing=volume.voxel_spacing
            )
            mask = (result.probability_map[0] >= threshold).astype(np.uint8)
            save_mask(mask, volume.affine, out_path)

            if args.save_probabilities:
                np.save(
                    output_dir / f"{case.case_id}_prob.npy",
                    result.probability_map[0].astype(np.float16),
                )

            provenance.n_cases_succeeded += 1
            print(
                f"  [{position}/{len(todo)}] {case.token}: "
                f"{result.lesion_count} lesion(s), {int(mask.sum())} voxels, "
                f"{time.time() - case_started:.1f}s"
            )
        except Exception as exc:
            logger.exception("inference failed for %s", case.token)
            provenance.record_failure(case.token, str(exc))
            print(f"  [{position}/{len(todo)}] {case.token}: FAILED — {exc}")

    provenance.finish()
    manifest_path = provenance.write(output_dir / "provenance.json")

    print("\n" + RULE)
    for line in provenance.summary_lines():
        print(f"  {line}")
    print(f"  elapsed {time.time() - started:.0f}s")
    print(RULE)
    print(f"\nMasks:      {output_dir}")
    print(f"Provenance: {manifest_path}")

    if provenance.n_cases_succeeded == 0:
        raise PipelineError(
            f"All {provenance.n_cases_attempted} case(s) failed.",
            remedy="See provenance.json for per-case reasons.",
        )
    return 0 if not provenance.failures else 6


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def cmd_evaluate(args) -> int:
    from ..data.layout import scan_dataset
    from ..data.loading import load_mask
    from ..evaluation.harness import evaluate_cohort, match_predictions_to_truth
    from ..evaluation.report import render_markdown_report, render_text_summary

    _banner("evaluation")
    provenance = _setup_run(args, "evaluate")
    config = _require_config(args)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    index = scan_dataset(args.input, config.data, require_ground_truth=True)
    print(f"\nFound {len(index)} case(s) with ground truth under {args.input}")

    paired = match_predictions_to_truth(args.predictions, index, suffix=args.suffix)
    if not paired:
        raise DataLayoutError(
            f"No prediction/ground-truth pairs found. Looked for "
            f"'{{case_id}}{args.suffix}' in {args.predictions}.",
            remedy="Confirm --predictions points at the output of `jannus predict`.",
        )
    print(f"Matched {len(paired)} prediction/ground-truth pair(s)\n")

    loaded = []
    for case_token, pred_path, gt_path, spacing in paired:
        try:
            loaded.append(
                (case_token, load_mask(pred_path), load_mask(gt_path), spacing)
            )
        except Exception as exc:
            logger.error("could not load masks for %s: %s", case_token, exc)
            provenance.record_failure(case_token, f"mask load failed: {exc}")

    provenance.n_cases_attempted = len(paired)
    print("Computing metrics (bootstrap CIs may take a minute)...")
    result = evaluate_cohort(
        loaded, iou_threshold=args.iou_threshold, n_bootstrap=args.bootstrap
    )
    provenance.n_cases_succeeded = result.n_cases

    # Carry the operating point into the report: a reader must be able to see
    # which threshold produced these numbers.
    provenance.config = {
        "source": str(config.source_path),
        "inference": {"threshold": config.inference.threshold},
        "iou_threshold": args.iou_threshold,
        "bootstrap": args.bootstrap,
    }
    provenance.finish()

    results_path = output_dir / "results.json"
    results_path.write_text(
        json.dumps(result.to_dict(), indent=2, default=str) + "\n", encoding="utf-8"
    )
    report_path = output_dir / "report.md"
    report_path.write_text(
        render_markdown_report(
            result,
            site_name=args.site_name,
            provenance=provenance.to_dict(),
        ),
        encoding="utf-8",
    )
    provenance.write(output_dir / "provenance.json")

    print("\n" + RULE)
    print(render_text_summary(result))
    print(RULE)
    print(f"\nResults: {results_path}")
    print(f"Report:  {report_path}")
    print("\nThe Markdown report contains no PHI and is safe to return to the "
          "coordinating site.")
    return 0


# ---------------------------------------------------------------------------
# fetch-weights
# ---------------------------------------------------------------------------

def cmd_fetch_weights(args) -> int:
    _banner("model weights")
    root = _repo_root(args)
    manifest_path = (
        Path(args.weights_manifest) if args.weights_manifest else root / MANIFEST_FILENAME
    )
    dest_root = Path(args.dest) if args.dest else root

    entries = load_manifest(manifest_path)
    print(f"\nManifest: {manifest_path}")
    print(f"Target:   {dest_root}\n")

    from ..core.checksums import verify_entry

    missing = []
    for entry in entries:
        result = verify_entry(entry, dest_root)
        print(f"  {result.describe()}")
        if not result.ok:
            missing.append(entry)

    if not missing:
        print("\nAll checkpoints present and verified.")
        return 0

    if args.verify_only:
        print(f"\n{len(missing)} checkpoint(s) missing or corrupt (--verify-only, "
              f"nothing downloaded).")
        return 5

    downloadable = [e for e in missing if e.url]
    if not downloadable:
        raise WeightsError(
            f"{len(missing)} checkpoint(s) are missing and the manifest lists no "
            f"download URL for them.",
            remedy=(
                "Model weights are distributed separately from the source repository. "
                "Request access from the maintainer and place the files at the paths "
                "listed in weights.lock.json, then re-run `jannus fetch-weights "
                "--verify-only`. See docs/INSTALL.md."
            ),
        )

    import urllib.request

    for entry in downloadable:
        target = dest_root / entry.path
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nDownloading {entry.name} ({entry.bytes / 1e6:.0f} MB)...")
        tmp = target.with_suffix(target.suffix + ".partial")
        try:
            urllib.request.urlretrieve(entry.url, tmp)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise WeightsError(f"Download failed for {entry.name}: {exc}") from exc
        tmp.replace(target)

        result = verify_entry(entry, dest_root)
        print(f"  {result.describe()}")
        if not result.ok:
            # A checkpoint that fails its hash must never be left where the
            # pipeline could load it.
            target.unlink(missing_ok=True)
            raise WeightsError(
                f"{entry.name} failed verification after download and was removed."
            )

    print("\nAll checkpoints downloaded and verified.")
    return 0
