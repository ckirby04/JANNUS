"""Generate and cache probability maps for the detection experiment.

A wrapper around the unchanged production pipeline. For each case it calls
``BrainMetPipeline.predict_volume`` (the same call ``jannus predict`` makes,
via the same ``load_case`` normalisation) and saves:

    stacker   (H, W, D) float16    PipelineResult.probability_map[0]
    base      (7, H, W, D) uint8   PipelineResult.base_probs, round(p * 255)
    base_names                     STACKING_MODEL_NAMES order

to ``cache/maps/{case}.npz``. The script resumes: a case whose npz already
exists is skipped. Each written file's SHA-256 goes into a JSONL manifest that
is committed, while the maps themselves are gitignored.

    python experiments/lesion_detection/run_inference.py --config experiments/lesion_detection/configs/inference.yaml
    python experiments/lesion_detection/run_inference.py --config ... --only BMS_Mets_026   # single case
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from jannus.core.checksums import sha256_file  # noqa: E402
from jannus.core.config import load_config  # noqa: E402
from jannus.core.determinism import configure_determinism  # noqa: E402
from jannus.data.layout import scan_dataset  # noqa: E402
from jannus.data.loading import load_case  # noqa: E402
from jannus.evaluation.detection.provenance import (  # noqa: E402
    build_provenance,
    split_record,
    write_json,
)
from jannus.evaluation.detection.scoring import quantize_uint8  # noqa: E402


def read_ids(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--only", nargs="*", help="restrict to these case IDs (must be in the config's lists)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if cfg["storage"] != {"stacker_dtype": "float16", "base_dtype": "uint8"}:
        raise SystemExit("only float16 stacker / uint8 base storage is implemented")

    split_files = [REPO / p for p in cfg["cases"]]
    ordered: list[str] = []
    for f in split_files:
        ordered += [c for c in read_ids(f) if c not in ordered]
    if args.only:
        unknown = set(args.only) - set(ordered)
        if unknown:
            raise SystemExit(f"--only cases not in the configured splits: {sorted(unknown)}")
        ordered = [c for c in ordered if c in set(args.only)]

    cache_dir = REPO / cfg["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = REPO / cfg["manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    import torch

    from jannus.segmentation.pipeline import BrainMetPipeline
    from jannus.segmentation.stacking import STACKING_MODEL_NAMES

    determinism = configure_determinism(cfg["determinism"]["seed"], strict=cfg["determinism"]["strict"])
    models_cfg = load_config(REPO / cfg["models_config"])
    provenance = build_provenance(
        command="run_inference",
        repo_root=REPO,
        config=cfg,
        config_path=args.config,
        splits=[split_record(f.stem, f, read_ids(f)) for f in split_files],
        extra={
            "inference_threshold_in_models_yaml": models_cfg.inference.threshold,
            "determinism": getattr(determinism, "__dict__", str(determinism)),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    )

    index = scan_dataset(REPO / cfg["data"]["images_root"] if not Path(cfg["data"]["images_root"]).is_absolute()
                         else cfg["data"]["images_root"], models_cfg.data)
    by_id = {c.case_id: c for c in index.cases}
    missing = [c for c in ordered if c not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} configured case(s) not found or unresolved: {missing[:5]}")

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    pipeline = BrainMetPipeline.from_config(models_cfg.source_path, device=device, verify=True)

    todo = [c for c in ordered if args.overwrite or not (cache_dir / f"{c}.npz").exists()]
    print(f"{len(ordered)} configured, {len(ordered) - len(todo)} cached, {len(todo)} to run on {device}",
          flush=True)
    write_json(REPO / cfg["provenance"], provenance)

    for i, case_id in enumerate(todo, start=1):
        t0 = time.time()
        volume = load_case(by_id[case_id], models_cfg.data.sequences)
        with torch.inference_mode():
            result = pipeline.predict_volume(volume.array, voxel_spacing=volume.voxel_spacing)
        stacker = np.clip(result.probability_map[0], 0.0, 1.0)
        base = np.stack([result.base_probs[n] for n in STACKING_MODEL_NAMES], axis=0)

        out = cache_dir / f"{case_id}.npz"
        tmp = cache_dir / f"{case_id}.npz.partial"
        with open(tmp, "wb") as fh:
            np.savez_compressed(
                fh,
                stacker=stacker.astype(np.float16),
                base=quantize_uint8(np.clip(base, 0.0, 1.0)),
                base_names=np.array(STACKING_MODEL_NAMES),
            )
        os.replace(tmp, out)

        record = {
            "case": case_id,
            "file": str(out.relative_to(REPO)).replace("\\", "/"),
            "sha256": sha256_file(out),
            "bytes": out.stat().st_size,
            "seconds": round(time.time() - t0, 1),
            "loader_spacing": list(volume.voxel_spacing),
            "shape": list(stacker.shape),
            "stacker_max": float(stacker.max()),
            "n_vox_ge_055": int((stacker >= 0.55).sum()),
        }
        with open(manifest_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        print(f"[{i}/{len(todo)}] {case_id}: {record['seconds']}s, {record['bytes'] / 2**20:.1f} MiB",
              flush=True)
        del result, stacker, base, volume
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
