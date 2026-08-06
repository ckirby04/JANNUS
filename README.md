# JANNUS

**Brain metastasis segmentation on multi-sequence MRI, packaged for multi-site external validation.**

> **RESEARCH USE ONLY.** JANNUS is not a medical device. It is not cleared or approved by
> any regulatory body, and must not be used for diagnosis, treatment planning, or any
> other clinical decision-making. See [`docs/INTENDED_USE.md`](docs/INTENDED_USE.md).

JANNUS segments measurable brain metastases from four co-registered MRI sequences
(T1 pre-contrast, T1 post-gadolinium, FLAIR, T2) using a seven-model ensemble fused by a
learned stacking meta-classifier. Version 1.50 exists for one purpose: to let hospitals,
academic groups, and independent researchers run the model on **their own** cohorts and
report back numbers that can be trusted and compared.

---

## What v1.50 is for

If you are an external site, this is your entire workflow:

```bash
pip install "jannus[segmentation,nnunet]"

jannus doctor                                    # is this machine set up correctly?
jannus fetch-weights                             # download + verify checkpoints
jannus validate-data --input /data/cohort        # QC before spending GPU time
jannus predict  --input /data/cohort --output /out/pred
jannus evaluate --input /data/cohort --predictions /out/pred --output /out/results
```

`/out/results/report.md` is a self-contained, PHI-free Markdown report you can send back
to the coordinating site. Your imaging never leaves your network — JANNUS makes no
outbound network calls in this workflow.

Start with [`docs/CROSS_VALIDATION.md`](docs/CROSS_VALIDATION.md), which walks the full
protocol including what to do when the numbers disagree with ours.

---

## Architecture

```
Four co-registered MRI sequences (T1, T1+Gd, FLAIR, T2), native resolution
  │
  ├─ nnU-Net 3D full-res ─┐
  ├─ nnU-Net 2D           │
  ├─ LightweightUNet3D ps=8   │
  ├─ LightweightUNet3D ps=12  ├─ 7 full-volume probability maps
  ├─ LightweightUNet3D ps=24  │
  ├─ LightweightUNet3D ps=36  │
  └─ SwinUNETR (96³ ROI) ─┘
  │
  ▼
9-channel feature cube  =  7 predictions + voxel-wise variance + (max − min) range
  │
  ▼
StackingClassifierV2  (~1.36M params, 32³ sliding window, 0.5 overlap;
                       residual 3D CNN, SE attention, multi-scale branch)
  │
  ▼
threshold 0.55  →  morphological cleanup (min_size = 0)  →  binary NIfTI mask
```

Channel order is a hard contract: the stacker was trained against that exact ordering,
and permuting it produces a well-formed but meaningless output. `jannus.core.config`
enforces it regardless of the order entries appear in your config file.

---

## Performance (internal 84-case held-out cohort)

These are **our** numbers on **our** cohort. Your cohort will differ, and that is the
point of running the validation.

### All annotated lesions

| Metric | Value |
|---|---|
| Voxel Dice | 0.7858 |
| Lesion-wise sensitivity | 0.810 (95% CI 0.758–0.858) |
| Lesion-wise Dice (matched) | 0.791 |
| False positives / case | 1.68 (95% CI 1.26–2.12) |
| HD95 | 18.36 mm (95% CI 12.5–25.0) |

### RANO-BM measurable disease (longest axis ≥ 10 mm)

This is the scope of the proposed Indication for Use.

| Metric | JANNUS | Neosoma K252922 | FDA threshold |
|---|---|---|---|
| Lesion-wise sensitivity | **0.943** (0.90–0.98) | 0.90 | ≥ 0.85 — pass |
| Lesion-wise Dice (matched) | 0.835 | 0.86 | ≥ 0.70 — pass |
| False positives / case | 1.61 | 0.57 | ≤ 5 — pass |
| HD95 | 17.26 mm | 1.78 mm | ≤ 2.94 — **fail** |
| Mean surface distance | 3.89 mm | 0.36 mm | ≤ 0.66 — **fail** |

HD95 and MSD are the outstanding gaps and are stated plainly rather than omitted. They
are dominated by isolated false-positive components far from any true lesion; see
[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for the analysis and known limitations.

### By lesion size

| Longest axis | GT lesions | Sensitivity | FP/case | Mean Dice |
|---|---|---|---|---|
| < 3 mm | 33 | 0.061 | 0.54 | 0.53 |
| 3–5 mm | 82 | 0.439 | 0.54 | 0.64 |
| 5–10 mm | 304 | 0.773 | 0.46 | 0.72 |
| 10–20 mm | 144 | **0.903** | 0.12 | 0.81 |
| > 20 mm | 73 | **0.986** | 0.02 | 0.87 |

Below 3 mm the model is at the floor of MRI resolution and inter-annotator agreement.
Do not read the sub-3 mm row as a performance claim.

---

## Installation

Requires Python 3.10+. A CUDA GPU is strongly recommended — CPU inference works but takes
roughly an hour per case instead of a few minutes.

```bash
pip install "jannus[segmentation,nnunet]"
```

The core `jannus` install is deliberately minimal (numpy, scipy, pyyaml, nibabel) so you
can install it, run `jannus validate-data` against your cohort, and review the source
before pulling in a multi-gigabyte deep-learning stack. Extras: `segmentation`, `nnunet`,
`api`, `rag`, `demo`.

**Model weights are not distributed in this repository.** They are pinned by SHA-256 in
[`weights.lock.json`](weights.lock.json) and obtained separately — see
[`docs/INSTALL.md`](docs/INSTALL.md). `jannus doctor` refuses to run inference against
unverified checkpoints, because a silently-stubbed model produces plausible-looking noise
that would otherwise be indistinguishable from a real validation result.

---

## Repository layout

```
src/jannus/
  core/          config, logging, provenance, determinism, checksums, paths
  data/          dataset discovery, QC validation, volume loading
  segmentation/  base models, adapters, the stacker, postprocessing
  evaluation/    metrics, bootstrap CIs, stratification, reporting
  api/           optional FastAPI service
  rag/           optional literature retrieval
  cli/           the `jannus` command
configs/models.yaml    base-model registry, operating point, sequence naming
weights.lock.json      SHA-256 of every checkpoint
tests/unit/            synthetic-fixture tests, no weights or PHI required
scripts/               research and training scripts (not part of the package)
docs/                  installation, data spec, validation protocol, model card
```

---

## Documentation

| Document | Read it when |
|---|---|
| [INSTALL.md](docs/INSTALL.md) | Setting up, including air-gapped installs |
| [DATA_REQUIREMENTS.md](docs/DATA_REQUIREMENTS.md) | Preparing your cohort |
| [CROSS_VALIDATION.md](docs/CROSS_VALIDATION.md) | Running the validation protocol |
| [MODEL_CARD.md](docs/MODEL_CARD.md) | Understanding scope and limitations |
| [INTENDED_USE.md](docs/INTENDED_USE.md) | Regulatory status and IFU |
| [PHI_AND_DEIDENTIFICATION.md](docs/PHI_AND_DEIDENTIFICATION.md) | Before touching patient data |
| [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Something failed |
| [CHANGELOG.md](CHANGELOG.md) | Upgrading from 1.40 — **read the breaking changes** |
| [SECURITY.md](SECURITY.md) | Reporting a vulnerability |

---

## Upgrading from 1.40

v1.50 changes results. The most important item, in full:

> v1.40's `configs/models.yaml` never defined `inference.threshold`, so the pipeline fell
> through to a hardcoded `0.5` — while every published metric was produced by the
> evaluation scripts at `0.55`. A site running the documented inference command
> reproduced neither the config nor the paper. v1.50 makes the key explicit and correct.

If you ran v1.40 inference, your masks were produced at the wrong operating point and
should be regenerated. The full list is in [CHANGELOG.md](CHANGELOG.md).

---

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE). Model weights are distributed under separate terms.
