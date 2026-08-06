# Installation

Python 3.10+. A CUDA GPU is strongly recommended: CPU inference works but takes roughly an
hour per case against a few minutes on GPU.

---

## Standard install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install "jannus[segmentation,nnunet]"
jannus doctor
```

## From source

```bash
git clone https://github.com/ckirby04/JANNUS.git
cd JANNUS
pip install -e ".[segmentation,nnunet,dev]"
jannus doctor
```

---

## Extras

The core `jannus` install pulls only numpy, scipy, pyyaml and nibabel. That is deliberate:
you can install it, run `jannus validate-data` against your cohort, and have your security
team review the source before a multi-gigabyte deep-learning stack enters your network.

| Extra | Adds | Needed for |
|---|---|---|
| *(core)* | numpy, scipy, pyyaml, nibabel | `validate-data`, `evaluate` |
| `segmentation` | torch, monai, SimpleITK, scikit-image | `predict` |
| `nnunet` | nnunetv2 | 2 of the 7 base models |
| `api` | fastapi, uvicorn, pydicom, highdicom, reportlab | the optional REST service |
| `rag` | langchain, chromadb, transformers, LLM clients | literature retrieval |
| `demo` | gradio, matplotlib, plotly | the interactive demo |

`jannus[segmentation]` without `nnunet` will fail to build the pipeline — nnU-Net 3D and
2D are two of the seven required base models. `jannus doctor` says so explicitly.

**Do not install `rag` inside a hospital network without review.** It brings in outbound
LLM clients, and report generation on that path can transmit case context to a third
party. See [PHI_AND_DEIDENTIFICATION.md](PHI_AND_DEIDENTIFICATION.md).

---

## Model weights

**Weights are not distributed in this repository**, and a fresh clone cannot run inference
until you obtain them.

Eight checkpoints are required, totalling roughly 780 MB. Every one is pinned by SHA-256
and byte size in [`weights.lock.json`](../weights.lock.json):

| Name | Role | Size |
|---|---|---|
| `nnunet_3d` | base | 247 MB |
| `nnunet_2d` | base | 165 MB |
| `swin_unetr` | base | 256 MB |
| `patch_8` / `patch_12` / `patch_24` / `patch_36` | base | 27 MB each |
| `stacker` | meta-learner | 5.5 MB |

### Obtaining them

Weights are released under separate terms from the source code. Request access from the
maintainer (see [CITATION.cff](../CITATION.cff)), then place each file at the path listed
in `weights.lock.json` relative to the repository root, and verify:

```bash
jannus fetch-weights --verify-only
```

Every line must read `OK`. If the manifest carries download URLs for your distribution,
`jannus fetch-weights` (without `--verify-only`) fetches and verifies them automatically;
a checkpoint that fails its hash after download is deleted rather than left where the
pipeline could load it.

### Why verification is mandatory

A missing checkpoint used to fall back to a randomly-initialised stub. A stub emits a
well-formed probability map in the right shape and range — it just contains no
information. In a validation study that becomes a published number with nothing to flag
it. `jannus predict` therefore refuses to run unless every checkpoint loads and verifies.

`--allow-stub` exists for plumbing smoke tests only. It stamps a warning into the
provenance manifest, and its outputs must be discarded.

---

## Air-gapped installation

The validation workflow itself needs no network. Installing does.

On a networked machine with the same OS, Python version and CPU architecture:

```bash
pip download "jannus[segmentation,nnunet]" -d ./jannus-wheels
```

Transfer `jannus-wheels/` and the model checkpoints, then:

```bash
pip install --no-index --find-links ./jannus-wheels "jannus[segmentation,nnunet]"
jannus doctor
```

Two things reach the network at *runtime* and must be avoided or pre-staged inside an
air-gapped environment:

- **BiomedCLIP weights** are fetched from HuggingFace when `jannus.rag` builds an
  embedder. Only affects the optional `rag` extra.
- **MONAI model-zoo pretrained weights** (`swin_unetr_btcv.pt`) are only needed for
  *training* a SwinUNETR from scratch, not for inference.

Neither is touched by `validate-data`, `predict` or `evaluate`.

---

## Docker

```bash
docker build -t jannus:1.50 .
docker run --gpus all \
  -v /data/cohort:/data:ro \
  -v /out:/out \
  jannus:1.50 \
  jannus predict --input /data --output /out
```

Mount your imaging read-only. The container runs as a non-root user.

---

## Verifying the install

```bash
jannus doctor
```

Checks Python and package versions, GPU availability, config validity (including whether
your operating threshold matches the published one), and every checkpoint hash. It exits
non-zero if anything would prevent a valid run.

To confirm the code itself is intact:

```bash
pip install -e ".[dev]"
pytest -m "not slow and not gpu and not weights"
```

That suite needs no GPU, no network, no weights and no patient data.

---

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
