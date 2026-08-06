# Troubleshooting

Start with `jannus doctor` — it diagnoses most setup problems without touching patient
data.

---

## "No usable cases found under /data/cohort"

JANNUS found directories but none contained all four required sequences. The error lists
the missing channel per case.

```bash
jannus validate-data --input /data/cohort
```

Most common cause: your site names sequences differently. Add your naming to
`configs/models.yaml` rather than renaming your imaging:

```yaml
data:
  sequence_aliases:
    t1_gd: ["t1_gd", "t1c", "T1c", "your_local_name"]
```

Second most common: an extra nesting level. JANNUS expects
`<root>/<case_id>/<sequence>.nii.gz`, not `<root>/<case_id>/<study>/<sequence>.nii.gz`.

---

## "All sequences must be co-registered to a common grid"

Two channels have different dimensions. JANNUS will not resample between them, because
doing so silently would hide a co-registration failure that changes the result.

Co-register upstream (ANTs, elastix, FSL FLIRT), then re-run. See
[DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md).

---

## "checkpoint(s) failed verification" / `UnverifiedPipelineError`

```bash
jannus fetch-weights --verify-only
```

- `MISSING` — the file is not at the path in `weights.lock.json`.
- `SIZE` — truncated download, or a Git LFS pointer file rather than the real weights.
- `HASH` — the file is the right size but different content. Do not use it.

JANNUS refuses to proceed here on purpose. An unverified checkpoint falls back to random
initialisation, which produces a well-formed probability map containing no information —
indistinguishable from a real result until someone publishes it.

---

## "nnunetv2 is not installed"

```bash
pip install "jannus[nnunet]"
```

nnU-Net supplies two of the seven base models; the pipeline cannot be built without it.
It was not declared as a dependency at all before v1.50.

---

## Inference is extremely slow

Check `jannus doctor` for `cuda available`. On CPU, expect roughly an hour per case
against 3–8 minutes on GPU.

If a GPU is present but unused, your torch build is probably CPU-only:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Reinstall torch with the CUDA build matching your driver, from pytorch.org.

---

## CUDA out of memory

Native-resolution volumes with a 7-model ensemble are memory-hungry. Options, in order of
preference:

1. Run on a GPU with more memory.
2. Fall back to CPU for the affected cases: `--device cpu`.
3. Process in smaller batches — JANNUS already handles one case at a time, so this means
   splitting the cohort across runs.

Do not reduce the stacker patch size or overlap to fit: those are part of the trained
configuration and changing them invalidates comparison with published results.

---

## Results differ between two runs on the same data

Check `provenance.json`:

```json
"determinism": { "fully_deterministic": false, "warnings": [...] }
```

Some GPU operations have no deterministic implementation. Add `--strict-determinism` to
error rather than silently vary. Note that full determinism can be unattainable on some
hardware/driver combinations; JANNUS records what it achieved rather than claiming
success.

---

## My metrics are much lower than the published figures

Work through, in order:

1. **Check the threshold.** `jannus doctor` warns if it differs from 0.55. If you ran
   v1.40, your masks were generated at 0.5 and should be regenerated.
2. **Check the size distribution.** Sensitivity ranges from 0.061 below 3 mm to 0.986
   above 20 mm. A cohort weighted toward small lesions looks worse for reasons unrelated
   to your site. Compare the stratified table, not the headline number.
3. **Check the validation warnings.** Anisotropic voxels and out-of-range spacing both
   reduce small-lesion sensitivity, and are flagged.
4. **Check the annotation convention.** Enhancing rim only versus rim plus necrosis moves
   lesion-wise Dice by 0.05–0.10.

A lower number at your site is a legitimate finding. Report it as measured — see
[CROSS_VALIDATION.md](CROSS_VALIDATION.md).

---

## "No prediction/ground-truth pairs found"

`jannus evaluate` looks for `{case_id}_seg.nii.gz` in `--predictions`, matched against
case directory names under `--input`. If you renamed the masks, pass `--suffix`.

---

## API returns 401 Unauthorized

Authentication is required by default as of v1.50 (it was off by default before, which
left an unauthenticated service handling patient imaging). Provide an `X-API-Key` header.

On an isolated research machine you may set `AUTH_REQUIRED=false`, but admin endpoints
still require a real key, and you should not do this on a networked host.

---

## Import errors after upgrading from 1.40

The installed package was renamed from `src` to `jannus`:

```python
from src.segmentation.pipeline import BrainMetPipeline    # 1.40
from jannus.segmentation.pipeline import BrainMetPipeline # 1.50
```

`fda_metrics` moved to `jannus.evaluation.metrics`. `scripts/inference/run_inference.py`
still works but is a deprecation shim over `jannus predict`.

---

## Still stuck

Open an issue at <https://github.com/ckirby04/JANNUS/issues> with the output of
`jannus doctor` and your `provenance.json`.

**Never attach patient imaging, DICOM files, or raw case identifiers.** The JSON reports
and `--log-file` output are already pseudonymised.
