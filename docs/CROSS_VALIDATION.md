# External cross-validation protocol

For hospitals, academic groups and independent researchers validating JANNUS on their own
cohort. Following this end-to-end produces a self-contained, PHI-free report you can
return to the coordinating site.

**Your imaging never leaves your network.** The commands below make no outbound
connections.

---

## Before you begin

- [ ] Local IRB / research-ethics approval, or a documented exemption, covering
      retrospective algorithmic analysis of this cohort.
- [ ] Data de-identified per your institution's policy —
      see [PHI_AND_DEIDENTIFICATION.md](PHI_AND_DEIDENTIFICATION.md).
- [ ] Cohort laid out per [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md).
- [ ] A machine with a CUDA GPU (CPU works but takes ~1 hour/case).
- [ ] Model weights obtained and verified — see [INSTALL.md](INSTALL.md).

A note on cohort size: with fewer than about 30 cases, bootstrap confidence intervals
will be wide enough that a difference from our figures is hard to interpret. 50+ annotated
cases is a reasonable target for a meaningful comparison.

---

## Step 1 — Verify the installation

```bash
jannus doctor
```

This touches no patient data. It reports the Python environment, GPU availability,
required packages, the effective configuration, and verifies every checkpoint against
`weights.lock.json`.

**Do not proceed until this prints `RESULT: READY`.** In particular, if checkpoints fail
verification, the pipeline would run on randomly-initialised weights and produce
well-formed noise — indistinguishable from a real result until someone tried to publish it.

Save the output. It documents the environment your numbers came from.

---

## Step 2 — Validate the cohort

```bash
jannus validate-data \
  --input  /data/cohort \
  --output /out/validation \
  --require-ground-truth
```

Run this **before** spending GPU time. It checks every case for sequence presence,
co-registration, geometry, intensity sanity and ground-truth plausibility.

Findings are graded:

- **ERROR** — blocks that case. Fix it or exclude it, and record which cases you excluded.
- **WARNING** — the case runs, but the deviation must be reported with your numbers.
  JANNUS carries warnings into the provenance manifest so they travel with the results.
- **INFO** — worth knowing, no action needed.

For a formal validation cohort, add `--strict` to treat every warning as an error, so no
protocol deviation slips through unnoticed.

Send `/out/validation/validation_report.json` to the coordinating site along with your
final results. It contains no PHI.

---

## Step 3 — Run inference

```bash
jannus predict \
  --input  /data/cohort \
  --output /out/predictions \
  --log-file /out/predictions/run.log
```

Writes `{case_id}_seg.nii.gz` per case, plus `provenance.json`.

Notes:

- **Do not pass `--threshold`.** The default (0.55) is the published operating point.
  Overriding it makes your results incomparable, and JANNUS records the override in the
  provenance manifest so a reviewer can see it.
- The run resumes: re-running skips cases whose output already exists. Use `--overwrite`
  to force recomputation.
- A failing case is recorded and skipped rather than aborting the cohort.
- For a formal run, add `--strict-determinism` so any nondeterministic operation errors
  rather than silently varying between runs.

Expect roughly 3–8 minutes per case on a modern GPU.

---

## Step 4 — Evaluate

```bash
jannus evaluate \
  --input       /data/cohort \
  --predictions /out/predictions \
  --output      /out/results \
  --site-name   "Example University Hospital"
```

Produces:

| File | Contents |
|---|---|
| `report.md` | Human-readable report — **this is what you return** |
| `results.json` | Full per-case and aggregate metrics, machine-readable |
| `provenance.json` | Environment, config, checkpoint hashes |

All three are PHI-free: case identifiers appear only as salted pseudonyms.

---

## Step 5 — Return the results

Send `report.md`, `results.json`, `provenance.json` and the validation report.

Please also include, in prose:

1. **Cohort description** — n, date range, scanner vendors and field strengths, contrast
   protocol, and how cases were selected (consecutive? enriched for anything?).
2. **Annotation protocol** — who annotated, their experience, whether reads were consensus
   or single-reader, the boundary convention (enhancing rim only, or rim plus necrosis),
   and the minimum lesion size recorded.
3. **Exclusions** — how many cases you dropped and why.

Items 1 and 2 matter as much as the numbers. Most differences between sites come from
cohort difficulty and annotation convention rather than from the model.

---

## Interpreting your results

Compare against the internal reference in [MODEL_CARD.md](MODEL_CARD.md). The
measurable-disease scope (≥ 10 mm) is the meaningful comparison — it is the proposed
Indication for Use, and the scope on which JANNUS is competitive with the cleared
predicate.

### If your numbers are lower than ours

This is common and usually explicable. In rough order of likelihood:

- **Annotation convention.** A different boundary convention easily moves lesion-wise
  Dice by 0.05–0.10.
- **Smaller lesions.** Sensitivity is strongly size-dependent — 0.061 below 3 mm versus
  0.986 above 20 mm. Check the stratified table in your report before comparing headline
  numbers; a cohort weighted toward small lesions will look worse for reasons that have
  nothing to do with your site.
- **Acquisition protocol.** Thick slices and high anisotropy both reduce small-lesion
  sensitivity. Your validation report flags these.
- **Contrast protocol.** Dose and timing differences change enhancement.
- **Post-treatment cases.** The model was trained largely on pre-treatment imaging.
  Post-radiosurgery change, resection cavities and treatment-related enhancement are
  outside its training distribution.

A lower number at your site is a legitimate finding, not a failure of the run. It is
precisely the information external validation exists to produce — please report it as
measured.

### If your numbers are much *higher* than ours

Worth double-checking. Verify that ground truth was not accidentally used as input, that
prediction/ground-truth pairing is correct, and that `jannus doctor` reported READY.

### HD95 and MSD

These are far above the FDA thresholds in our own results (17.26 mm vs ≤ 2.94 mm) and
will likely be high at your site too. They are dominated by isolated false-positive
components far from any true lesion, which drag the 95th-percentile surface distance
enormously. This is a known limitation, documented rather than hidden — see
[MODEL_CARD.md](MODEL_CARD.md).

---

## Reproducibility

To confirm a run reproduces on your own hardware, run `jannus predict` twice into
different output directories and compare:

```bash
diff <(sha256sum /out/pred_a/*_seg.nii.gz | awk '{print $1}') \
     <(sha256sum /out/pred_b/*_seg.nii.gz | awk '{print $1}')
```

Identical hashes mean determinism held. If they differ, check
`provenance.json` → `determinism.fully_deterministic`; some GPU operations have no
deterministic implementation, and JANNUS records that rather than claiming otherwise.

Two sites can confirm they ran the same model by comparing the `checkpoints` block in
their provenance manifests — those are SHA-256 hashes of the actual loaded weights.

---

## Getting help

Open an issue at <https://github.com/ckirby04/JANNUS/issues>.

**Never attach patient imaging, DICOM files, or raw case identifiers to an issue.** The
JSON reports and the `--log-file` output are already pseudonymised and safe to share.
Start with [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
