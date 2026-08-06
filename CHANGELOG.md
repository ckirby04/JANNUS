# Changelog

All notable changes to JANNUS are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow semantic versioning, with the caveat that any change able to move a
reported metric is treated as breaking regardless of its size.

---

## [1.50.0] — 2026-08-05

The v1.40 → v1.50 release converts a working research codebase into a product that can be
handed to an external hospital or academic group and run against their own patients.
Nothing about the model architecture or the trained weights changed. What changed is
everything around them: correctness of the operating point, reproducibility, data
portability, security, and the ability to verify that a returned result is real.

### ⚠️ Breaking — results change

- **The inference threshold is now 0.55, and is actually read.**
  `configs/models.yaml` in v1.40 never defined `inference.threshold`. `BrainMetPipeline`
  looked it up, found nothing, and fell through to a hardcoded `0.5`. Meanwhile every
  published figure — voxel Dice 0.7858, measurable-disease sensitivity 0.943 — was
  produced by the evaluation scripts, which used `0.55`.

  The consequence: a site that followed the documented inference command reproduced
  neither the shipped config nor the published paper, and had no way to notice. This was
  the single most likely cause of a failed external cross-validation.

  Masks generated with v1.40 were produced at the wrong operating point and should be
  regenerated. `jannus doctor` now warns whenever the configured threshold differs from
  the published one.

- **`ensemble:` and `detection:` config sections removed.** Neither was read by the
  inference path. `ensemble.default_threshold: 0.5` was actively harmful: it looked like
  the operating point, sat beside the real (missing) key, and editing it did nothing.
  `jannus doctor` now reports any config key no consumer reads, so dead configuration
  cannot silently reaccumulate.

- **Case discovery no longer filters on folder-name prefixes.** v1.40 gated discovery on
  a hardcoded allowlist (`Mets_`, `UCSF_`, `Yale_`, `BraTS_`, `BMS_`, …). A site whose
  cases were named `PT_001/` or `MGH_0042/` loaded **zero cases**, with no error and no
  explanation. A directory is now a case if it contains the required imaging, and a scan
  that resolves nothing raises with a per-case breakdown instead of returning empty.

- **Python 3.10 is the new minimum** (was a nominal 3.9 that CI never tested).

### 🔒 Security

Every item below was a live defect in v1.40.

- **Arbitrary file write via DICOM upload filename (critical).** `server.py` joined the
  client-supplied multipart filename straight onto the temp directory. Both
  `../../../etc/cron.d/x` and an absolute `/etc/cron.d/x` worked — the latter because
  `pathlib` discards the left operand entirely when the right side is absolute. As the
  container ran as root, this was remote code execution. Filenames are now reduced to
  their basename, matching what the sibling NIfTI handler already did.
- **`AUTH_REQUIRED` now defaults to `true`.** v1.40 defaulted to `false`, so a site that
  followed the README got an unauthenticated service handling patient imaging.
- **`check_permission` denies anonymous callers.** It previously returned early on
  `key_info=None`, making every `check_permission(..., "admin")` guard a no-op. Combined
  with the old default, `/admin/stats`, `/admin/predictions` (which enumerates case
  identifiers) and `/admin/keys` were readable by anyone who could reach the port. Admin
  permissions now require a real key even when a site has explicitly enabled anonymous
  access.
- **`/predict/{job_id}/mask` requires authentication.** It was the only handler with no
  auth dependency at all, serving patient segmentation masks unauthenticated even when
  `AUTH_REQUIRED=true`.
- **Anonymous requests are rate-limited by client IP.** The unauthenticated path
  bypassed the limiter entirely, making denial of service against a GPU-bound endpoint
  trivial.
- **CORS defaults to no cross-origin access** (was `*` with `allow_credentials=True`).
- **Temp-directory leak fixed.** Every `/mask` download left a directory containing a
  patient segmentation on disk permanently.
- **DICOM private tags are no longer read.** `LoadPrivateTagsOn()` pulled in vendor
  private blocks — a well-known PHI reservoir — despite nothing in the code reading one.

### 🔁 Reproducibility and provenance

- **`provenance.json` accompanies every run**: JANNUS version, pipeline revision, git
  commit and dirty-tree flag, package versions, hardware, the effective config including
  the operating point, per-case failures, and the **SHA-256 of every checkpoint actually
  loaded**. This is what lets a coordinating site verify a returned result.
- **`weights.lock.json` pins all eight checkpoints by SHA-256 and byte size.** Nothing in
  v1.40 ever hashed a `.pth`, so there was no way to prove two sites ran the same model.
- **Unverified pipelines are fatal.** A stubbed adapter is randomly initialised and emits
  a well-formed probability map that is pure noise. `jannus predict` refuses to run
  unless every checkpoint loads, rather than letting noise become a published number.
- **Seeding and determinism are configured before any tensor work**, across python,
  numpy and torch, including cuDNN determinism and `CUBLAS_WORKSPACE_CONFIG`. What was
  actually achieved is recorded in the provenance manifest rather than assumed.
- **The version is single-sourced** from `jannus._version`. It was hardcoded as `1.23.0`
  in five places, including the `SoftwareVersions` tag written into every emitted
  DICOM-SEG and the footer of every clinician-facing PDF — a traceability defect.

### 🧪 Testing

- **`fda_metrics` (now `jannus.evaluation.metrics`) has tests.** 540 lines computing every
  FDA-facing and RANO-BM number a site reports back carried **zero** coverage. It now has
  35 tests covering lesion matching, IoU thresholding, greedy one-to-one assignment, both
  Dice conventions, surface distances under anisotropic spacing, bootstrap CIs, and size
  stratification. Expectations are hand-derived from synthetic geometry, not recorded
  from a previous run.
- 103 new unit tests overall, none requiring a GPU, network, weights, or patient data.
- Markers (`slow`, `gpu`, `weights`, `network`) let a site run a fast subset.
- `pytest -x` removed: fail-fast showed one failure per run, which is wrong for a suite
  external sites run and report back.
- `test_swin_unetr_resume.py` fixed — its test double pinned an exact `run_phase`
  signature and broke when `early_stop_patience` was added, so the whole file errored at
  collection. Now marked `slow` (it runs four full SwinUNETR CPU backward passes).

### 📦 Packaging

- **Renamed the installed package from `src` to `jannus`.** v1.40 installed a top-level
  package literally named `src`, which collides with any other project doing the same.
- **`jannus` console entrypoint** with `doctor`, `validate-data`, `predict`, `evaluate`
  and `fetch-weights`. Typed exit codes so a site's batch scheduler can branch on them.
- **`configs/` ships as package data.** It did not in v1.40, so a pip-installed copy had
  no `models.yaml` and could not build a pipeline at all.
- **Dependencies are bounded, and split into extras.** v1.40 made every dependency
  mandatory with no upper bound, so a hospital wanting only segmentation was forced to
  install langchain, chromadb, gradio and outbound LLM clients. MONAI's `SwinUNETR`
  signature changed across 1.3→1.4, so unbounded ranges alone could invalidate a
  cross-site comparison.
- **`nnunetv2` is declared.** It supplies 2 of the 7 base models and was not a dependency
  at all — `pip install .` produced an installation that could never load them.
- **CI actually verifies the package**: matrix across Python 3.10–3.12 and Windows,
  wheel build installed into a clean venv, secret scanning, a check that no imaging or
  weights are tracked, and dependency audit.

### 🧭 Portability

- **One sequence resolver.** v1.40 had four mutually inconsistent ones: the CLI defaulted
  to `bravo.nii.gz`, the API hardcoded `t2.nii.gz` and rejected `bravo`, the demo probed
  one then the other, and the dataset loader had its own list. Naming is now declared in
  `configs/models.yaml` under `data.sequence_aliases`, and one resolver serves every entry
  point. BrainMetShare (`bravo`), TCIA (`T1c`/`T2w`) and BIDS-ish naming all work with no
  code change.
- **Repo-root discovery replaced.** Eight modules located the root by counting
  `Path(__file__).parent` hops, encoding their own depth in the tree; two were already
  wrong in v1.40, resolving to `<repo>/scripts`. `jannus.core.paths` searches upward for
  root markers and honours `$JANNUS_HOME`.
- **Checkpoint paths resolve against the config file, not the working directory.**
- **The `data/` symlink is gone.** It pointed at an absolute path on one developer's
  machine and was dangling or absent everywhere else. Sites pass `--input`, or set
  `$JANNUS_DATA_ROOT`.
- **Hardcoded cohort identifiers removed.** `weighted_sampling.DIFFICULT_CASES` shipped 13
  internal case IDs that matched nothing at any other site, so the intended oversampling
  silently never happened.

### ✨ Added

- **`jannus validate-data`** — cohort QC before any GPU time is spent. Checks sequence
  presence, co-registration onto a common grid, voxel spacing and field of view against
  the training range, anisotropy, intensity distributions, and ground-truth plausibility.
  Findings are graded, and warnings travel into the provenance manifest so a deviation is
  reported alongside the numbers rather than lost.
- **`jannus doctor`** — environment, config and checkpoint verification without touching
  patient data.
- **PHI-safe structured logging** — case identifiers are pseudonymised with a salted
  digest, and identifier-shaped text (long digit runs, DICOM UIDs, API keys) is redacted
  by a filter rather than by asking call sites to remember. v1.40's
  `docs/risk_analysis.md` claimed "no PHI in system logs — only anonymized case IDs";
  no anonymization existed.
- **Bootstrap confidence intervals and size stratification in every report**, plus a
  self-contained PHI-free Markdown report to return to the coordinating site.
- **Per-case error isolation** — one unreadable file no longer destroys a cohort run that
  took days.

### 🐛 Fixed

- `scripts/inference/run_inference.py` referenced an undefined `SEQUENCES` in its
  missing-file error path, so the first problem a new site hit raised a `NameError` from
  inside the error handler. The script is now a thin deprecation shim over
  `jannus predict`.
- `scripts/preprocessing/setup_nnunet.py` and `scripts/inference/nnunet_probs.py`
  computed the project root with one too few `.parent` hops.
- `aggregate_stratified` result keys (`n_gt_total`, not `n_gt`) were rendered incorrectly
  by the report generator, blanking a column.
- `update-rag-corpus.yml` was broken three ways: it called a script that does not exist,
  `git add`-ed a gitignored directory so it could never commit, and pushed to `main`
  unreviewed. Now manual-dispatch only, fails loudly, and opens a PR.

### 📚 Documentation

- New: `INSTALL.md`, `DATA_REQUIREMENTS.md`, `CROSS_VALIDATION.md`, `MODEL_CARD.md`,
  `PHI_AND_DEIDENTIFICATION.md`, `TROUBLESHOOTING.md`, `SECURITY.md`.
- `docs/risk_analysis.md` corrected: mitigations M-013, M-021, M-022 and M-023 were
  marked implemented but were absent from the code. In a document accompanying an FDA or
  IRB submission those were material misstatements.

### Known limitations carried forward

- HD95 and MSD remain far above the FDA thresholds, dominated by isolated false-positive
  components distant from true lesions. Improving them requires new imaging modalities
  (DWI/ADC) or consensus re-annotation — data-side investments, not model-side. Eleven
  stacker configurations across seven rounds were explored; see
  `docs/research/stacking_architectures_explored.md`.
- Sub-3 mm sensitivity (0.061) is at the floor of MRI resolution and annotator agreement.
- nnU-Net 2D uses fold 0 only; folds 1–4 stopped early without a final checkpoint.

---

## [1.40.0] and earlier

Development history predating the v1.50 productization. See
`docs/research/` for the experiment chronology.
