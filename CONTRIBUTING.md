# Contributing to JANNUS

Thanks for your interest. JANNUS processes patient imaging, so a few of the rules below
are stricter than you may be used to.

Collaboration is especially welcome on: external validation cohorts, consensus
re-annotation, HD95/MSD improvement (see the limitations in
[docs/MODEL_CARD.md](docs/MODEL_CARD.md)), and DWI/ADC sequence integration.

## Ground rules

1. **Never commit patient data.** No `.nii`, `.nii.gz`, `.dcm`, `.npz`, `.npy`, or any
   file derived from real imaging. CI fails the build if any appear. Tests use synthetic
   volumes generated in-process.
2. **Never commit credentials.** `.env` is gitignored; use `.env.example` for
   placeholders. If a key ever reaches a remote, rotate it — do not rewrite it out of
   history and assume it is safe.
3. **Never commit model weights.** They are pinned by hash in `weights.lock.json` and
   distributed separately.
4. **Anything that can move a metric is a breaking change**, however small the diff, and
   must be called out in `CHANGELOG.md`.

## Setup

```bash
git clone https://github.com/ckirby04/JANNUS.git
cd JANNUS
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,segmentation]"
pytest -m "not slow"
```

## Before opening a pull request

```bash
ruff check src/ tests/
mypy
pytest -m "not slow"
```

New code in `jannus/core/`, `jannus/data/` and `jannus/cli/` is type-checked strictly
(`disallow_untyped_defs`). The modules migrated from v1.40 are not yet annotated; if you
touch one substantially, annotating it is welcome but not required.

## Testing expectations

- Tests must pass with **no GPU, no network, no model weights, and no patient data**.
- Mark anything that needs those: `@pytest.mark.gpu`, `@pytest.mark.network`,
  `@pytest.mark.weights`, `@pytest.mark.slow`.
- Derive expected values from the test's own construction rather than recording them from
  a previous run. A test that only pins current behaviour cannot tell you that behaviour
  was wrong — which is how `fda_metrics` reached 540 lines with zero coverage while
  producing the numbers in our publications.

## Changing the operating point

`inference.threshold` in `configs/models.yaml` determines every published metric. If you
change it:

1. Update `PUBLISHED_THRESHOLD` in `jannus/core/config.py` to match.
2. Bump `PIPELINE_REVISION` in `jannus/_version.py`.
3. Regenerate the reference figures in `README.md` and `docs/MODEL_CARD.md`.
4. Document it prominently in `CHANGELOG.md` under breaking changes.

Sites compare results across versions using `PIPELINE_REVISION`. Leaving it unchanged
after a metric-affecting edit silently tells them two incomparable runs are comparable.

## Adding a config key

Add it to `CONSUMED_KEYS` in `jannus/core/config.py` at the same time as the code that
reads it. `jannus doctor` reports keys nothing consumes, which is how the inert
`ensemble.default_threshold` in v1.40 would have been caught.

## Editing configs

`configs/*.yaml` and `src/jannus/configs/*.yaml` must stay byte-identical — the first is
what a clone uses, the second is what ships in the wheel. A test enforces this. After
editing the repo copy:

```bash
cp configs/models.yaml src/jannus/configs/models.yaml
```

## Security and privacy issues

Report privately — see [SECURITY.md](SECURITY.md). Do not open a public issue.

## Commit messages

Conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`) are
preferred but not enforced. Explain *why* in the body; the diff already shows what.
