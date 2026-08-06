# Retired configuration files

These two files lived in `configs/` through v1.40 and were removed in v1.50.

**Neither had a single consumer.** No Python file in `src/`, `scripts/`, `demo/` or
`tests/` referenced either filename, and none of their keys — `healthy_brain_dir`,
`train_data_dir`, `output_dir`, `num_workers`, `log_dir`, `synthetic_ratio`, `seed`,
`batch_size` — was read anywhere in the codebase.

`augmentation.yaml` even documented an invocation that does not exist:

```
python -m augmentation.pipeline --config configs/augmentation.yaml
```

`src/segmentation/augmentation.py` has no `argparse`, no `yaml` import, and no config
parameter.

They are retained here because they record intended training hyperparameters and are
useful as a historical reference. They are **not** live configuration: editing them has
no effect on anything.

They were removed from `configs/` because shipping dead configuration that looks live is
actively misleading to an external site — someone would reasonably tune `num_workers` or
`seed` here and conclude the software ignores them, which it does.

`augmentation.yaml` additionally contained an absolute path to a licensed cohort on one
developer's machine (`healthy_brain_dir: "G:/BrainMetShare/Raw Data/..."`), which is
another reason it should not have been distributed.

`jannus doctor` now reports any config key the inference path does not read, so this class
of dead configuration cannot silently reaccumulate.
