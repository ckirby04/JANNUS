# Split ID sources

Copied 2026-10-04 under a one-time, read-only exception granted by the project owner
(Phase 0 approval, Q2). Only case identifiers were copied; no metric values, imaging or
other content. Nothing was written to the source folders.

| File here | Source (read-only) | Source SHA-256 | What was copied |
|---|---|---|---|
| `eval84_ids.txt` | `G:\BrainMetShare\1.40\model\evaluation_results\stacking_v5_results_per_case.json` | `14711580de25a79e2be965d88a6dc761a71811287bfb1e216526c158b54d12f4` | sorted top-level keys (84 case IDs) |
| `nnunet_splits_final_ids.json` | `G:\BrainMetShare\1.40\nnUNet\nnUNet_preprocessed\Dataset001_BrainMets\splits_final.json` | `50ef0e13837ec22b6183647fab484dc687604c9422dfd1808535e208cc0ff021` | 5 folds x {train, val} ID lists, each sorted |

SHA-256 of the files written here: `eval84_ids.txt` = `52b6c668440b8bb144b0c0ec0ec676cea696bccc9f8f39ed26d20ac2fe73e36f`,
`nnunet_splits_final_ids.json` = `b48592c91ed7390e58b3d37086130c8102442387e322365203298f5412cfca5b`.

## Not found

**Stacker internal-validation IDs: not logged.** `scripts/evaluation/seven_model_stacker.py`
prints only the counts of its 90/10 split. The production checkpoint stores only
`model_state_dict`, `in_channels`, `epoch` (22) and `val_loss` (0.1789). Only the exact
paths above were opened; no frozen directory was listed or searched.

The equivalent paths under `1.30` and `1.1` were tested for existence only (`test -f`),
and do not exist.
