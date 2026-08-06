# Data requirements

What your cohort must look like for JANNUS to run on it. Read this before preparing data;
then run `jannus validate-data`, which checks every requirement here automatically and
tells you per case what is wrong.

---

## Directory layout

One directory per case, each containing four co-registered sequences:

```
/data/cohort/
  PT_001/
    t1_pre.nii.gz
    t1_gd.nii.gz
    flair.nii.gz
    t2.nii.gz
    seg.nii.gz        # ground truth — only needed for `jannus evaluate`
  PT_002/
    ...
```

**Case directory names do not matter.** Use whatever your site already uses. A directory
is treated as a case if it contains the required imaging. (v1.40 filtered on a hardcoded
prefix allowlist, so sites with their own naming loaded zero cases and were told nothing;
that is fixed.)

Pointing `--input` at a single case directory also works, if you want to test one case.

---

## Sequences

Four channels are required. JANNUS accepts several filenames for each, tried in order:

| Channel | Accepted filenames |
|---|---|
| `t1_pre` | `t1_pre`, `t1`, `T1`, `t1_native`, `T1w` |
| `t1_gd` | `t1_gd`, `t1c`, `T1c`, `t1_post`, `t1ce`, `T1wCE`, `t1_gd_reg` |
| `flair` | `flair`, `FLAIR`, `t2_flair`, `T2wFLAIR` |
| `t2` | `t2`, `T2`, `bravo`, `BRAVO`, `T2w` |

Matching is case-insensitive. Extensions: `.nii.gz` (preferred), `.nii`, `.nrrd`, `.mha`,
`.mhd`.

`bravo` maps to the T2 channel because the BrainMetShare cohort used the GE BRAVO
sequence there.

**If your site's naming isn't listed, add it to the config rather than renaming your
imaging.** In `configs/models.yaml`:

```yaml
data:
  sequence_aliases:
    t1_gd: ["t1_gd", "t1c", "T1c", "your_local_name"]
```

Run `jannus validate-data` afterwards to confirm the resolution — the report shows which
filename matched each channel.

### The post-contrast sequence matters most

T1 post-gadolinium carries most of the signal for enhancing metastases. If your contrast
protocol differs substantially (dose, timing, agent), expect sensitivity to move, and
record the protocol alongside your results.

---

## Co-registration

**All four sequences must be on a common voxel grid.** JANNUS does not resample between
them, and will refuse a case whose channels disagree in shape.

This is deliberate. Silently resampling would hide a co-registration failure that
materially changes the result, and a validation study is exactly where that must not
happen. Co-register upstream with your own tooling (ANTs, elastix, FSL FLIRT, or your
scanner vendor's), then run JANNUS.

You do **not** need to resample to any particular resolution — native spacing is fine and
preferred.

---

## Geometry

| Property | Expected | Outside the range |
|---|---|---|
| Dimensions | each axis 32–1024 voxels | < 32 is an error; > 1024 warns |
| Voxel spacing | 0.3–5.0 mm | warns — results are extrapolation |
| Anisotropy | max/min spacing ≤ 4 | warns — small-lesion sensitivity drops |

The spacing range reflects what the model saw in training. Outside it, JANNUS still runs,
but the validation report marks the deviation and it must be reported with your numbers.

Voxel spacing is read from the NIfTI header and is used for every physical-distance
metric (HD95, MSD) and for the lesion-size stratification. **A wrong header silently
corrupts those metrics**, so confirm your spacing is right before running.

---

## Intensities

- Raw scanner intensities are expected. JANNUS z-scores each channel internally.
- Pre-normalised input is accepted — validation notes it, so you can confirm no
  intensity clipping was also applied.
- Skull-stripped or brain-extracted input is acceptable, but keep it consistent across
  the cohort and note it in your report.
- A constant channel (std = 0) is an error: it means an empty acquisition or a
  placeholder file.

---

## Ground truth (for `jannus evaluate` only)

Place a segmentation in the case directory named `seg`, `gt`, `label`, `mask`, or `truth`.

- Any non-zero label is foreground. If you also annotate oedema or resection cavities
  with distinct labels, they collapse into the metastasis-vs-background task JANNUS is
  scored on — separate them out beforehand if that is not what you want.
- It must match the imaging dimensions exactly.
- Empty ground truth is allowed (the case then contributes only false positives), but is
  flagged.
- Ground truth covering more than 5% of the volume is flagged as implausible — brain
  metastases are typically well under 1%, so this usually means an inverted mask or the
  wrong structure.

### Annotation protocol affects your numbers

Lesion-wise Dice and HD95 are sensitive to boundary convention — whether you include the
enhancing rim only, or rim plus central necrosis. Sensitivity is sensitive to the minimum
lesion size your annotators recorded. Please state both in your report; a difference from
our figures is often an annotation-protocol difference rather than a model difference.

---

## What JANNUS does *not* require

- No particular scanner vendor or field strength.
- No specific matrix size or slice thickness (within the ranges above).
- No DICOM — NIfTI is the supported input for the validation workflow.
- No network access. The `validate-data` → `predict` → `evaluate` path makes no outbound
  connections. Your imaging never leaves your network.

---

## Before you start: de-identification

JANNUS does not de-identify your data, and NIfTI conversion is not by itself a
de-identification step. See [PHI_AND_DEIDENTIFICATION.md](PHI_AND_DEIDENTIFICATION.md)
before copying imaging anywhere.

Note in particular that **case directory names become case identifiers**. If you name
directories by MRN, that MRN is your identifier. JANNUS pseudonymises identifiers in logs
and reports with a salted hash, but the right control is not to put an MRN in a folder
name in the first place — use a study ID and keep the linking key separately.
