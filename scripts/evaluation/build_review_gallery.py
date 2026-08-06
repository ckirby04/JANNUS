"""
Build a static HTML review gallery for the top-N triage-ranked cases.

For each flagged case:
  - Load T1-gd (primary brain-mets sequence)
  - Load GT mask + ensemble prediction (from cached base preds)
  - Find lesion-containing axial slices
  - Render each as a PNG with toggleable GT (green) and prediction (red) overlays
  - Create a per-case HTML page with the slice grid + stats
  - Create an index.html linking to all cases ranked by triage score

Output:
  model/review_gallery/index.html            (master list)
  model/review_gallery/cases/<case_id>.html  (per-case page)
  model/review_gallery/img/<case_id>/*.png   (slice images)

Usage:
  python scripts/evaluation/build_review_gallery.py
  python scripts/evaluation/build_review_gallery.py --slices-per-case 20
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import label as ndimage_label

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.evaluation.seven_model_stacker import load_seven_preds

TRIAGE_PATH = PROJECT / "model" / "evaluation_results" / "triage_ranking.json"
GALLERY_DIR = PROJECT / "model" / "review_gallery"
DATA_DIRS = [
    PROJECT / "data" / "preprocessed_256" / "train",
    PROJECT / "data" / "train",
]


def find_t1gd(case_id: str) -> Optional[Path]:
    for base in DATA_DIRS:
        p = base / case_id / "t1_gd.nii.gz"
        if p.exists():
            return p
    return None


def select_informative_slices(gt: np.ndarray, pred: np.ndarray,
                              n_slices: int) -> List[int]:
    """Pick axial slice indices focused on lesion-containing slices, plus a
    couple of context slices above/below."""
    lesion_axial = np.where((gt.sum(axis=(1, 2)) + pred.sum(axis=(1, 2))) > 0)[0]
    if len(lesion_axial) == 0:
        H = gt.shape[0]
        return list(np.linspace(H // 4, 3 * H // 4, n_slices).astype(int))
    lo, hi = lesion_axial.min(), lesion_axial.max()
    # pad +/- 2 slices for context
    lo = max(lo - 2, 0)
    hi = min(hi + 2, gt.shape[0] - 1)
    if hi - lo + 1 <= n_slices:
        return list(range(lo, hi + 1))
    return list(np.linspace(lo, hi, n_slices).astype(int))


def render_slice(mri: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                 z: int, out_png: Path, dpi: int = 80):
    """Render one axial slice with GT (green) and prediction (red) contours."""
    fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)
    ax.imshow(np.rot90(mri[z]), cmap="gray", vmin=np.percentile(mri, 1),
              vmax=np.percentile(mri, 99))
    if gt[z].any():
        ax.contour(np.rot90(gt[z]), levels=[0.5], colors="#00e676",
                   linewidths=1.2)
    if pred[z].any():
        ax.contour(np.rot90(pred[z]), levels=[0.5], colors="#ff5252",
                   linewidths=1.2, linestyles="--")
    ax.set_axis_off()
    ax.set_title(f"z = {z}", fontsize=10, color="white")
    fig.patch.set_facecolor("#111")
    fig.tight_layout(pad=0.2)
    fig.savefig(out_png, facecolor="#111", bbox_inches="tight",
                pad_inches=0.05)
    plt.close(fig)


def build_case_page(case_id: str, stats: dict, slice_pngs: List[str],
                    rank: int, total: int) -> str:
    stats_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v:.4f}</td></tr>"
        if isinstance(v, float) else
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in stats.items()
    )
    thumbs = "\n".join(
        f'<div class="slice"><img src="../img/{case_id}/{p}" alt="{p}"></div>'
        for p in slice_pngs
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Triage: {case_id}</title>
<style>
body {{ background: #0b0b0f; color: #e7e7e7; font-family: system-ui, sans-serif;
       margin: 16px 24px; }}
a {{ color: #4fc3f7; }}
h1 {{ margin: 0 0 4px 0; }}
.meta {{ color: #a0a0a0; margin-bottom: 12px; font-size: 14px; }}
.legend {{ display: flex; gap: 18px; margin-bottom: 14px; font-size: 14px; }}
.legend .box {{ display: inline-block; width: 14px; height: 14px; vertical-align: middle; }}
.legend .gt {{ background: #00e676; }}
.legend .pred {{ background: #ff5252; border: 1px dashed #fff; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
        gap: 8px; }}
.slice img {{ width: 100%; display: block; border-radius: 4px; }}
.stats {{ border-collapse: collapse; margin-bottom: 16px; }}
.stats td {{ padding: 3px 14px; border-bottom: 1px solid #222; font-size: 13px; }}
.stats td:first-child {{ color: #8c8c8c; }}
.nav {{ margin: 10px 0 20px 0; }}
</style>
</head><body>
<div class="nav"><a href="../index.html">&laquo; back to index</a>
&nbsp;&nbsp;rank {rank}/{total}</div>
<h1>{html.escape(case_id)}</h1>
<div class="meta">Per-case triage stats</div>
<table class="stats">{stats_rows}</table>
<div class="legend">
  <span><span class="box gt"></span> Ground truth mask (solid)</span>
  <span><span class="box pred"></span> Ensemble prediction (dashed)</span>
</div>
<div class="grid">
{thumbs}
</div>
<div class="nav" style="margin-top:24px;">
To edit: run <code>scripts/evaluation/open_in_itksnap.ps1 {html.escape(case_id)}</code>
from PowerShell, edit the mask, save, then run
<code>python scripts/evaluation/commit_mask_fix.py {html.escape(case_id)} &lt;path-to-new-mask.nii.gz&gt;</code>.
</div>
</body></html>
"""


def build_index_page(entries: List[dict]) -> str:
    rows = "\n".join(
        f'<tr><td>{i + 1}</td><td><a href="cases/{e["case_id"]}.html">{html.escape(e["case_id"])}</a></td>'
        f'<td>{e["case_dice_vs_ensemble"]:.3f}</td>'
        f'<td>{e["mean_agreement"]:.3f}</td>'
        f'<td>{100 * e["confident_fp_frac_of_burden"]:.2f}%</td>'
        f'<td>{100 * e["confident_fn_frac_of_burden"]:.2f}%</td>'
        f'<td>{e["triage_score"]:.4f}</td></tr>'
        for i, e in enumerate(entries)
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>BMS triage gallery</title>
<style>
body {{ background: #0b0b0f; color: #e7e7e7; font-family: system-ui, sans-serif;
       margin: 16px 24px; }}
a {{ color: #4fc3f7; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: 6px 10px; border-bottom: 1px solid #222; text-align: left;
         font-size: 13px; }}
th {{ background: #18181c; color: #a0a0a0; }}
tr:hover {{ background: #14141a; }}
.summary {{ color: #a0a0a0; margin-bottom: 16px; font-size: 13px; }}
</style>
</head><body>
<h1>BMS triage gallery</h1>
<div class="summary">
Top {len(entries)} training cases ranked by <code>(ensemble agreement) x (1 - GT Dice)</code>.
High-rank cases are where the 7 base models CONFIDENTLY agree on something that
doesn't match the ground truth -- candidate label errors.
<br><br>
Workflow: click a case, scroll through axial slices, decide if the GT (green)
or the ensemble (dashed red) is more reasonable. Edit in ITK-SNAP, commit back,
retrain.
</div>
<table>
<thead><tr><th>#</th><th>Case</th><th>Dice vs ensemble</th><th>Mean agreement</th>
<th>Confident FP %</th><th>Confident FN %</th><th>Triage score</th></tr></thead>
<tbody>
{rows}
</tbody></table>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices-per-case", type=int, default=20)
    ap.add_argument("--top-n", type=int, default=None,
                    help="Override -- by default uses all top_cases from triage")
    args = ap.parse_args()

    with open(TRIAGE_PATH) as f:
        triage = json.load(f)
    entries = triage["top_cases"]
    if args.top_n is not None:
        entries = entries[:args.top_n]
    print(f"Building gallery for {len(entries)} cases", flush=True)

    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    (GALLERY_DIR / "cases").mkdir(exist_ok=True)
    (GALLERY_DIR / "img").mkdir(exist_ok=True)

    total = len(entries)
    for i, e in enumerate(entries):
        cid = e["case_id"]
        t1_path = find_t1gd(cid)
        if t1_path is None:
            print(f"  skip {cid}: no t1_gd.nii.gz found", flush=True)
            continue
        mri = nib.load(str(t1_path)).get_fdata().astype(np.float32)
        preds, gt = load_seven_preds(cid)
        ensemble_mean = preds.mean(axis=0)
        # For gallery purposes, threshold ensemble at 0.5 to get a clean binary
        pred_bin = (ensemble_mean > 0.5).astype(np.float32)
        gt_bin = (gt > 0.5).astype(np.float32)

        if mri.shape != gt_bin.shape:
            # Resample MRI to match the mask shape — mask is 256^3, MRI might
            # differ. Use simple nearest-neighbor-like selection.
            # (BMS/UCSF cases are already 256^3 so this rarely fires.)
            from scipy.ndimage import zoom
            z = tuple(g / m for g, m in zip(gt_bin.shape, mri.shape))
            mri = zoom(mri, z, order=1)

        slice_ids = select_informative_slices(gt_bin, pred_bin,
                                               args.slices_per_case)
        img_dir = GALLERY_DIR / "img" / cid
        img_dir.mkdir(parents=True, exist_ok=True)
        slice_pngs = []
        for z in slice_ids:
            name = f"z{z:03d}.png"
            render_slice(mri, gt_bin, pred_bin, z, img_dir / name)
            slice_pngs.append(name)

        (GALLERY_DIR / "cases" / f"{cid}.html").write_text(
            build_case_page(cid, e, slice_pngs, rank=i + 1, total=total),
            encoding="utf-8")

        if (i + 1) % 5 == 0 or i + 1 == total:
            print(f"  rendered [{i + 1}/{total}] {cid}", flush=True)

    (GALLERY_DIR / "index.html").write_text(
        build_index_page(entries), encoding="utf-8")
    print(f"\n  Gallery -> {GALLERY_DIR / 'index.html'}", flush=True)
    print(f"  Open in browser: file:///{GALLERY_DIR / 'index.html'}",
          flush=True)


if __name__ == "__main__":
    main()
