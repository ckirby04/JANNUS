"""
Side-by-side review gallery: raw MRI + raw MRI with GT overlay (+ optional
ensemble prediction) for each triage-flagged case.

Difference from build_review_gallery.py: this renders each slice as a
2- or 3-panel strip with FILLED overlays (not contour lines), so you can
see the lesion in context next to the annotation. Easier for eyeballing
whether a label is plausible.

Output:
  model/review_gallery_panels/index.html
  model/review_gallery_panels/cases/<case_id>.html
  model/review_gallery_panels/img/<case_id>/*.png   (each is a 3-panel strip)

Usage:
  python scripts/evaluation/build_raw_vs_gt_gallery.py
  python scripts/evaluation/build_raw_vs_gt_gallery.py --slices-per-case 25 --show-prediction
  python scripts/evaluation/build_raw_vs_gt_gallery.py --top-n 20
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

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.evaluation.seven_model_stacker import load_seven_preds

TRIAGE_PATH = PROJECT / "model" / "evaluation_results" / "triage_ranking.json"
GALLERY_DIR = PROJECT / "model" / "review_gallery_panels"
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
    """Prefer slices with GT lesions. If GT empty, use predicted slices.
    Pad +/- 2 slices of context above and below."""
    gt_axial = np.where(gt.sum(axis=(1, 2)) > 0)[0]
    pred_axial = np.where(pred.sum(axis=(1, 2)) > 0)[0]
    if len(gt_axial) > 0:
        focus = gt_axial
    elif len(pred_axial) > 0:
        focus = pred_axial
    else:
        H = gt.shape[0]
        return list(np.linspace(H // 4, 3 * H // 4, n_slices).astype(int))
    lo = max(int(focus.min()) - 2, 0)
    hi = min(int(focus.max()) + 2, gt.shape[0] - 1)
    if hi - lo + 1 <= n_slices:
        return list(range(lo, hi + 1))
    return list(np.linspace(lo, hi, n_slices).astype(int))


def _overlay(mri_slice: np.ndarray, mask_slice: np.ndarray,
             color: tuple) -> np.ndarray:
    """Blend a colored alpha overlay onto a grayscale MRI slice.
    Returns (H, W, 3) float [0, 1] RGB image.
    """
    mri_norm = (mri_slice - np.percentile(mri_slice, 1)) / \
               max(np.percentile(mri_slice, 99) - np.percentile(mri_slice, 1), 1e-6)
    mri_norm = np.clip(mri_norm, 0.0, 1.0)
    rgb = np.stack([mri_norm, mri_norm, mri_norm], axis=-1)
    if mask_slice.any():
        alpha = 0.45
        for c, v in enumerate(color):
            rgb[..., c] = np.where(mask_slice > 0,
                                    (1 - alpha) * rgb[..., c] + alpha * v,
                                    rgb[..., c])
    return rgb


def render_panel(mri: np.ndarray, gt: np.ndarray, pred: Optional[np.ndarray],
                 z: int, out_png: Path, show_prediction: bool, dpi: int = 80):
    mri_s = np.rot90(mri[z])
    gt_s = np.rot90(gt[z])
    panels = [
        ("Raw MRI", _overlay(mri_s, np.zeros_like(gt_s), (0, 0, 0))),
        ("Raw + GT (green)", _overlay(mri_s, gt_s, (0.2, 1.0, 0.35))),
    ]
    if show_prediction and pred is not None:
        pred_s = np.rot90(pred[z])
        panels.append(("Raw + Pred (red)", _overlay(mri_s, pred_s, (1.0, 0.2, 0.25))))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), dpi=dpi)
    if n == 1:
        axes = [axes]
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=11, color="white")
        ax.set_axis_off()
    fig.patch.set_facecolor("#111")
    fig.suptitle(f"z = {z}", fontsize=10, color="#bbb", y=0.02)
    fig.tight_layout(pad=0.3, rect=(0, 0.03, 1, 1))
    fig.savefig(out_png, facecolor="#111", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def build_case_page(case_id: str, stats: dict, slice_pngs: List[str],
                    rank: int, total: int, show_prediction: bool) -> str:
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
    pred_legend = ('<span><span class="box pred"></span> Ensemble prediction (red filled)</span>'
                   if show_prediction else '')
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Panels: {case_id}</title>
<style>
body {{ background: #0b0b0f; color: #e7e7e7; font-family: system-ui, sans-serif;
       margin: 16px 24px; }}
a {{ color: #4fc3f7; }}
h1 {{ margin: 0 0 4px 0; }}
.meta {{ color: #a0a0a0; margin-bottom: 12px; font-size: 14px; }}
.legend {{ display: flex; gap: 18px; margin-bottom: 14px; font-size: 14px; }}
.legend .box {{ display: inline-block; width: 14px; height: 14px;
               vertical-align: middle; border-radius: 2px; }}
.legend .gt {{ background: #33ff59; }}
.legend .pred {{ background: #ff3340; }}
.grid {{ display: grid; grid-template-columns: 1fr; gap: 8px; }}
.slice img {{ width: 100%; max-width: 1200px; display: block;
             border-radius: 4px; }}
.stats {{ border-collapse: collapse; margin-bottom: 16px; }}
.stats td {{ padding: 3px 14px; border-bottom: 1px solid #222; font-size: 13px; }}
.stats td:first-child {{ color: #8c8c8c; }}
.nav {{ margin: 10px 0 20px 0; }}
</style>
</head><body>
<div class="nav"><a href="../index.html">&laquo; back to index</a>
&nbsp;&nbsp;rank {rank}/{total}</div>
<h1>{html.escape(case_id)}</h1>
<div class="meta">Side-by-side panels: raw T1-gd MRI alongside annotated overlay.</div>
<table class="stats">{stats_rows}</table>
<div class="legend">
  <span>Left: raw MRI</span>
  <span><span class="box gt"></span> Middle: raw + GT (green filled)</span>
  {pred_legend}
</div>
<div class="grid">
{thumbs}
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
<title>BMS triage panels</title>
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
<h1>BMS triage panels (raw vs GT side-by-side)</h1>
<div class="summary">
Top {len(entries)} training cases ranked by <code>(ensemble agreement) x (1 - GT Dice)</code>.
Each case page shows side-by-side slices: raw T1-gd MRI next to an overlay view
with the ground-truth mask filled in green. Useful for eyeballing whether each
flagged annotation looks plausible on the raw image.
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
                    help="Override, default uses all top_cases from triage")
    ap.add_argument("--show-prediction", action="store_true",
                    help="Add a 3rd panel with the ensemble prediction overlay")
    args = ap.parse_args()

    with open(TRIAGE_PATH) as f:
        triage = json.load(f)
    entries = triage["top_cases"]
    if args.top_n is not None:
        entries = entries[:args.top_n]
    print(f"Building panel gallery for {len(entries)} cases "
          f"(show_prediction={args.show_prediction})", flush=True)

    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    (GALLERY_DIR / "cases").mkdir(exist_ok=True)
    (GALLERY_DIR / "img").mkdir(exist_ok=True)

    total = len(entries)
    for i, e in enumerate(entries):
        cid = e["case_id"]
        t1_path = find_t1gd(cid)
        if t1_path is None:
            print(f"  skip {cid}: no t1_gd.nii.gz", flush=True)
            continue
        mri = nib.load(str(t1_path)).get_fdata().astype(np.float32)
        preds, gt = load_seven_preds(cid)
        pred_bin = (preds.mean(axis=0) > 0.5).astype(np.float32) if args.show_prediction else None
        gt_bin = (gt > 0.5).astype(np.float32)

        if mri.shape != gt_bin.shape:
            from scipy.ndimage import zoom
            zf = tuple(g / m for g, m in zip(gt_bin.shape, mri.shape))
            mri = zoom(mri, zf, order=1)

        slice_ids = select_informative_slices(
            gt_bin, pred_bin if pred_bin is not None else gt_bin,
            args.slices_per_case)
        img_dir = GALLERY_DIR / "img" / cid
        img_dir.mkdir(parents=True, exist_ok=True)
        slice_pngs = []
        for z in slice_ids:
            name = f"z{z:03d}.png"
            render_panel(mri, gt_bin, pred_bin, z, img_dir / name,
                         args.show_prediction)
            slice_pngs.append(name)

        (GALLERY_DIR / "cases" / f"{cid}.html").write_text(
            build_case_page(cid, e, slice_pngs, rank=i + 1, total=total,
                            show_prediction=args.show_prediction),
            encoding="utf-8")

        if (i + 1) % 5 == 0 or i + 1 == total:
            print(f"  rendered [{i + 1}/{total}] {cid}", flush=True)

    (GALLERY_DIR / "index.html").write_text(
        build_index_page(entries), encoding="utf-8")
    print(f"\n  Gallery -> {GALLERY_DIR / 'index.html'}", flush=True)
    print(f"  Open: file:///{GALLERY_DIR / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
