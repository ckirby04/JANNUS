"""FROC figures and markdown helpers for detection reports.

matplotlib is an optional dependency (the ``demo`` extra), so it is imported
inside each plotting function. The colours are a CVD-validated categorical
order, applied in fixed order and never cycled. Thresholds are ordered, so
they use a single-hue sequential ramp. Every figure carries a legend, and every
number it shows is also in a table in the report, so no value is conveyed by
colour alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .metrics import FROC

#: Categorical slots in fixed order (validated: CVD dE >= 9.1 adjacent, light surface).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
#: Sequential blue ramp for ordered values (voxel thresholds), light -> dark.
SEQUENTIAL = ("#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
              "#1c5cab", "#184f95", "#104281", "#0d366b", "#0b2c58")
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _axes(ax, title: str, x_max: float) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Mean false positives per scan", color=MUTED)
    ax.set_ylabel("Lesion sensitivity", color=MUTED)
    ax.set_title(title, color=TEXT, fontsize=10, loc="left")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_visible(False)


def _step(ax, curve: FROC, x_max: float, **kw) -> None:
    x = np.append(curve.fp_per_scan, min(curve.fp_per_scan[-1], x_max))
    y = np.append(curve.sensitivity, curve.sensitivity[-1])
    ax.step(x, y, where="post", linewidth=2, **kw)


def plot_froc_comparison(
    panels: Sequence[tuple[str, Sequence[tuple[str, FROC]], int]],
    path: str | Path,
    *,
    x_max: float = 4.0,
    targets: Sequence[float] = (0.5, 1, 2, 3),
    points: Sequence[Sequence[tuple[str, float, float]]] | None = None,
) -> Path:
    """Small multiples, one panel per entry ``(title, [(label, curve), ...], n_lesions)``.

    The curves in a panel take categorical slots in order. ``points[i]`` may
    mark operating points ``(label, fp_per_scan, sensitivity)`` on panel ``i``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(panels)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False,
                             facecolor=SURFACE)
    for i, (title, curves, n_les) in enumerate(panels):
        ax = axes[i // cols][i % cols]
        _axes(ax, f"{title}  (n={n_les} lesions)", x_max)
        for t in targets:
            ax.axvline(t, color=GRID, linewidth=1, linestyle=":")
        for j, (label, curve) in enumerate(curves):
            _step(ax, curve, x_max, color=SERIES[j], label=label)
        for j, (label, fp, sens) in enumerate(points[i] if points else []):
            ax.plot([fp], [sens], marker="o", markersize=8, color=SERIES[j], markeredgecolor=SURFACE,
                    markeredgewidth=2, linestyle="none", label=label)
        ax.legend(fontsize=7, frameon=False, labelcolor=TEXT, loc="lower right")
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].set_visible(False)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_threshold_grid(
    panels: Sequence[tuple[str, Sequence[tuple[float, FROC]], float | None]],
    path: str | Path,
    *,
    x_max: float = 4.0,
) -> Path:
    """One panel per scorer. Each panel holds one curve per voxel threshold on a sequential
    ramp, and the selected threshold is drawn thicker. Panel entry: ``(title, [(thr, curve)],
    selected_thr_or_None)``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(panels)
    cols = min(n, 2)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.8 * rows), squeeze=False,
                             facecolor=SURFACE)
    for i, (title, curves, selected) in enumerate(panels):
        ax = axes[i // cols][i % cols]
        _axes(ax, title, x_max)
        ramp = np.linspace(0, len(SEQUENTIAL) - 1, len(curves)).round().astype(int)
        for k, (thr, curve) in enumerate(curves):
            is_sel = selected is not None and abs(thr - selected) < 1e-9
            _step(ax, curve, x_max, color=SERIES[1] if is_sel else SEQUENTIAL[ramp[k]],
                  label=f"{thr:.2f}{' (selected)' if is_sel else ''}",
                  zorder=3 if is_sel else 2)
        ax.legend(title="voxel threshold", fontsize=6, title_fontsize=7, frameon=False,
                  labelcolor=TEXT, loc="lower right", ncol=2)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].set_visible(False)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def fmt_ci(point: float, lo: float, hi: float, digits: int = 3) -> str:
    if not np.isfinite(point):
        return "n/a"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{point:.{digits}f}"
    return f"{point:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def md_table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)
