"""
visualization/enhanced.py
-------------------------
Per-apply() 3-panel diagnostic figure for GridCorrector2D, enabled via
GridCorrector2DConfig.enhanced_visualization.

  Panel 1 — input cloud, points violating rd_test (nn < rd_test) highlighted
  Panel 2 — tiling grid overlaid on the cloud, one example tile expanded:
            core / adjacent-ghost / PBC-ghost points, ghost-buffer boundary,
            PBC images drawn at their wrapped positions outside the domain
  Panel 3 — corrected cloud, an arrow before -> after for every moved point

All coordinates are in the corrector's internal frame [0, domain)^2 — the
same frame the tiling grid lives in, so grid lines align with the tiles the
model actually saw.

Palette validated for CVD separation + contrast on the dark surface
(dataviz six-checks, all-pairs within each panel). The one floor-band pair
(teal <-> pink) never shares a panel and differs in marker shape.
"""
from typing import Optional, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from inference.pipeline.pbc import build_ghost_tile, min_nn_pbc
from inference.pipeline.tiling import TilingConfig

C_CORE    = '#3b82f6'   # blue   — core points of the example tile
C_ADJ     = '#ea580c'   # orange — adjacent-tile ghosts
C_PBC     = '#ec4899'   # pink   — PBC-wrapped ghosts (diamond marker)
C_VIOL    = '#ef4444'   # red    — rd_test-violating points (panel 1)
C_MOVED   = '#0d9488'   # teal   — moved points + arrows (panel 3)
C_CONTEXT = '#4b5563'   # muted context points (not a categorical slot)
C_GHOSTBOX = '#eab308'  # annotation line: ghost-buffer boundary (dashed)
C_DOMAIN   = '#22c55e'  # annotation line: domain edge (dotted)
BG_FIG    = '#0f1117'
BG_AX     = '#141821'
GRID_COLOR = '#334455'
INK       = '#e2e8f0'
INK_MUTED = '#94a3b8'


def _style_ax(ax, title):
    ax.set_facecolor(BG_AX)
    for sp in ax.spines.values():
        sp.set_color('#1e2a3a')
    ax.tick_params(colors='#475569', labelsize=7)
    ax.set_aspect('equal')
    ax.set_title(title, color=INK, fontsize=9, pad=5)


def _legend(ax):
    ax.legend(fontsize=7, loc='upper right', facecolor='#1e2a3a',
              labelcolor=INK, edgecolor='#334455', framealpha=0.9)


def _pbc_displacement(before, after, domain):
    d = after - before
    d -= np.round(d / domain) * domain
    return d


def plot_enhanced_correction(
    pts_before:     np.ndarray,
    pts_after:      np.ndarray,
    tc:             TilingConfig,
    ghost_width:    float,
    rd_test:        float,
    k:              int,
    highlight_tile: Tuple[int, int] = (0, 0),
    moved_thresh:   Optional[float] = None,
    save_path:      Optional[str] = None,
    show:           bool = False,
):
    """
    Parameters
    ----------
    pts_before   : (N, 2) input positions in [0, domain)^2
    pts_after    : (N, 2) corrected positions in [0, domain)^2
    tc           : TilingConfig used for this apply() call (carries domain)
    ghost_width  : the ghost-buffer width that was used
    rd_test      : minimum-distance constraint of the input cloud
    k            : number of correction passes that produced pts_after
    highlight_tile : (i, j) tile expanded in panel 2 — default (0,0), a
                     corner tile, so PBC wrapping is visible
    moved_thresh : |displacement| above which a point counts as moved
                   (default 0.05 * rd_test)
    """
    domain = tc.domain
    G, c   = tc.n_cells, tc.cell_size
    thresh = moved_thresh if moved_thresh is not None else 0.05 * rd_test

    nn_before = min_nn_pbc(pts_before, domain)
    nn_after  = min_nn_pbc(pts_after,  domain)
    viol_before = nn_before < rd_test
    viol_after  = nn_after  < rd_test

    disp  = _pbc_displacement(pts_before, pts_after, domain)
    mag   = np.linalg.norm(disp, axis=1)
    moved = mag > thresh

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.6))
    fig.patch.set_facecolor(BG_FIG)

    # ── Panel 1: input cloud + violations ────────────────────────────────
    ax = axes[0]
    _style_ax(ax, f'input cloud — {viol_before.sum()} points violate rd_test={rd_test}')
    ax.scatter(pts_before[~viol_before, 0], pts_before[~viol_before, 1],
               c=C_CONTEXT, s=4, alpha=0.6, linewidths=0,
               label=f'ok  (n={int((~viol_before).sum())})')
    ax.scatter(pts_before[viol_before, 0], pts_before[viol_before, 1],
               c=C_VIOL, s=10, alpha=0.9, linewidths=0, zorder=3,
               label=f'nn < rd_test  (n={int(viol_before.sum())})')
    ax.set_xlim(-0.02 * domain, 1.02 * domain)
    ax.set_ylim(-0.02 * domain, 1.02 * domain)
    _legend(ax)

    # ── Panel 2: tiling grid + example tile's ghost neighbourhood ────────
    ti, tj = highlight_tile
    lo = np.array([ti * c, tj * c])
    hi = lo + c
    ext, is_core, _ = build_ghost_tile(pts_before, lo, hi, ghost_width, domain)
    ghosts  = ext[~is_core]
    is_pbc  = (ghosts.min(axis=1) < -1e-9) | (ghosts.max(axis=1) > domain + 1e-9)
    adj_g, pbc_g = ghosts[~is_pbc], ghosts[is_pbc]
    core    = ext[is_core]

    ax = axes[1]
    _style_ax(ax, f'{G}x{G} grid — tile ({ti},{tj}) neighbourhood: '
                  f'{int(is_core.sum())} core + {len(ghosts)} ghost')
    ax.scatter(pts_before[:, 0], pts_before[:, 1],
               c=C_CONTEXT, s=4, alpha=0.6, linewidths=0)
    for g in range(G + 1):
        ax.axhline(g * c, color=GRID_COLOR, lw=0.7)
        ax.axvline(g * c, color=GRID_COLOR, lw=0.7)
    ax.scatter(core[:, 0], core[:, 1], c=C_CORE, s=16, alpha=0.9, zorder=4,
               label=f'core  (n={len(core)})')
    ax.scatter(adj_g[:, 0], adj_g[:, 1], c=C_ADJ, s=14, alpha=0.9, zorder=3,
               label=f'adjacent ghost  (n={len(adj_g)})')
    if len(pbc_g):
        ax.scatter(pbc_g[:, 0], pbc_g[:, 1], c=C_PBC, s=20, marker='D',
                   alpha=0.95, zorder=3, label=f'PBC ghost  (n={len(pbc_g)})')
    ax.add_patch(mpatches.Rectangle(lo, c, c, fill=True, facecolor=C_CORE,
                                    alpha=0.10, edgecolor=C_CORE, lw=1.8))
    ax.add_patch(mpatches.Rectangle(lo - ghost_width, c + 2 * ghost_width,
                                    c + 2 * ghost_width, fill=False,
                                    edgecolor=C_GHOSTBOX, lw=1.4, ls='--',
                                    label='ghost buffer'))
    for v in (0.0, domain):
        ax.axhline(v, color=C_DOMAIN, lw=0.9, ls=':', alpha=0.6)
        ax.axvline(v, color=C_DOMAIN, lw=0.9, ls=':', alpha=0.6)
    pad = max(2.5 * ghost_width, 0.03 * domain)
    ax.set_xlim(-pad, domain + pad)
    ax.set_ylim(-pad, domain + pad)
    _legend(ax)

    # ── Panel 3: corrected cloud + displacement arrows ────────────────────
    ax = axes[2]
    _style_ax(ax, f'after K={k} — {int(moved.sum())} points moved '
                  f'(|d| > {thresh / rd_test:.2f} rd), '
                  f'{int(viol_after.sum())} violations remain')
    ax.scatter(pts_after[~moved, 0], pts_after[~moved, 1],
               c=C_CONTEXT, s=4, alpha=0.6, linewidths=0,
               label=f'unmoved  (n={int((~moved).sum())})')
    if moved.any():
        ax.scatter(pts_before[moved, 0], pts_before[moved, 1],
                   facecolors='none', edgecolors=INK_MUTED, s=22,
                   linewidths=0.8, zorder=3,
                   label=f'before  (n={int(moved.sum())})')
        ax.quiver(pts_before[moved, 0], pts_before[moved, 1],
                  disp[moved, 0], disp[moved, 1],
                  angles='xy', scale_units='xy', scale=1.0,
                  color=C_MOVED, width=0.0028, alpha=0.9, zorder=4)
        ax.scatter(pts_before[moved, 0] + disp[moved, 0],
                   pts_before[moved, 1] + disp[moved, 1],
                   c=C_MOVED, s=12, alpha=0.95, linewidths=0, zorder=5,
                   label='after (arrow head)')
        stats = (f'|d|/rd  mean={mag[moved].mean() / rd_test:.2f}  '
                 f'max={mag.max() / rd_test:.2f}')
        ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=7,
                color=INK, family='monospace', va='top',
                bbox=dict(boxstyle='round,pad=0.35', fc='#1e2a3a',
                          ec='#334455', alpha=0.9))
    ax.set_xlim(-0.02 * domain, 1.02 * domain)
    ax.set_ylim(-0.02 * domain, 1.02 * domain)
    _legend(ax)

    fig.suptitle(
        f'GridCorrector2D apply()   N={len(pts_before)}   K={k}   '
        f'{G}x{G} grid   domain={domain:.4f}   ghost_w={ghost_width:.4f}   |   '
        f'mean nn {nn_before.mean():.4f} -> {nn_after.mean():.4f}   '
        f'violations {int(viol_before.sum())} -> {int(viol_after.sum())}',
        color='#f0f6ff', fontsize=10, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
