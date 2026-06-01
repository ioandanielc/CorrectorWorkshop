"""
visualization/tiling.py
-----------------------
Four functions for inspecting the spatial tiling decomposition.

  plot_tiling_overview       — full domain coloured by tile + one tile highlighted
  plot_tile_detail           — zoom on one tile: core / adjacent ghost / PBC ghost
  plot_violation_proof       — a PBC-violating pair and how the ghost buffer captures it
  plot_shifted_grid_explainer — diagram explaining the two-pass shifted-grid strategy

All functions accept save_path and show for flexible use in notebooks,
scripts, and automated experiment runners.
"""
from typing import Optional, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from inference.pipeline.pbc import build_ghost_tile, pbc_dists
from inference.pipeline.tiling import TilingConfig, iter_tiles


C_CORE     = '#60a5fa'   # blue
C_ADJ      = '#f5924e'   # orange
C_PBC      = '#a78bfa'   # purple
C_TILE_BOX = '#60a5fa'
C_GHOST_BOX= '#facc15'
C_DOMAIN   = '#4ade80'
BG_DARK    = '#141821'
GRID_COLOR = '#334455'


def _classify_ghosts(ext, is_core, domain=1.0):
    """Split ghost particles into adjacent-tile vs PBC-wrapped."""
    ghost = ext[~is_core]
    is_pbc = (ghost.min(axis=1) < -1e-6) | (ghost.max(axis=1) > domain + 1e-6)
    return ghost[~is_pbc], ghost[is_pbc]


def _tile_bounds(tc, tile_ij):
    c  = tc.cell_size
    lo = np.array([tile_ij[0]*c, tile_ij[1]*c])
    return lo, lo + c


def _style_ax(ax):
    ax.set_facecolor(BG_DARK)
    for sp in ax.spines.values():
        sp.set_color('#1e2a3a')
    ax.tick_params(colors='#475569')


# ── 1. Full-domain tiling overview ────────────────────────────────────────────
def plot_tiling_overview(
    pts:            np.ndarray,
    tc:             TilingConfig,
    rd_test:        float,
    highlight_tile: Tuple[int, int] = (0, 0),
    domain:         float = 1.0,
    figsize:        Tuple[float, float] = (14, 6),
    save_path:      Optional[str] = None,
    show:           bool = True,
):
    """
    Two-panel figure:
      Left  — full domain, particles coloured by tile ID, one tile highlighted
              with its ghost buffer boundary.
      Right — zoom on the highlighted tile; particles colour-coded as
              core (blue) / adjacent ghost (orange) / PBC ghost (purple).

    Parameters
    ----------
    pts            : (N, 2) particle positions in [0, domain)^2
    tc             : TilingConfig
    rd_test        : float  the SPH minimum distance (used to compute ghost_width)
    highlight_tile : (i, j) grid indices of the tile to zoom on
    """
    G, c  = tc.grid_size, tc.cell_size
    gw    = tc.ghost_width(rd_test)
    lo, hi = _tile_bounds(tc, highlight_tile)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor('#0f1117')

    # ── Panel 1: full domain ─────────────────────────────────────────────
    ax = axes[0]; _style_ax(ax)
    tile_ij = np.floor(pts / c).astype(int).clip(0, G-1)
    tile_id = tile_ij[:, 0] + tile_ij[:, 1] * G
    ax.scatter(pts[:, 0], pts[:, 1],
               c=tile_id, cmap='tab20', s=1.5, alpha=0.55, linewidths=0)
    for k in range(G+1):
        v = k * c
        ax.axhline(v, color=GRID_COLOR, lw=0.8)
        ax.axvline(v, color=GRID_COLOR, lw=0.8)
    ax.add_patch(mpatches.Rectangle(lo, c, c,
        fill=True, facecolor=C_TILE_BOX, alpha=0.15,
        edgecolor=C_TILE_BOX, lw=2))
    ax.add_patch(mpatches.Rectangle(lo - gw, c + 2*gw, c + 2*gw,
        fill=False, edgecolor=C_GHOST_BOX, lw=1.5, ls='--'))
    ti, tj = highlight_tile
    ax.annotate(f'tile ({ti},{tj})', (lo[0]+c/2, lo[1]+c/2),
        color=C_TILE_BOX, fontsize=9, ha='center', va='center', fontweight='bold')
    ax.annotate(f'ghost buffer\ng={gw:.3f}', (lo[0]+c/2, lo[1]-gw*2.5),
        color=C_GHOST_BOX, fontsize=8, ha='center')
    ax.text(0.02, 0.97,
        f'N={len(pts)} | {G}x{G} = {G**2} tiles | ~{round(len(pts)/G**2)} pts/tile',
        transform=ax.transAxes, color='#94a3b8', fontsize=8, va='top')
    ax.set_xlim(-0.01, domain+0.01); ax.set_ylim(-0.01, domain+0.01)
    ax.set_aspect('equal')
    ax.set_title(f'{G}x{G} grid — particles coloured by tile ID',
                 color='#e2e8f0', pad=6)

    # ── Panel 2: tile zoom ───────────────────────────────────────────────
    ax = axes[1]; _style_ax(ax)
    ext, is_core, orig_idx = build_ghost_tile(pts, lo, hi, gw, domain)
    adj_pts, pbc_pts = _classify_ghosts(ext, is_core, domain)
    core_pts = ext[is_core]

    ax.scatter(adj_pts[:, 0], adj_pts[:, 1],
        c=C_ADJ, s=20, alpha=0.85, zorder=3,
        label=f'adjacent ghost  (n={len(adj_pts)})')
    if len(pbc_pts):
        ax.scatter(pbc_pts[:, 0], pbc_pts[:, 1],
            c=C_PBC, s=25, alpha=0.9, zorder=3, marker='D',
            label=f'PBC ghost  (n={len(pbc_pts)})')
        orig_pbc = pts[orig_idx[~is_core][(ext[~is_core].min(axis=1) < -1e-6) |
                                           (ext[~is_core].max(axis=1) > domain+1e-6)]]
        for gpt, rpt in zip(pbc_pts[:6], orig_pbc[:6]):
            ax.annotate('', xy=gpt, xytext=rpt,
                arrowprops=dict(arrowstyle='->', color=C_PBC,
                                lw=0.8, linestyle='dashed', alpha=0.5))
    ax.scatter(core_pts[:, 0], core_pts[:, 1],
        c=C_CORE, s=22, alpha=0.9, zorder=4, label=f'core  (n={len(core_pts)})')
    ax.add_patch(mpatches.Rectangle(lo, c, c, fill=False,
                                     edgecolor=C_TILE_BOX, lw=2.0))
    ax.add_patch(mpatches.Rectangle(lo - gw, c + 2*gw, c + 2*gw,
        fill=False, edgecolor=C_GHOST_BOX, lw=1.5, ls='--'))
    for v in [0, domain]:
        ax.axhline(v, color=C_DOMAIN, lw=0.9, ls=':', alpha=0.5)
        ax.axvline(v, color=C_DOMAIN, lw=0.9, ls=':', alpha=0.5)
    margin = gw * 3
    ax.set_xlim(lo[0]-margin, hi[0]+margin)
    ax.set_ylim(lo[1]-margin, hi[1]+margin)
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='upper right',
              facecolor='#1e2a3a', labelcolor='#e2e8f0', edgecolor='#334455')
    ax.set_title(f'Tile ({ti},{tj}) | solid=tile | dashed=ghost buffer | '
                 f'green=domain edge', color='#e2e8f0', pad=6)

    fig.suptitle('Ghost buffer: core (blue) | adjacent ghost (orange) | '
                 'PBC-wrapped ghost (purple)',
                 color='#f0f6ff', fontsize=10, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ── 2. Single tile detail ─────────────────────────────────────────────────────
def plot_tile_detail(
    pts:       np.ndarray,
    tc:        TilingConfig,
    rd_test:   float,
    tile_ij:   Tuple[int, int] = (0, 0),
    domain:    float = 1.0,
    figsize:   Tuple[float, float] = (7, 6),
    save_path: Optional[str] = None,
    show:      bool = True,
):
    """Zoomed view of one tile: core / adjacent ghost / PBC ghost."""
    gw = tc.ghost_width(rd_test)
    lo, hi = _tile_bounds(tc, tile_ij)
    c = tc.cell_size

    ext, is_core, orig_idx = build_ghost_tile(pts, lo, hi, gw, domain)
    adj_pts, pbc_pts = _classify_ghosts(ext, is_core, domain)
    core_pts = ext[is_core]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#0f1117'); _style_ax(ax)

    ax.scatter(adj_pts[:, 0], adj_pts[:, 1],
        c=C_ADJ, s=20, alpha=0.85, zorder=3, label=f'adjacent ghost ({len(adj_pts)})')
    if len(pbc_pts):
        ax.scatter(pbc_pts[:, 0], pbc_pts[:, 1],
            c=C_PBC, s=25, alpha=0.9, zorder=3, marker='D',
            label=f'PBC ghost ({len(pbc_pts)})')
    ax.scatter(core_pts[:, 0], core_pts[:, 1],
        c=C_CORE, s=22, alpha=0.9, zorder=4, label=f'core ({len(core_pts)})')
    ax.add_patch(mpatches.Rectangle(lo, c, c, fill=False, edgecolor=C_TILE_BOX, lw=2))
    ax.add_patch(mpatches.Rectangle(lo-gw, c+2*gw, c+2*gw,
        fill=False, edgecolor=C_GHOST_BOX, lw=1.5, ls='--'))
    for v in [0, domain]:
        ax.axhline(v, color=C_DOMAIN, lw=0.9, ls=':', alpha=0.5)
        ax.axvline(v, color=C_DOMAIN, lw=0.9, ls=':', alpha=0.5)
    margin = gw * 3
    ax.set_xlim(lo[0]-margin, hi[0]+margin)
    ax.set_ylim(lo[1]-margin, hi[1]+margin)
    ax.set_aspect('equal')
    ax.legend(fontsize=9, facecolor='#1e2a3a', labelcolor='#e2e8f0', edgecolor='#334455')
    ti, tj = tile_ij
    ax.set_title(f'Tile ({ti},{tj}) detail', color='#e2e8f0', pad=6)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ── 3. Violation pair proof ───────────────────────────────────────────────────
def plot_violation_proof(
    pts:       np.ndarray,
    D:         Optional[np.ndarray],
    tc:        TilingConfig,
    rd_test:   float,
    pair:      Optional[Tuple[int, int]] = None,
    domain:    float = 1.0,
    figsize:   Tuple[float, float] = (14, 5.5),
    save_path: Optional[str] = None,
    show:      bool = True,
):
    """
    Two-panel figure showing a PBC-violating pair and how the ghost buffer
    captures both endpoints in one tile.

    Parameters
    ----------
    D    : (N, N) PBC distance matrix. If None, computed from pts.
    pair : (i, j) indices. If None, uses the closest violating pair.
    """
    if D is None:
        D = pbc_dists(pts, domain)
        np.fill_diagonal(D, np.inf)
    if pair is None:
        ii, jj = np.unravel_index(np.argmin(D), D.shape)
    else:
        ii, jj = pair

    gw = tc.ghost_width(rd_test)
    c, G = tc.cell_size, tc.grid_size
    tile_lo_ij = np.floor(pts[ii] / c).astype(int).clip(0, G-1)
    lo_v = tile_lo_ij * c; hi_v = lo_v + c

    ext_v, is_core_v, oidx_v = build_ghost_tile(pts, lo_v, hi_v, gw, domain)
    ghost_v = ext_v[~is_core_v]
    is_pbc_v = (ghost_v.min(axis=1) < -1e-6) | (ghost_v.max(axis=1) > domain+1e-6)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor('#0f1117')

    for ax_idx, ax in enumerate(axes):
        _style_ax(ax)
        ax.scatter(pts[:, 0], pts[:, 1], c='#2a3a4a', s=1, alpha=0.3, zorder=1)

        if ax_idx == 0:
            ax.scatter(*pts[ii], c='#f87171', s=90, zorder=5, marker='*',
                label=f'p_{ii}  ({pts[ii,0]:.3f}, {pts[ii,1]:.3f})')
            ax.scatter(*pts[jj], c='#fb923c', s=90, zorder=5, marker='*',
                label=f'p_{jj}  ({pts[jj,0]:.3f}, {pts[jj,1]:.3f})')
            diff_pbc = (pts[jj]-pts[ii]) - np.round(pts[jj]-pts[ii])
            mid = pts[ii] + diff_pbc/2
            ax.annotate('', xy=pts[ii]+diff_pbc, xytext=pts[ii],
                arrowprops=dict(arrowstyle='<->', color='#f87171', lw=2))
            ax.text(mid[0], mid[1]+0.01, f'd_PBC={D[ii,jj]:.5f}',
                color='#f87171', fontsize=7.5, ha='center')
            ax.add_patch(mpatches.Rectangle(lo_v, c, c, fill=True,
                facecolor=C_TILE_BOX, alpha=0.12, edgecolor=C_TILE_BOX, lw=2))
            ax.add_patch(mpatches.Rectangle(lo_v-gw, c+2*gw, c+2*gw,
                fill=False, edgecolor=C_GHOST_BOX, lw=1.5, ls='--'))
            for k in range(G+1):
                v = k*c
                ax.axhline(v, color=GRID_COLOR, lw=0.6)
                ax.axvline(v, color=GRID_COLOR, lw=0.6)
            ax.set_xlim(-0.01, domain+0.01); ax.set_ylim(-0.01, domain+0.01)
            ax.legend(fontsize=7.5, facecolor='#1e2a3a',
                      labelcolor='#e2e8f0', edgecolor='#334455')
            ax.set_title('Full domain: violating pair + owning tile',
                         color='#e2e8f0', pad=6)

        else:
            adj_v = ghost_v[~is_pbc_v]; pbc_v = ghost_v[is_pbc_v]
            core_v = ext_v[is_core_v]
            ax.scatter(adj_v[:, 0], adj_v[:, 1], c=C_ADJ, s=16, alpha=0.75, zorder=3,
                label=f'adjacent ghost ({len(adj_v)})')
            if len(pbc_v):
                ax.scatter(pbc_v[:, 0], pbc_v[:, 1], c=C_PBC, s=20, alpha=0.9,
                    zorder=3, marker='D', label=f'PBC ghost ({len(pbc_v)})')
            ax.scatter(core_v[:, 0], core_v[:, 1], c=C_CORE, s=18, alpha=0.85,
                zorder=4, label=f'core ({len(core_v)})')
            for idx_p, col_p in [(ii, '#f87171'), (jj, '#fb923c')]:
                hit = np.where(oidx_v == idx_p)[0]
                if len(hit):
                    lbl = f'p_{idx_p} ({"core" if is_core_v[hit[0]] else "ghost"})'
                    ax.scatter(ext_v[hit, 0], ext_v[hit, 1], c=col_p, s=160,
                        zorder=6, marker='*', edgecolors='white', linewidths=0.5,
                        label=lbl)
            hits_ii = ext_v[oidx_v == ii]; hits_jj = ext_v[oidx_v == jj]
            if len(hits_ii) and len(hits_jj):
                ax.annotate('', xy=hits_ii[0], xytext=hits_jj[0],
                    arrowprops=dict(arrowstyle='<->', color='#f87171', lw=2))
                mid2 = (hits_ii[0] + hits_jj[0]) / 2
                ax.text(mid2[0], mid2[1]+0.003, f'd={D[ii,jj]:.5f}',
                    color='#f87171', fontsize=7.5, ha='center')
            ax.add_patch(mpatches.Rectangle(lo_v, c, c,
                fill=False, edgecolor=C_TILE_BOX, lw=2))
            ax.add_patch(mpatches.Rectangle(lo_v-gw, c+2*gw, c+2*gw,
                fill=False, edgecolor=C_GHOST_BOX, lw=1.5, ls='--'))
            for v in [0, domain]:
                ax.axhline(v, color=C_DOMAIN, lw=0.8, ls=':', alpha=0.5)
                ax.axvline(v, color=C_DOMAIN, lw=0.8, ls=':', alpha=0.5)
            m = gw * 4
            ax.set_xlim(lo_v[0]-m, hi_v[0]+m); ax.set_ylim(lo_v[1]-m, hi_v[1]+m)
            ax.legend(fontsize=7.5, facecolor='#1e2a3a',
                      labelcolor='#e2e8f0', edgecolor='#334455')
            ax.set_title('Tile zoom: both violation endpoints captured',
                         color='#e2e8f0', pad=6)

        ax.set_aspect('equal')

    j_vis = jj in oidx_v
    fig.suptitle(
        f'Correctness proof: violating pair (p_{ii}, p_{jj}) '
        f'd_PBC={D[ii,jj]:.5f} — both visible in one tile',
        color='#f0f6ff', fontsize=10, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f'p_{ii} in tile: {ii in oidx_v}  |  p_{jj} in ghost buffer: {j_vis}')
    return fig


# ── 4. Shifted-grid strategy explainer ───────────────────────────────────────
def plot_shifted_grid_explainer(
    grid_size:      int   = 3,
    shift_fraction: float = 0.5,
    domain:         float = 1.0,
    figsize:        Tuple[float, float] = (13, 5.5),
    save_path:      Optional[str] = None,
    show:           bool = True,
):
    """
    Two-panel diagram explaining the shifted-grid correction strategy.

      Left  — Pass 1: standard grid.  A marginal particle (star) sits within
               ghost_width of a tile boundary and receives an asymmetric context.
      Right — Pass 2: grid shifted by cell_size/2.  The same particle is now
               well inside a tile; its context is fully symmetric.

    Parameters
    ----------
    grid_size      : number of tiles per dimension (shown in diagram)
    shift_fraction : fraction of cell_size used as the shift (default 0.5)
    """
    c      = domain / grid_size
    shift  = shift_fraction * c
    gw     = 0.02           # representative ghost_width for illustration

    # A marginal particle near a vertical boundary
    bx     = c              # boundary at x = c
    margin = gw * 0.6       # how close the particle is to the boundary
    px, py = bx - margin, c * 1.3

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor('#0f1117')

    for ax_idx, ax in enumerate(axes):
        ax.set_facecolor(BG_DARK)
        ax.set_xlim(-0.01, domain + 0.01)
        ax.set_ylim(-0.01, domain + 0.01)
        ax.set_aspect('equal')
        for sp in ax.spines.values(): sp.set_color('#1e2a3a')
        ax.tick_params(colors='#475569')

        # Draw Pass 1 grid (always shown for reference)
        for k in range(grid_size + 1):
            v = k * c
            lw, ls, alpha = (1.8, '-', 0.9) if ax_idx == 0 else (0.8, '--', 0.35)
            col = '#60a5fa' if ax_idx == 0 else '#334455'
            ax.axhline(v, color=col, lw=lw, ls=ls, alpha=alpha)
            ax.axvline(v, color=col, lw=lw, ls=ls, alpha=alpha)

        if ax_idx == 1:
            # Draw Pass 2 grid (shifted)
            offsets = [(shift + k * c) % domain for k in range(grid_size)]
            for v in offsets:
                ax.axhline(v, color='#f5924e', lw=1.8, alpha=0.9)
                ax.axvline(v, color='#f5924e', lw=1.8, alpha=0.9)

        # Ghost buffer of the left tile (Pass 1) around x = c
        if ax_idx == 0:
            # Shade the marginal zone [c - gw, c) — particles here get
            # asymmetric context in Pass 1
            ax.axvspan(bx - gw, bx, color='#f87171', alpha=0.18,
                       label='marginal zone  (within ghost buffer of boundary)')
            ax.axvspan(bx, bx + gw, color='#f87171', alpha=0.10)

        # The marginal particle
        ax.scatter([px], [py], color='#facc15', s=200, zorder=6,
                   marker='*', edgecolors='white', linewidths=0.5,
                   label=f'marginal particle\n(dist to P1 boundary = {margin:.3f})')

        if ax_idx == 1:
            # Distance to nearest Pass-2 boundary
            nearest_p2 = min(abs(px - v) for v in
                             [(shift + k * c) % domain for k in range(grid_size)])
            ax.annotate(
                f'dist to P2 boundary\n= {nearest_p2:.3f} >> gw={gw}',
                xy=(px, py), xytext=(px + 0.15, py - 0.06),
                color='#4ade80', fontsize=8,
                arrowprops=dict(arrowstyle='->', color='#4ade80', lw=1.2),
            )
            ax.scatter([px], [py], color='#4ade80', s=200, zorder=7,
                       marker='*', edgecolors='white', linewidths=0.5,
                       label=f'same particle — now inside\na Pass-2 tile (dist={nearest_p2:.3f})')

        # Label the shift arrow in the right panel header
        if ax_idx == 1:
            ax.set_title(
                f'Pass 2: grid shifted by cell/2 = {shift:.3f}\n'
                f'(orange lines = new boundaries)',
                color='#e2e8f0', pad=6)
        else:
            ax.set_title(
                f'Pass 1: standard {grid_size}×{grid_size} grid\n'
                f'(blue lines = tile boundaries)',
                color='#e2e8f0', pad=6)

        ax.legend(fontsize=8, loc='upper right',
                  facecolor='#1e2a3a', labelcolor='#e2e8f0',
                  edgecolor='#334455')

    # Shift arrow between panels (figure-level annotation)
    fig.text(0.50, 0.50, f'→  shift = {shift:.3f}  →',
             ha='center', va='center', color='#facc15',
             fontsize=11, fontweight='bold')

    fig.suptitle(
        'Shifted-grid strategy: Pass 1 (standard) + Pass 2 (grid offset by cell/2)\n'
        'Every marginal particle is well inside at least one tile.',
        color='#f0f6ff', fontsize=10, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
