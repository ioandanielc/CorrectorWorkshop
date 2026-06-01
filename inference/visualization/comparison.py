"""
visualization/comparison.py
----------------------------
Standard comparison figure for one SPH timestep.

Layout: 4 rows (one per K value) x 5 columns:
  Col 0  scatter — nonTV original (no correction)
  Col 1  scatter — nonTV + model at K passes
  Col 2  scatter — TV variant
  Col 3  nn-distance histogram overlay (3 curves)
  Col 4  RDF overlay (3 curves)

All scatter panels share a colormap range.
All histogram panels share an x-axis.
No rd reference lines.

Usage
-----
    from inference.visualization.comparison import plot_comparison_frame

    fig = plot_comparison_frame(
        pts_nonTV, {1: corr_k1, 2: corr_k2, 3: corr_k3, 5: corr_k5},
        pts_tv, timestep=300, grid=6, scale=3.8,
        save_path='experiments/exp_xxx/frames/t0300.png'
    )
"""
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from inference.pipeline.pbc import min_nn_pbc, compute_rdf

C_ORIG  = '#888888'
C_MODEL = '#2471a3'   # blue
C_TV    = '#e67e22'   # orange


# ── per-panel helpers ─────────────────────────────────────────────────────────
def _draw_scatter(ax, pts, nn, vmin, vmax, title):
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=nn,
                    cmap='plasma_r', vmin=vmin, vmax=vmax,
                    s=3, alpha=0.65, linewidths=0, rasterized=True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('x', fontsize=8); ax.set_ylabel('y', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=8.5, pad=3)
    return sc


def _cv(nn):
    return float(nn.std() / nn.mean())


def _draw_hist(ax, nn_orig, nn_model, nn_tv, title, shared_xlim=None):
    all_nn = np.concatenate([nn_orig, nn_model, nn_tv])
    lo = max(0.0, all_nn.min() - (all_nn.max()-all_nn.min()) * 0.05)
    hi = all_nn.max() + (all_nn.max()-all_nn.min()) * 0.05
    if shared_xlim is not None:
        lo, hi = shared_xlim
    bins = np.linspace(lo, hi, 55)
    ax.hist(nn_orig,  bins=bins, color=C_ORIG,  alpha=0.45, histtype='stepfilled',
            label=f'nonTV orig   mean={nn_orig.mean():.4f}  CV={_cv(nn_orig):.3f}')
    ax.hist(nn_model, bins=bins, color=C_MODEL, alpha=0.55, histtype='stepfilled',
            label=f'nonTV+model  mean={nn_model.mean():.4f}  CV={_cv(nn_model):.3f}')
    ax.hist(nn_tv,    bins=bins, color=C_TV,    alpha=0.45, histtype='stepfilled',
            label=f'TV           mean={nn_tv.mean():.4f}  CV={_cv(nn_tv):.3f}')
    for nn, c in [(nn_orig,C_ORIG),(nn_model,C_MODEL),(nn_tv,C_TV)]:
        ax.axvline(nn.mean(), color=c, lw=1.3, ls='--')
    ax.set_xlabel('min nn distance', fontsize=8); ax.set_ylabel('count', fontsize=8)
    ax.tick_params(labelsize=7); ax.set_xlim(lo, hi)
    ax.set_title(title, fontsize=8.5, pad=3)
    ax.legend(fontsize=7, loc='upper left', framealpha=0.85)


def _draw_rdf(ax, r_orig, g_orig, r_m, g_m, r_tv, g_tv, title):
    ax.plot(r_orig, g_orig, color=C_ORIG,  lw=1.3, ls=':', alpha=0.75, label='nonTV orig')
    ax.plot(r_m,    g_m,    color=C_MODEL, lw=1.6, alpha=0.85, label='nonTV + model')
    ax.plot(r_tv,   g_tv,   color=C_TV,    lw=1.6, alpha=0.85, label='TV')
    ax.axhline(1.0, color='#aaa', lw=0.8, ls=':', zorder=0)
    ax.set_xlabel('r', fontsize=8); ax.set_ylabel('g(r)', fontsize=8)
    ax.set_xlim(0, r_m[-1]); ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=7); ax.set_title(title, fontsize=8.5, pad=3)
    ax.legend(fontsize=7.5, framealpha=0.85)


# ── main function ─────────────────────────────────────────────────────────────
def plot_comparison_frame(
    pts_nonTV:       np.ndarray,
    corrected_by_k:  Dict[int, np.ndarray],
    pts_tv:          np.ndarray,
    domain:          float = 1.0,
    timestep:        Optional[int] = None,
    grid:            int   = 6,
    scale:           float = 3.8,
    save_path:       Optional[str] = None,
    show:            bool  = False,
):
    """
    Parameters
    ----------
    corrected_by_k : dict mapping K -> (N,2) corrected positions
                     e.g. {1: corr_k1, 2: corr_k2, 3: corr_k3, 5: corr_k5}
    """
    k_values = sorted(corrected_by_k.keys())
    step_str = f't = {timestep}' if timestep is not None else ''

    print('  Computing nn distances and RDFs...')
    nn_raw = min_nn_pbc(pts_nonTV, domain)
    nn_tv  = min_nn_pbc(pts_tv,   domain)
    nn_by_k = {k: min_nn_pbc(v, domain) for k, v in corrected_by_k.items()}

    all_nn = np.concatenate([nn_raw, nn_tv] + list(nn_by_k.values()))
    vmin, vmax = np.percentile(all_nn, 1), np.percentile(all_nn, 99)
    xlim = (max(0.0, all_nn.min()*0.97), all_nn.max()*1.03)

    r_max = 0.02 * 4.0   # 4 * typical rd_test
    r_orig, g_orig = compute_rdf(pts_nonTV, r_max, domain=domain)
    r_tv,   g_tv   = compute_rdf(pts_tv,   r_max, domain=domain)
    rdf_by_k = {k: compute_rdf(v, r_max, domain=domain)
                for k, v in corrected_by_k.items()}

    n_rows = len(k_values)
    fig, axes = plt.subplots(
        n_rows, 5, figsize=(27, 4*n_rows),
        gridspec_kw={'width_ratios':[1,1,1,1.4,1.4], 'hspace':0.44, 'wspace':0.30}
    )
    fig.subplots_adjust(left=0.04, right=0.99,
                        top=0.92, bottom=max(0.06, 0.12 - 0.01*n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    sc_last = None
    for row_idx, k in enumerate(k_values):
        corr   = corrected_by_k[k]
        nn_k   = nn_by_k[k]
        r_k, g_k = rdf_by_k[k]
        ax_raw, ax_model, ax_tv, ax_hist, ax_rdf = axes[row_idx]

        _draw_scatter(ax_raw,   pts_nonTV, nn_raw, vmin, vmax,
                      f'nonTV original   {step_str}')
        sc_last = _draw_scatter(ax_model, corr, nn_k, vmin, vmax,
                      f'nonTV + model  K={k}   {step_str}')
        _draw_scatter(ax_tv,    pts_tv,   nn_tv,  vmin, vmax,
                      f'TV   {step_str}')
        _draw_hist(ax_hist, nn_raw, nn_k, nn_tv,
                   f'nn-dist: K={k} vs TV   {step_str}', shared_xlim=xlim)
        _draw_rdf(ax_rdf, r_orig, g_orig, r_k, g_k, r_tv, g_tv,
                  f'RDF: K={k} vs TV   {step_str}')

    cbar_ax = fig.add_axes([0.04, 0.01, 0.56, 0.018])
    cbar = fig.colorbar(sc_last, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('min nearest-neighbour distance', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f'SPH corrector   {step_str}   |   N=2500   |   '
        f'{grid}x{grid} grid   scale s={scale:.1f}',
        fontsize=10, y=0.975,
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved -> {save_path}')
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ── Shifted-grid comparison ────────────────────────────────────────────────────
def plot_shifted_grid_comparison(
    pts_nonTV:  np.ndarray,
    pts_tv:     np.ndarray,
    corrector,                        # Corrector instance
    timestep:   Optional[int] = None,
    domain:     float = 1.0,
    figsize:    Tuple[float, float] = (27, 8),
    save_path:  Optional[str] = None,
    show:       bool = True,
):
    """
    Three-row comparison at a single timestep:
      Row 0 — K=1 standard grid          (one pass, baseline)
      Row 1 — shifted K=1+K=1            (two passes, different grids)
      Row 2 — K=5 standard grid          (five passes, best single-strategy)

    Columns: nonTV original | corrected | TV | nn-hist | RDF
    """
    step_str = f't = {timestep}' if timestep is not None else ''

    print('  Applying correctors...')
    k1         = corrector.apply(pts_nonTV, k=1)
    shifted    = corrector.apply_shifted_grid(pts_nonTV, shift_fraction=0.5)
    k5         = corrector.apply(pts_nonTV, k=5)

    print('  Computing nn and RDFs...')
    nn_raw    = min_nn_pbc(pts_nonTV, domain)
    nn_tv     = min_nn_pbc(pts_tv,    domain)
    nn_k1     = min_nn_pbc(k1,        domain)
    nn_sh     = min_nn_pbc(shifted,   domain)
    nn_k5     = min_nn_pbc(k5,        domain)

    all_nn = np.concatenate([nn_raw, nn_tv, nn_k1, nn_sh, nn_k5])
    vmin, vmax = np.percentile(all_nn, 1), np.percentile(all_nn, 99)
    xlim = (max(0.0, all_nn.min()*0.97), all_nn.max()*1.03)

    r_max = 0.02 * 4.0
    r_orig, g_orig = compute_rdf(pts_nonTV, r_max, domain=domain)
    r_tv,   g_tv   = compute_rdf(pts_tv,    r_max, domain=domain)
    r_k1,   g_k1   = compute_rdf(k1,        r_max, domain=domain)
    r_sh,   g_sh   = compute_rdf(shifted,   r_max, domain=domain)
    r_k5,   g_k5   = compute_rdf(k5,        r_max, domain=domain)

    rows = [
        (k1,     nn_k1, r_k1, g_k1, 'K=1  standard grid'),
        (shifted, nn_sh, r_sh, g_sh, 'K=1+K=1  shifted grid  (shift=cell/2)'),
        (k5,     nn_k5, r_k5, g_k5, 'K=5  standard grid'),
    ]

    fig, axes = plt.subplots(3, 5, figsize=figsize,
        gridspec_kw={'width_ratios':[1,1,1,1.4,1.4], 'hspace':0.44, 'wspace':0.30})
    fig.subplots_adjust(left=0.04, right=0.99, top=0.92, bottom=0.08)

    sc_last = None
    for row_idx, (corr, nn_corr, r_m, g_m, label) in enumerate(rows):
        ax_raw, ax_model, ax_tv, ax_hist, ax_rdf = axes[row_idx]
        _draw_scatter(ax_raw,   pts_nonTV, nn_raw,  vmin, vmax,
                      f'nonTV original   {step_str}')
        sc_last = _draw_scatter(ax_model, corr, nn_corr, vmin, vmax,
                      f'{label}   {step_str}')
        _draw_scatter(ax_tv,    pts_tv,   nn_tv,  vmin, vmax,
                      f'TV   {step_str}')
        _draw_hist(ax_hist, nn_raw, nn_corr, nn_tv,
                   f'nn-dist: {label}', shared_xlim=xlim)
        _draw_rdf(ax_rdf, r_orig, g_orig, r_m, g_m, r_tv, g_tv,
                  f'RDF: {label}')

    cbar_ax = fig.add_axes([0.04, 0.01, 0.56, 0.018])
    cbar = fig.colorbar(sc_last, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('min nearest-neighbour distance', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    shift = 0.5 * corrector.tiling.cell_size
    fig.suptitle(
        f'Shifted-grid strategy comparison   {step_str}'
        f'   |   shift = cell/2 = {shift:.3f}   |   N=2500   |   {corrector.cfg.grid_size}x{corrector.cfg.grid_size} grid',
        fontsize=10, y=0.975)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved -> {save_path}')
    if show:
        plt.show()
    plt.close(fig)

    print(f'\nMean nn-distance at {step_str}:')
    for label, nn in [('K=1 standard', nn_k1), ('K=1+K=1 shifted', nn_sh),
                      ('K=5 standard', nn_k5), ('TV', nn_tv)]:
        print(f'  {label:20s}  {nn.mean():.5f}  '
              f'(+{(nn.mean()/nn_tv.mean()-1)*100:.1f}% vs TV)')
    return fig
