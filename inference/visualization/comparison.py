"""
visualization/comparison.py
----------------------------
Standard comparison figure for one SPH timestep.

Layout: (1 + n_K) rows × 4 columns
  Row 0  (reference)  nonTV orig scatter | TV scatter | nn-hist orig/TV | RDF orig/TV
  Row i+1 (K=k_i)    model K scatter | displacement hist | nn-hist orig/model/TV | RDF orig/model/TV

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
C_DISP  = '#1abc9c'   # teal for displacement


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


def _draw_ref_hist(ax, nn_orig, nn_tv, title, shared_xlim=None):
    """2-curve histogram for the reference row (orig vs TV only)."""
    all_nn = np.concatenate([nn_orig, nn_tv])
    lo = max(0.0, all_nn.min() - (all_nn.max()-all_nn.min()) * 0.05)
    hi = all_nn.max() + (all_nn.max()-all_nn.min()) * 0.05
    if shared_xlim is not None:
        lo, hi = shared_xlim
    bins = np.linspace(lo, hi, 55)
    ax.hist(nn_orig, bins=bins, color=C_ORIG, alpha=0.45, histtype='stepfilled',
            label=f'nonTV orig   mean={nn_orig.mean():.4f}  CV={_cv(nn_orig):.3f}')
    ax.hist(nn_tv,   bins=bins, color=C_TV,   alpha=0.45, histtype='stepfilled',
            label=f'TV           mean={nn_tv.mean():.4f}  CV={_cv(nn_tv):.3f}')
    for nn, c in [(nn_orig, C_ORIG), (nn_tv, C_TV)]:
        ax.axvline(nn.mean(), color=c, lw=1.3, ls='--')
    ax.set_xlabel('min nn distance', fontsize=8); ax.set_ylabel('count', fontsize=8)
    ax.tick_params(labelsize=7); ax.set_xlim(lo, hi)
    ax.set_title(title, fontsize=8.5, pad=3)
    ax.legend(fontsize=7, loc='upper left', framealpha=0.85)


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


def _draw_ref_rdf(ax, r_orig, g_orig, r_tv, g_tv, title):
    """2-curve RDF for the reference row (orig vs TV only)."""
    ax.plot(r_orig, g_orig, color=C_ORIG, lw=1.3, ls=':', alpha=0.75, label='nonTV orig')
    ax.plot(r_tv,   g_tv,   color=C_TV,   lw=1.6, alpha=0.85, label='TV')
    ax.axhline(1.0, color='#aaa', lw=0.8, ls=':', zorder=0)
    ax.set_xlabel('r', fontsize=8); ax.set_ylabel('g(r)', fontsize=8)
    ax.set_xlim(0, r_tv[-1]); ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=7); ax.set_title(title, fontsize=8.5, pad=3)
    ax.legend(fontsize=7.5, framealpha=0.85)


def _draw_rdf(ax, r_orig, g_orig, r_m, g_m, r_tv, g_tv, title):
    ax.plot(r_orig, g_orig, color=C_ORIG,  lw=1.3, ls=':', alpha=0.75, label='nonTV orig')
    ax.plot(r_m,    g_m,    color=C_MODEL, lw=1.6, alpha=0.85, label='nonTV + model')
    ax.plot(r_tv,   g_tv,   color=C_TV,    lw=1.6, alpha=0.85, label='TV')
    ax.axhline(1.0, color='#aaa', lw=0.8, ls=':', zorder=0)
    ax.set_xlabel('r', fontsize=8); ax.set_ylabel('g(r)', fontsize=8)
    ax.set_xlim(0, r_m[-1]); ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=7); ax.set_title(title, fontsize=8.5, pad=3)
    ax.legend(fontsize=7.5, framealpha=0.85)


def _pbc_disp_mag(pts_orig, pts_corr, domain=1.0):
    """Per-particle displacement magnitude, PBC-corrected (no wraparound artifacts)."""
    d = pts_corr - pts_orig
    d -= np.round(d / domain) * domain
    return np.linalg.norm(d, axis=1)


def _draw_disp_hist(ax, pts_orig, pts_corr, title, rd=1.0, domain=1.0,
                    elapsed=None, color=C_MODEL, perf_info=None):
    """
    Histogram of per-particle PBC displacement magnitude, normalised by rd.

    perf_info (optional dict): keys cpu_ms, gpu_ms, vram_mb — shown as a
    highlighted performance box in the lower-right of the panel.
    """
    disp = _pbc_disp_mag(pts_orig, pts_corr, domain) / rd
    p95  = np.percentile(disp, 95)
    bins = np.linspace(0, max(disp.max() * 1.05, 1e-4), 60)
    ax.hist(disp, bins=bins, color=color, alpha=0.65, histtype='stepfilled')
    ax.axvline(disp.mean(), color=color,     lw=1.5, ls='--', label=f'mean = {disp.mean():.3f} rd')
    ax.axvline(p95,         color='#c0392b', lw=1.0, ls=':',  label=f'p95  = {p95:.3f} rd')
    time_str = f'\ntime = {elapsed:.2f} s' if elapsed is not None else ''
    stats = (f'mean = {disp.mean():.3f} rd\n'
             f'std  = {disp.std():.3f} rd\n'
             f'max  = {disp.max():.3f} rd{time_str}')
    ax.text(0.97, 0.95, stats, transform=ax.transAxes,
            fontsize=7, va='top', ha='right', family='monospace',
            bbox=dict(boxstyle='round', fc='white', alpha=0.85, lw=0.5))
    ax.set_xlabel('|displacement| / rd', fontsize=8)
    ax.set_ylabel('count', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=8.5, pad=3)
    ax.legend(fontsize=7, framealpha=0.85)

    if perf_info:
        cpu_ms  = perf_info.get('cpu_ms')
        gpu_ms  = perf_info.get('gpu_ms')
        vram_mb = perf_info.get('vram_mb')
        lines = ['── timing ──']
        if cpu_ms  is not None: lines.append(f'CPU  {cpu_ms:6.1f} ms')
        if gpu_ms  is not None: lines.append(f'GPU  {gpu_ms:6.1f} ms')
        if cpu_ms and gpu_ms:   lines.append(f'↑ {cpu_ms/gpu_ms:.1f}× speedup')
        if vram_mb is not None: lines.append(f'VRAM {vram_mb:.0f} MB')
        ax.text(0.03, 0.03, '\n'.join(lines), transform=ax.transAxes,
                fontsize=7.5, va='bottom', ha='left', family='monospace',
                color='white',
                bbox=dict(boxstyle='round,pad=0.4', fc='#1a252f', alpha=0.90, lw=0))


# ── main function ─────────────────────────────────────────────────────────────
def plot_comparison_frame(
    pts_nonTV:       np.ndarray,
    corrected_by_k:  Dict[int, np.ndarray],
    pts_tv:          np.ndarray,
    domain:          float = 1.0,
    timestep:        Optional[int] = None,
    grid:            int   = 6,
    scale:           float = 3.8,
    rd:              Optional[float] = None,
    time_by_k:       Optional[Dict[int, float]] = None,
    save_path:       Optional[str] = None,
    show:            bool  = False,
):
    """
    Parameters
    ----------
    corrected_by_k : dict mapping K -> (N,2) corrected positions
    rd             : rd_test value used for normalising displacement histogram
    time_by_k      : dict mapping K -> elapsed seconds for that correction pass
    """
    k_values  = sorted(corrected_by_k.keys())
    step_str  = f't = {timestep}' if timestep is not None else ''
    n_rows    = 1 + len(k_values)   # row 0 = reference, rows 1+ = per K
    rd_val    = rd if rd is not None else 1.0
    time_by_k = time_by_k or {}

    print('  Computing nn distances and RDFs...')
    nn_raw  = min_nn_pbc(pts_nonTV, domain)
    nn_tv   = min_nn_pbc(pts_tv,   domain)
    nn_by_k = {k: min_nn_pbc(v, domain) for k, v in corrected_by_k.items()}

    all_nn = np.concatenate([nn_raw, nn_tv] + list(nn_by_k.values()))
    vmin, vmax = np.percentile(all_nn, 1), np.percentile(all_nn, 99)
    xlim = (max(0.0, all_nn.min()*0.97), all_nn.max()*1.03)

    r_max = 0.02 * 4.0
    r_orig, g_orig = compute_rdf(pts_nonTV, r_max, domain=domain)
    r_tv,   g_tv   = compute_rdf(pts_tv,   r_max, domain=domain)
    rdf_by_k = {k: compute_rdf(v, r_max, domain=domain)
                for k, v in corrected_by_k.items()}

    fig, axes = plt.subplots(
        n_rows, 4,
        figsize=(22, 4 * n_rows),
        gridspec_kw={'width_ratios': [1, 1.2, 1.4, 1.4],
                     'hspace': 0.44, 'wspace': 0.30}
    )
    fig.subplots_adjust(left=0.04, right=0.99,
                        top=0.92, bottom=max(0.06, 0.12 - 0.01*n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # ── row 0: reference (nonTV orig + TV, no model) ──────────────────────────
    ax_orig, ax_tv_sc, ax_ref_hist, ax_ref_rdf = axes[0]
    _draw_scatter(ax_orig,  pts_nonTV, nn_raw, vmin, vmax,
                  f'nonTV original   {step_str}')
    sc_ref = _draw_scatter(ax_tv_sc, pts_tv, nn_tv, vmin, vmax,
                           f'TV   {step_str}')
    _draw_ref_hist(ax_ref_hist, nn_raw, nn_tv,
                   f'nn-dist: nonTV orig vs TV   {step_str}', shared_xlim=xlim)
    _draw_ref_rdf(ax_ref_rdf, r_orig, g_orig, r_tv, g_tv,
                  f'RDF: nonTV orig vs TV   {step_str}')

    # ── rows 1+: per K ────────────────────────────────────────────────────────
    sc_last = sc_ref
    for row_idx, k in enumerate(k_values, start=1):
        corr     = corrected_by_k[k]
        nn_k     = nn_by_k[k]
        r_k, g_k = rdf_by_k[k]
        ax_model, ax_disp, ax_hist, ax_rdf = axes[row_idx]

        sc_last = _draw_scatter(ax_model, corr, nn_k, vmin, vmax,
                                f'nonTV + model  K={k}   {step_str}')
        _draw_disp_hist(ax_disp, pts_nonTV, corr,
                        f'displacement  K={k}   {step_str}',
                        rd=rd_val, domain=domain,
                        elapsed=time_by_k.get(k))
        _draw_hist(ax_hist, nn_raw, nn_k, nn_tv,
                   f'nn-dist: K={k} vs TV   {step_str}', shared_xlim=xlim)
        _draw_rdf(ax_rdf, r_orig, g_orig, r_k, g_k, r_tv, g_tv,
                  f'RDF: K={k} vs TV   {step_str}')

    cbar_ax = fig.add_axes([0.04, 0.01, 0.38, 0.015])
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
    corrector,
    timestep:   Optional[int] = None,
    domain:     float = 1.0,
    figsize:    Optional[Tuple[float, float]] = None,
    save_path:  Optional[str] = None,
    show:       bool = True,
    rd:         Optional[float] = None,
    perf_data:  Optional[Dict] = None,
):
    """
    Five-row comparison at a single timestep:
      Row 0 — reference: nonTV original | TV | nn-hist orig/TV | RDF orig/TV
      Row 1 — K=1 standard   (1 call)
      Row 2 — K=2 standard   (2 calls)
      Row 3 — K=1+K=1 shifted grid  (2 calls, same cost as row 2)
      Row 4 — K=5 standard   (5 calls, upper bound)

    Each correction row: model scatter | displacement hist | nn-hist 3-way | RDF 3-way

    perf_data: optional dict keyed by strategy short-name ('k1', 'k2', 'shifted', 'k5'),
               each value a dict with optional keys: cpu_ms, gpu_ms, vram_mb.
               Shown as a highlighted box in each displacement panel.
    """
    import time as _time
    step_str  = f't = {timestep}' if timestep is not None else ''
    rd_val    = rd if rd is not None else (corrector.cfg.rd_test if hasattr(corrector.cfg, 'rd_test') else 1.0)
    perf_data = perf_data or {}

    print('  Applying correctors...')
    t0 = _time.perf_counter(); k1      = corrector.apply(pts_nonTV, k=1); t_k1  = _time.perf_counter() - t0
    t0 = _time.perf_counter(); k2      = corrector.apply(pts_nonTV, k=2); t_k2  = _time.perf_counter() - t0
    t0 = _time.perf_counter(); shifted = corrector.apply_shifted_grid(pts_nonTV, shift_fraction=0.5); t_sh = _time.perf_counter() - t0
    t0 = _time.perf_counter(); k5      = corrector.apply(pts_nonTV, k=5); t_k5  = _time.perf_counter() - t0

    print('  Computing nn and RDFs...')
    nn_raw = min_nn_pbc(pts_nonTV, domain)
    nn_tv  = min_nn_pbc(pts_tv,    domain)
    nn_k1  = min_nn_pbc(k1,        domain)
    nn_k2  = min_nn_pbc(k2,        domain)
    nn_sh  = min_nn_pbc(shifted,   domain)
    nn_k5  = min_nn_pbc(k5,        domain)

    all_nn = np.concatenate([nn_raw, nn_tv, nn_k1, nn_k2, nn_sh, nn_k5])
    vmin, vmax = np.percentile(all_nn, 1), np.percentile(all_nn, 99)
    xlim = (max(0.0, all_nn.min()*0.97), all_nn.max()*1.03)

    r_max = 0.02 * 4.0
    r_orig, g_orig = compute_rdf(pts_nonTV, r_max, domain=domain)
    r_tv,   g_tv   = compute_rdf(pts_tv,    r_max, domain=domain)
    r_k1,   g_k1   = compute_rdf(k1,        r_max, domain=domain)
    r_k2,   g_k2   = compute_rdf(k2,        r_max, domain=domain)
    r_sh,   g_sh   = compute_rdf(shifted,   r_max, domain=domain)
    r_k5,   g_k5   = compute_rdf(k5,        r_max, domain=domain)

    strategies = [
        (k1,      nn_k1, r_k1, g_k1, 'K=1  standard  (1 call)',           t_k1, perf_data.get('k1')),
        (k2,      nn_k2, r_k2, g_k2, 'K=2  standard  (2 calls)',          t_k2, perf_data.get('k2')),
        (shifted, nn_sh,  r_sh, g_sh, 'K=1+K=1  shifted grid  (2 calls)', t_sh, perf_data.get('shifted')),
        (k5,      nn_k5, r_k5, g_k5, 'K=5  standard  (5 calls)',          t_k5, perf_data.get('k5')),
    ]
    n_rows = 1 + len(strategies)

    if figsize is None:
        figsize = (22, 4 * n_rows)

    fig, axes = plt.subplots(
        n_rows, 4, figsize=figsize,
        gridspec_kw={'width_ratios': [1, 1.2, 1.4, 1.4],
                     'hspace': 0.44, 'wspace': 0.30}
    )
    fig.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.06)

    # ── row 0: reference ──────────────────────────────────────────────────────
    ax_orig, ax_tv_sc, ax_ref_hist, ax_ref_rdf = axes[0]
    _draw_scatter(ax_orig,  pts_nonTV, nn_raw, vmin, vmax,
                  f'nonTV original   {step_str}')
    sc_ref = _draw_scatter(ax_tv_sc, pts_tv, nn_tv, vmin, vmax,
                           f'TV   {step_str}')
    _draw_ref_hist(ax_ref_hist, nn_raw, nn_tv,
                   f'nn-dist: nonTV orig vs TV   {step_str}', shared_xlim=xlim)
    _draw_ref_rdf(ax_ref_rdf, r_orig, g_orig, r_tv, g_tv,
                  f'RDF: nonTV orig vs TV   {step_str}')

    # ── rows 1+: per strategy ─────────────────────────────────────────────────
    sc_last = sc_ref
    for row_idx, (corr, nn_corr, r_m, g_m, label, elapsed, perf_info) in enumerate(strategies, start=1):
        ax_model, ax_disp, ax_hist, ax_rdf = axes[row_idx]

        sc_last = _draw_scatter(ax_model, corr, nn_corr, vmin, vmax,
                                f'{label}   {step_str}')
        _draw_disp_hist(ax_disp, pts_nonTV, corr,
                        f'displacement   {label}',
                        rd=rd_val, domain=domain, elapsed=elapsed,
                        perf_info=perf_info)
        _draw_hist(ax_hist, nn_raw, nn_corr, nn_tv,
                   f'nn-dist: {label}', shared_xlim=xlim)
        _draw_rdf(ax_rdf, r_orig, g_orig, r_m, g_m, r_tv, g_tv,
                  f'RDF: {label}')

    cbar_ax = fig.add_axes([0.04, 0.01, 0.38, 0.012])
    cbar = fig.colorbar(sc_last, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('min nearest-neighbour distance', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    shift = 0.5 * corrector.tiling.cell_size
    vram_vals = [p.get('vram_mb') for p in perf_data.values() if p and p.get('vram_mb')]
    vram_str  = f'   |   peak VRAM {max(vram_vals):.0f} MB' if vram_vals else ''
    gpu_k5    = (perf_data.get('k5') or {}).get('gpu_ms')
    cpu_k5    = (perf_data.get('k5') or {}).get('cpu_ms')
    timing_str = ''
    if gpu_k5 is not None:
        timing_str = f'   |   K=5: GPU {gpu_k5:.1f} ms'
        if cpu_k5 is not None:
            timing_str += f' / CPU {cpu_k5:.0f} ms ({cpu_k5/gpu_k5:.0f}× speedup)'
    fig.suptitle(
        f'Shifted-grid strategy comparison   {step_str}'
        f'   |   shift = cell/2 = {shift:.3f}   |   N=2500   |   {corrector.cfg.n_cells}x{corrector.cfg.n_cells} grid'
        f'{timing_str}{vram_str}',
        fontsize=9.5, y=0.975)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Saved -> {save_path}')
    if show:
        plt.show()
    plt.close(fig)

    print(f'\nMean nn-distance at {step_str}:')
    for label, nn, cost in [
        ('K=1 standard',       nn_k1, '1 call '),
        ('K=2 standard',       nn_k2, '2 calls'),
        ('K=1+K=1 shifted',    nn_sh,  '2 calls'),
        ('K=5 standard',       nn_k5, '5 calls'),
        ('TV',                 nn_tv,  '------'),
    ]:
        pct = (nn.mean()/nn_tv.mean()-1)*100
        print(f'  [{cost}]  {label:22s}  {nn.mean():.5f}  ({pct:+.1f}% vs TV)')
    return fig
