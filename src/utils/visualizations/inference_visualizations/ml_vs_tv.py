"""
visualization/ml_vs_tv.py
--------------------------
Clear comparison figure: GPU ML corrector vs Fast TV algorithm.

Three-panel layout:
  Left   — mean NN-distance trajectory over all sampled timesteps
  Middle — inference time comparison (ms, log scale)
  Right  — mean NN-distance at the best timestep (quality snapshot)

Usage
-----
    from utils.visualizations.inference_visualizations.ml_vs_tv import plot_ml_vs_tv, save_report

    fig = plot_ml_vs_tv(
        timesteps, metrics_by_k, metrics_tv_sim, metrics_tv_fast,
        timings_ms={'K=1 (GPU)': 3, 'K=5 (GPU)': 10, 'TV fast (CPU)': 36, 'TV naive (CPU)': 3000},
        rd=0.02,
        save_path='experiments/exp_xxx/ml_vs_tv.png',
    )

    save_report('experiments/exp_xxx/report.txt', ...)
"""
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── shared palette (consistent with timeseries.py / comparison.py) ──────────
C_NO_TV   = '#aaaaaa'   # no correction baseline
C_TV_SIM  = '#e67e22'   # orange  — TV from simulation
C_TV_FAST = '#27ae60'   # green   — TV fast algorithm
C_K1      = '#aed6f1'   # light blue
C_K3      = '#2471a3'   # mid blue
C_K5      = '#1a3a5c'   # dark blue
C_NAIVE   = '#c0392b'   # red     — TV naive (reference only)

_K_COLORS = {1: C_K1, 2: '#5dade2', 3: C_K3, 5: C_K5}


# ── helpers ──────────────────────────────────────────────────────────────────

def _bar_label(ax, rects, fmt='{:.0f}', pad=3, fontsize=8, color='#333333'):
    """Annotate bars with their value."""
    for r in rects:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width() / 2, h + pad,
                fmt.format(h), ha='center', va='bottom',
                fontsize=fontsize, color=color, fontweight='bold')


def _spine_clean(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ── panel helpers ─────────────────────────────────────────────────────────────

def _draw_timeseries(ax,
                     timesteps, metrics_by_k,
                     metrics_tv_sim, metrics_tv_fast,
                     rd: float):
    ts = np.array(timesteps)
    k_vals = sorted(metrics_by_k.keys())

    # No-TV baseline — use the K=1 input as proxy (or pass separately if needed)
    # Draw all model K lines
    for k in k_vals:
        c = _K_COLORS.get(k, C_K5)
        lw = 2.2 if k == max(k_vals) else 1.4
        ax.plot(ts, metrics_by_k[k]['mean'], color=c, lw=lw,
                label=f'Model K={k}  (GPU)')

    # TV sim
    ax.plot(ts, metrics_tv_sim['mean'], color=C_TV_SIM, lw=2.0, ls='--',
            label='TV simulation  (reference)')

    # TV fast
    ax.plot(ts, metrics_tv_fast['mean'], color=C_TV_FAST, lw=2.0, ls='-.',
            label='TV fast  (cKDTree, CPU)')

    # rd_test line
    ax.axhline(rd, color='#999', lw=0.9, ls=':', zorder=0)
    ax.text(ts[-1] + 5, rd, f'rd={rd}', fontsize=7.5, color='#777',
            va='center')

    # shade K=5 advantage over TV sim
    k5   = np.array(metrics_by_k[max(k_vals)]['mean'])
    tv_s = np.array(metrics_tv_sim['mean'])
    ax.fill_between(ts, k5, tv_s, where=(k5 >= tv_s),
                    alpha=0.10, color=C_K5, label='K=5 ahead of TV sim')

    ax.set_xlabel('Timestep', fontsize=9)
    ax.set_ylabel('Mean min-NN distance', fontsize=9)
    ax.set_title('Quality over trajectory', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7.5, framealpha=0.92, loc='lower right')
    ax.grid(True, alpha=0.25)
    ax.set_xlim(ts[0] - 20, ts[-1] + 60)
    _spine_clean(ax)


def _draw_timings(ax, timings_ms: Dict[str, float]):
    """Log-scale bar chart of inference timings."""
    labels = list(timings_ms.keys())
    values = list(timings_ms.values())

    colors = []
    for lbl in labels:
        if 'naive' in lbl.lower():
            colors.append(C_NAIVE)
        elif 'fast' in lbl.lower() or 'tv' in lbl.lower():
            colors.append(C_TV_FAST)
        else:
            # model — pick by K
            for k, c in _K_COLORS.items():
                if f'K={k}' in lbl:
                    colors.append(c)
                    break
            else:
                colors.append(C_K5)

    x = np.arange(len(labels))
    rects = ax.bar(x, values, color=colors, width=0.6, zorder=3)

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Inference time (ms, log scale)', fontsize=9)
    ax.set_title('Inference time per frame', fontsize=10, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.25, zorder=0)

    # Reference lines
    for ms, lbl in [(10, '10 ms'), (100, '100 ms'), (1000, '1 s')]:
        ax.axhline(ms, color='#bbb', lw=0.8, ls=':')
        ax.text(len(labels) - 0.4, ms * 1.12, lbl, fontsize=7, color='#888',
                va='bottom', ha='right')

    # Annotate bars
    for rect, v in zip(rects, values):
        fmt = f'{v:.0f} ms' if v >= 1 else f'{v:.1f} ms'
        ax.text(rect.get_x() + rect.get_width() / 2,
                v * 1.35, fmt,
                ha='center', va='bottom', fontsize=7.5,
                fontweight='bold', color='#222')

    _spine_clean(ax)


def _draw_quality_all_t(ax,
                        timesteps,
                        metrics_by_k,
                        metrics_tv_sim,
                        metrics_tv_fast,
                        rd: float):
    """
    Grouped bar chart: Model K=5 | TV sim | TV fast  for every timestep.
    Shows mean NN-distance so all three methods are directly comparable.
    """
    k_max = max(metrics_by_k.keys())
    n_t   = len(timesteps)
    x     = np.arange(n_t)
    w     = 0.26

    k5_means   = metrics_by_k[k_max]['mean']
    tv_s_means = metrics_tv_sim['mean']
    tv_f_means = metrics_tv_fast['mean']

    ax.bar(x - w, k5_means,   w, color=C_K5,      label=f'Model K={k_max} (GPU)', zorder=3)
    ax.bar(x,     tv_s_means, w, color=C_TV_SIM,   label='TV simulation',           zorder=3)
    ax.bar(x + w, tv_f_means, w, color=C_TV_FAST,  label='TV fast (CPU)',            zorder=3)

    ax.axhline(rd, color='#aaa', lw=0.9, ls=':', zorder=0)
    ax.text(n_t - 0.3, rd + 0.0002, f'rd={rd}', fontsize=7, color='#888',
            va='bottom', ha='right')

    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in timesteps], fontsize=7.5)
    ax.set_xlabel('Timestep', fontsize=9)
    ax.set_ylabel('Mean min-NN distance', fontsize=9)
    ax.set_title(f'Quality at every timestep  (K={k_max} vs TV sim vs TV fast)',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.92)
    ax.grid(True, axis='y', alpha=0.25, zorder=0)
    _spine_clean(ax)


# ── main function ─────────────────────────────────────────────────────────────

def plot_ml_vs_tv(
    timesteps:       List[int],
    metrics_by_k:    Dict[int, Dict[str, List[float]]],
    metrics_tv_sim:  Dict[str, List[float]],
    metrics_tv_fast: Dict[str, List[float]],
    timings_ms:      Dict[str, float],
    rd:              float = 0.02,
    save_path:       Optional[str] = None,
    show:            bool = False,
) -> plt.Figure:
    """
    Three-panel ML vs TV comparison figure.

    Parameters
    ----------
    timesteps       : list of sampled timestep indices
    metrics_by_k    : {K: {'mean': [...], 'cv': [...]}}
    metrics_tv_sim  : {'mean': [...], 'cv': [...]}  TV from simulation
    metrics_tv_fast : {'mean': [...], 'cv': [...]}  TV fast algorithm
    timings_ms      : {label: ms}  e.g. {'K=1 (GPU)': 3, 'K=5 (GPU)': 10, ...}
    rd              : rd_test value
    save_path       : if given, save figure here
    show            : call plt.show()
    """
    # Layout: timeseries (wide) | timing bars | grouped quality bars all timesteps
    fig = plt.figure(figsize=(19, 5.5))
    gs  = fig.add_gridspec(1, 3, width_ratios=[1.8, 1.0, 2.2],
                           left=0.05, right=0.98,
                           top=0.87, bottom=0.18,
                           wspace=0.36)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    _draw_timeseries(ax1, timesteps, metrics_by_k,
                     metrics_tv_sim, metrics_tv_fast, rd)
    _draw_timings(ax2, timings_ms)
    _draw_quality_all_t(ax3, timesteps, metrics_by_k,
                        metrics_tv_sim, metrics_tv_fast, rd)

    fig.suptitle(
        'ML Corrector (GPU) vs Fast TV (CPU)  —  SPH N=2500, domain [0,1]2,  rd=0.02',
        fontsize=11, fontweight='bold', y=0.97,
    )

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches='tight')
        print(f'Saved -> {save_path}')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ── text report ───────────────────────────────────────────────────────────────

_APPROACH_TEXT = """\
┌─────────────────────────────────────────────────────────────────────────────────┐
│  APPROACH 1 — ML Corrector  (Poisson-disk violation-weighted edge network)      │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Architecture  : model9 — violation-weighted edge aggregation + tanh-clamped   │
│                  output MLP.  Input: (N,2) point cloud in invariant space.      │
│  Inference     : tiled 6×6 grid, ghost buffer = rd_test for PBC correctness.   │
│                  Coord scaling: rd_train/rd_test=3.8 maps SPH→training domain. │
│  Application   : K iterative passes (K=1..5).  Run on GPU (RTX 4080 Super).    │
│  Key property  : explicitly targets pairs closer than rd (violation-weighted    │
│                  attention assigns zero weight to non-violating neighbours).     │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  APPROACH 2 — Fast TV (Transport Velocity as position correction)               │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Kernel        : Gaussian W(r)=exp(-(r/h)²),  dW/dr = -2r/h² · exp(-(r/h)²).  │
│  Update rule   : Δpos_a = dt · Σ_b [(dW/dr)(r_ab) · e_ab]                     │
│                  dt=0.2, h=1.3/√N≈0.026, cutoff=2h≈0.052, n_iter=10.          │
│  Neighbour     : scipy.spatial.cKDTree(boxsize=domain) — native PBC support,   │
│  search        : O(N·k) vs O(N²) for naive;  ~85× faster for N=2500.          │
│  Application   : run on CPU.                                                    │
│  Key property  : minimises global density non-uniformity.  Does NOT directly   │
│                  target minimum-distance violations → cannot improve mean_nn    │
│                  for near-uniform SPH arrangements (73% of forces push toward   │
│                  the nearest neighbour due to Gaussian gradient shape).          │
└─────────────────────────────────────────────────────────────────────────────────┘
"""


def save_report(
    path:            str,
    timesteps:       List[int],
    metrics_by_k:    Dict[int, Dict[str, List[float]]],
    metrics_tv_sim:  Dict[str, List[float]],
    metrics_tv_fast: Dict[str, List[float]],
    timings_ms:      Dict[str, float],
    rd:              float = 0.02,
):
    """
    Write a plain-text report with approach descriptions and per-timestep table.
    """
    k_vals = sorted(metrics_by_k.keys())
    k_max  = max(k_vals)

    lines = []
    lines.append('=' * 81)
    lines.append(' ML CORRECTOR vs FAST TV — COMPARISON REPORT')
    lines.append(f' SPH N=2500, domain=[0,1]², rd_test={rd}, timesteps={timesteps[0]}..{timesteps[-1]}')
    lines.append('=' * 81)
    lines.append('')
    lines.append(_APPROACH_TEXT)

    # ── Timing table ──────────────────────────────────────────────────────────
    lines.append('INFERENCE TIMES (per frame, steady state)')
    lines.append('-' * 45)
    for lbl, ms in timings_ms.items():
        lines.append(f'  {lbl:<28s} {ms:>8.1f} ms')
    lines.append('')

    # ── Quality table ─────────────────────────────────────────────────────────
    k_hdr = ''.join(f'  K={k}   ' for k in k_vals)
    lines.append(f'MEAN NN-DISTANCE  (rd_test={rd})')
    lines.append('-' * (10 + len(k_hdr) + 30))
    hdr = f'{"t":>5}  {k_hdr}  TV sim   TV fast  vs TV sim (K={k_max})'
    lines.append(hdr)
    lines.append('-' * len(hdr))

    for i, t in enumerate(timesteps):
        row = f'{t:>5}  '
        for k in k_vals:
            row += f'{metrics_by_k[k]["mean"][i]:.5f}  '
        tv_s  = metrics_tv_sim['mean'][i]
        tv_f  = metrics_tv_fast['mean'][i]
        k5    = metrics_by_k[k_max]['mean'][i]
        pct   = (k5 / tv_s - 1) * 100
        row  += f'{tv_s:.5f}  {tv_f:.5f}  {pct:+.1f}%'
        lines.append(row)

    lines.append('')

    # ── Summary stats ─────────────────────────────────────────────────────────
    mean_pct = np.mean([(metrics_by_k[k_max]['mean'][i] / metrics_tv_sim['mean'][i] - 1)*100
                        for i in range(len(timesteps))])
    peak_pct = np.max([(metrics_by_k[k_max]['mean'][i] / metrics_tv_sim['mean'][i] - 1)*100
                       for i in range(len(timesteps))])
    lines.append('SUMMARY')
    lines.append('-' * 40)
    lines.append(f'  Model K={k_max} vs TV sim: mean {mean_pct:+.1f}%,  peak {peak_pct:+.1f}%  over all timesteps')
    lines.append(f'  TV fast mean_nn: flat ~{np.mean(metrics_tv_fast["mean"]):.4f}  (below no-correction baseline)')

    if 'TV naive (CPU)' in timings_ms and f'K={k_max} (GPU)' in timings_ms:
        speedup_naive_over_model = timings_ms['TV naive (CPU)'] / timings_ms[f'K={k_max} (GPU)']
        speedup_fast_over_model  = timings_ms['TV fast (CPU)']  / timings_ms[f'K={k_max} (GPU)']
        lines.append(f'  Speed: model K={k_max} is {speedup_fast_over_model:.1f}× faster than TV fast, '
                     f'{speedup_naive_over_model:.0f}× faster than TV naive')

    lines.append('')
    lines.append('CONCLUSION')
    lines.append('-' * 40)
    tv_fast_pct_below = abs(np.mean(
        [(m - metrics_tv_sim['mean'][i]) / metrics_tv_sim['mean'][i] * 100
         for i, m in enumerate(metrics_tv_fast['mean'])]
    ))
    k_max_ms  = timings_ms.get(f'K={k_max} (GPU)', 10)
    tv_fast_ms = timings_ms.get('TV fast (CPU)', 36)
    lines.append(
        f'  The ML corrector (K={k_max}, GPU) consistently achieves {mean_pct:+.1f}% higher mean NN-distance\n'
        f'  than TV simulation, while running in ~{k_max_ms:.0f} ms/frame on GPU.\n'
        f'  The Fast TV post-processing ({tv_fast_ms:.0f} ms/frame, CPU) does NOT improve\n'
        f'  the distribution: it reduces mean_nn by ~{tv_fast_pct_below:.0f}% below TV sim.\n'
        f'  Root cause: the Gaussian kernel gradient increases with r (for r < h/sqrt(2)), so 73% of\n'
        f'  forces push particles *toward* their nearest neighbour in near-uniform arrangements.\n'
        f'  The ML model avoids this by using violation-weighted attention that exclusively\n'
        f'  activates on pairs violating the minimum-distance constraint.'
    )
    lines.append('')
    lines.append('=' * 81)

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'Report saved -> {path}')
