"""
visualization/timeseries.py
---------------------------
Trajectory-level summary plots comparing nonTV+model vs TV.

  plot_timeseries — mean nn-distance + CV over all sampled timesteps,
                    one line per K value plus TV baseline.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


C_TV      = '#e67e22'   # TV from simulation
C_TV_ALGO = '#27ae60'   # TV algorithm (post-processing)
BLUES     = ['#aed6f1', '#5dade2', '#2471a3', '#1a3a5c']


def plot_timeseries(
    timesteps:        List[int],
    metrics_by_k:     Dict[int, Dict[str, List[float]]],
    metrics_tv:       Dict[str, List[float]],
    metrics_tv_algo:  Optional[Dict[str, List[float]]] = None,
    tv_algo_label:    str = 'TV algo',
    figsize:          Tuple[float, float] = (13, 4.5),
    title:            str = 'SPH corrector vs TV — trajectory summary',
    save_path:        Optional[str] = None,
    show:             bool = True,
):
    """
    Two-panel timeseries: mean nn-distance (left) and CV = std/mean (right).

    Parameters
    ----------
    timesteps       : list of int timestep indices
    metrics_by_k    : {K: {'mean': [...], 'cv': [...]}}  one entry per K value
    metrics_tv      : {'mean': [...], 'cv': [...]}        TV from simulation
    metrics_tv_algo : {'mean': [...], 'cv': [...]}        TV algorithm (optional)
    tv_algo_label   : legend label for TV algo line
    """
    k_values = sorted(metrics_by_k.keys())
    colors   = {k: BLUES[min(i, len(BLUES)-1)] for i, k in enumerate(k_values)}
    ts       = np.array(timesteps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.13, wspace=0.30)

    for k in k_values:
        ax1.plot(ts, metrics_by_k[k]['mean'], color=colors[k], lw=1.8, label=f'K={k}')
        ax2.plot(ts, metrics_by_k[k]['cv'],   color=colors[k], lw=1.8, label=f'K={k}')

    ax1.plot(ts, metrics_tv['mean'], color=C_TV, lw=1.8, ls='--', label='TV sim')
    ax2.plot(ts, metrics_tv['cv'],   color=C_TV, lw=1.8, ls='--', label='TV sim')

    if metrics_tv_algo is not None:
        ax1.plot(ts, metrics_tv_algo['mean'], color=C_TV_ALGO, lw=1.8, ls='-.',
                 label=tv_algo_label)
        ax2.plot(ts, metrics_tv_algo['cv'],   color=C_TV_ALGO, lw=1.8, ls='-.',
                 label=tv_algo_label)

    # shade region where the best K is ahead of TV
    k_best = max(k_values)
    k_arr  = np.array(metrics_by_k[k_best]['mean'])
    tv_arr = np.array(metrics_tv['mean'])
    ax1.fill_between(ts, k_arr, tv_arr, where=(k_arr >= tv_arr),
                     alpha=0.12, color=colors[k_best], label=f'K={k_best} ahead')

    k_cv  = np.array(metrics_by_k[k_best]['cv'])
    tv_cv = np.array(metrics_tv['cv'])
    ax2.fill_between(ts, tv_cv, k_cv, where=(k_cv <= tv_cv),
                     alpha=0.12, color=colors[k_best], label=f'K={k_best} more uniform')

    for ax, ylabel, title_ax in [
        (ax1, 'mean min-nn distance',  'Mean NN-distance over time'),
        (ax2, 'CV = std/mean (min-nn)', 'Distribution uniformity  (lower = better)'),
    ]:
        ax.set_xlabel('timestep', fontsize=9)
        ax.set_ylabel(ylabel,     fontsize=9)
        ax.set_title(title_ax,    fontsize=10)
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(ts[0], ts[-1])

    fig.suptitle(title, fontsize=10)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
