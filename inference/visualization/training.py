"""
visualization/training.py
-------------------------
Two functions for inspecting training artifacts.

  plot_training_curve  — loss curve: K=1 vs K=unroll violation reduction
  plot_cloud_pair      — side-by-side clean / noisy cloud coloured by
                         violation status (green = valid, red = violating)
"""
import csv
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# ── 1. Training loss curve + sample image ─────────────────────────────────────
def plot_training_curve(
    run_dir:   str | Path,
    figsize:   Tuple[float, float] = (13, 4),
    save_path: Optional[str] = None,
    show:      bool = True,
):
    """
    Two-panel figure from a training_artifacts run directory:
      Left  — violation reduction % over iterations (K=1 deploy vs K=unroll train)
      Right — the most recent training sample image (noisy input | corrected output)

    Parameters
    ----------
    run_dir : path to training_artifacts/train_run_YYYY-MM-DD_HH-MM-SS/
    """
    run_dir = Path(run_dir)

    iters, vr_k1, vr_kK = [], [], []
    with open(run_dir / 'loss.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            iters.append(int(row['iteration']))
            vr_k1.append(float(row['viol_reduction_pct_k1']))
            vr_kK.append(float(row['viol_reduction_pct_kK']))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    ax.plot(iters, vr_k1, lw=1.8, label='K=1 (deploy)')
    ax.plot(iters, vr_kK, lw=1.8, label='K=unroll (train)', ls='--')
    ax.set_xlabel('iteration')
    ax.set_ylabel('violation reduction %')
    ax.set_title('Training curve')
    ax.legend(); ax.grid(alpha=0.3)
    ax.axhline(90, color='gray', ls=':', lw=0.8, alpha=0.6)
    if vr_k1:
        ax.text(0.98, 0.05, f'final K=1: {vr_k1[-1]:.1f}%',
            transform=ax.transAxes, ha='right', fontsize=9, color='#555')

    ax = axes[1]
    samples = sorted((run_dir / 'samples').glob('sample_*.png'))
    if samples:
        img = mpimg.imread(samples[-1])
        ax.imshow(img); ax.axis('off')
        ax.set_title(f'Last training sample: {samples[-1].name}\n'
                     '(noisy input | corrected output)')
    else:
        ax.text(0.5, 0.5, 'No sample images found',
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

    fig.suptitle(f'Training artifact: {run_dir.name}', fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ── 2. Clean vs noisy cloud ───────────────────────────────────────────────────
def plot_cloud_pair(
    clean:     np.ndarray,
    noisy:     np.ndarray,
    rd:        float,
    figsize:   Tuple[float, float] = (10, 4.5),
    save_path: Optional[str] = None,
    show:      bool = True,
):
    """
    Side-by-side scatter of a clean and noisy Poisson disk cloud.
    Points coloured green (valid, min-nn >= rd) or red (violating).

    Parameters
    ----------
    clean, noisy : (N, 2) arrays
    rd           : minimum distance constraint
    """
    from scipy.spatial import cKDTree

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, pts, title in zip(axes, [clean, noisy], ['clean', 'noisy']):
        dists  = cKDTree(pts).query(pts, k=2)[0][:, 1]
        colors = ['#f87171' if d < rd else '#4ade80' for d in dists]
        ax.scatter(pts[:, 0], pts[:, 1], c=colors, s=18)
        n_viol = (np.array(dists) < rd).sum()
        ax.set_title(f'{title}  (N={len(pts)}, rd={rd})\n'
                     f'{n_viol} violating particles')
        ax.set_aspect('equal')
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
