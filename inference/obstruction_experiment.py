"""
obstruction_experiment.py
-------------------------
Two initialization strategies for SPH particles around an obstacle.

Method 1 — grid + ghost particles
  - Real particles: regular grid excluding the obstacle interior
  - Ghost particles: eroded fill (erode_by=rd) so real particles can sit on contour
  - Add small noise to real particles to break the grid symmetry
  - Apply corrector cumulatively (K=1 → 3 → 5), ghosts held fixed each step

Method 2 — uniform + wireframe MC
  - Real particles: uniform random draw outside the obstacle
  - No ghost particles; the obstacle mask is queried directly
  - Apply corrector, then MC-resample any particle that entered the obstacle
  - Repeat cumulatively for K=1 → 3 → 5 steps

Visualization: 2 rows (methods) × 4 columns (initial, K=1, K=3, K=5)
  - Particles colored by min-NN distance (global color scale)
  - Ghost particles shown as faint gray in Method 1 panels
  - True obstacle boundary shown as red contour in all panels

Run:
    .venv\\Scripts\\python.exe inference/obstruction_experiment.py
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.pipeline.base import Experiment
from inference.pipeline.corrector import GridCorrector, GridCorrectorConfig
from inference.pipeline.obstruction import fill_obstruction, gear_mask


@dataclass
class ObstructionExperimentConfig:
    corrector_config: str = 'inference/configs/grid_6x6.yaml'
    rd:          float     = 0.012
    domain:      float     = 1.0
    cx:          float     = 0.5
    cy:          float     = 0.5
    noise_scale: float     = 0.3   # noise_std = noise_scale * rd
    k_values:    List[int] = field(default_factory=lambda: [1, 3, 5])
    seed:        int       = 42
    device:      str       = 'cuda'   # 'cpu' if no GPU


def grid_outside(rd, domain, exclude):
    xs = np.arange(0, domain + 0.5 * rd, rd)
    ys = np.arange(0, domain + 0.5 * rd, rd)
    gx, gy = np.meshgrid(xs, ys)
    pts    = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    return pts[~exclude(pts)]


def uniform_outside(N, domain, exclude, rng):
    buf = []
    n   = 0
    while n < N:
        batch = rng.uniform(0, domain, (max(N * 4, 512), 2)).astype(np.float32)
        valid = batch[~exclude(batch)]
        buf.append(valid)
        n += len(valid)
    return np.concatenate(buf)[:N]


def min_nn(pts):
    return cKDTree(pts).query(pts, k=2)[0][:, 1].astype(np.float32)


def draw_boundary(ax, mask, domain, resolution=400):
    t        = np.linspace(0, domain, resolution)
    gx, gy   = np.meshgrid(t, t)
    pts2     = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    m        = mask(pts2).reshape(resolution, resolution).astype(float)
    ax.contour(t, t, m, levels=[0.5], colors='#e74c3c', linewidths=1.0, zorder=5)


class ObstructionExperiment(Experiment):
    """
    Runs both obstacle-initialization strategies against a gear obstacle,
    correcting cumulatively for each k in cfg.k_values, and saves a
    comparison figure next to this script.
    """

    def __init__(self, cfg: ObstructionExperimentConfig):
        self.cfg = cfg

        corrector_cfg          = GridCorrectorConfig.from_yaml(cfg.corrector_config)
        corrector_cfg.rd_test  = cfg.rd     # override to match experiment rd (scale = rd_train/rd_test)
        corrector_cfg.device   = cfg.device
        self.corrector = GridCorrector(corrector_cfg)

        self.mask   = gear_mask(cfg.cx, cfg.cy, r=0.18, n_teeth=8, k=2)
        self.ghosts = fill_obstruction(self.mask, cfg.rd, erode_by=cfg.rd)
        print(f'Ghost particles: {len(self.ghosts)}')

    def run(self) -> None:
        cfg = self.cfg

        # ── Method 1: grid + ghosts + noise ──────────────────────────────────
        print('\n--- Method 1 ---')
        rng1       = np.random.default_rng(cfg.seed)
        real_grid  = grid_outside(cfg.rd, cfg.domain, self.mask)
        n_real     = len(real_grid)
        noise      = rng1.normal(0, cfg.noise_scale * cfg.rd, real_grid.shape).astype(np.float32)
        noisy_real = np.clip(real_grid + noise, 0.0, cfg.domain)

        m1 = {'initial': noisy_real.copy()}
        current_m1 = noisy_real.copy()
        k_prev = 0
        for k in cfg.k_values:
            for _ in range(k - k_prev):
                all_pts   = np.concatenate([current_m1, self.ghosts])   # ghosts fixed each step
                corrected = self.corrector.apply(all_pts, k=1)
                current_m1 = corrected[:n_real]
            m1[k] = current_m1.copy()
            print(f'  K={k}: {n_real} real pts,  mean_nn={min_nn(current_m1).mean():.4f}')
            k_prev = k

        # ── Method 2: uniform + wireframe MC ─────────────────────────────────
        print('\n--- Method 2 ---')
        rng2       = np.random.default_rng(cfg.seed + 1)
        init_unif  = uniform_outside(n_real, cfg.domain, self.mask, rng2)

        m2 = {'initial': init_unif.copy()}
        current_m2 = init_unif.copy()
        k_prev = 0
        for k in cfg.k_values:
            for _ in range(k - k_prev):
                corrected = self.corrector.apply(current_m2, k=1)
                inside    = self.mask(corrected)
                if inside.any():
                    replacement       = uniform_outside(int(inside.sum()), cfg.domain, self.mask, rng2)
                    corrected[inside] = replacement
                current_m2 = corrected
            m2[k] = current_m2.copy()
            print(f'  K={k}: {n_real} real pts,  mean_nn={min_nn(current_m2).mean():.4f}')
            k_prev = k

        self._plot(m1, m2, n_real)

    def _plot(self, m1, m2, n_real) -> None:
        cfg = self.cfg
        print('\nRendering...')
        cols       = ['initial'] + cfg.k_values
        col_titles = ['Initial'] + [f'K = {k}' for k in cfg.k_values]

        # global color range across all real-particle sets
        all_nn = np.concatenate([min_nn(v) for results in [m1, m2] for v in results.values()])
        vmin   = np.percentile(all_nn, 2)
        vmax   = np.percentile(all_nn, 98)

        fig, axes = plt.subplots(4, len(cols), figsize=(5.5 * len(cols), 22))
        fig.suptitle(
            f'Obstacle initialization  (gear: r=0.18, 8 teeth, k=2  |  rd={cfg.rd})',
            fontsize=12, fontweight='bold', y=1.005,
        )

        # ── rows 0-1: detailed view (ghosts + color + boundary) ──────────────
        row_info = [
            (m1, 'Method 1\ngrid + ghost particles + noise\n(gray = ghost, colored = real)'),
            (m2, 'Method 2\nuniform random + wireframe MC replacement'),
        ]

        sc = None
        for row_idx, (results, row_label) in enumerate(row_info):
            for col_idx, key in enumerate(cols):
                ax  = axes[row_idx, col_idx]
                pts = results[key]
                nn  = min_nn(pts)

                if row_idx == 0:
                    ax.scatter(self.ghosts[:, 0], self.ghosts[:, 1],
                               c='#aaaaaa', s=4, lw=0, alpha=0.45, zorder=2)

                sc = ax.scatter(pts[:, 0], pts[:, 1],
                                c=nn, cmap='viridis', s=5, lw=0,
                                vmin=vmin, vmax=vmax, alpha=0.9, zorder=3)

                draw_boundary(ax, self.mask, cfg.domain)

                ax.set_title(col_titles[col_idx], fontsize=10)
                ax.set_aspect('equal')
                ax.set_xlim(0.0, cfg.domain); ax.set_ylim(0.0, cfg.domain)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_linewidth(0.5)

                if col_idx == 0:
                    ax.set_ylabel(row_label, fontsize=9, labelpad=8)

                ax.text(0.02, 0.02, f'mean nn={nn.mean():.4f}',
                        transform=ax.transAxes, fontsize=7.5, color='white',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', fc='#333', alpha=0.6))

        # ── separator line between sections ───────────────────────────────────
        fig.add_artist(plt.Line2D(
            [0.02, 0.98], [0.505, 0.505],
            transform=fig.transFigure, color='#bbbbbb', lw=1.2, linestyle='--',
        ))

        # ── rows 2-3: clean view (black dots, no obstacle, no color) ─────────
        clean_row_info = [
            (m1, 'Method 1  (clean)'),
            (m2, 'Method 2  (clean)'),
        ]

        for row_idx, (results, row_label) in enumerate(clean_row_info):
            ax_row = row_idx + 2
            for col_idx, key in enumerate(cols):
                ax  = axes[ax_row, col_idx]
                pts = results[key]

                ax.scatter(pts[:, 0], pts[:, 1],
                           c='black', s=5, lw=0, alpha=0.85, zorder=3)

                ax.set_title(col_titles[col_idx] if ax_row == 2 else '', fontsize=10)
                ax.set_aspect('equal')
                ax.set_xlim(0.0, cfg.domain); ax.set_ylim(0.0, cfg.domain)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_facecolor('white')
                for sp in ax.spines.values():
                    sp.set_linewidth(0.5)

                if col_idx == 0:
                    ax.set_ylabel(row_label, fontsize=9, labelpad=8)

        # ── colorbar for top section ───────────────────────────────────────────
        plt.tight_layout(rect=[0, 0, 0.93, 1])
        cbar_ax = fig.add_axes([0.94, 0.515, 0.015, 0.465])
        cb = fig.colorbar(sc, cax=cbar_ax)
        cb.set_label('min-NN distance', fontsize=9)
        cb.ax.axhline(cfg.rd, color='red', lw=1.0, linestyle='--')
        cb.ax.text(1.6, cfg.rd, f'rd={cfg.rd}', fontsize=7.5, color='red',
                   va='center', transform=cb.ax.get_yaxis_transform())

        out = Path(__file__).parent / 'obstruction_experiment.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved -> {out}')


if __name__ == '__main__':
    ObstructionExperiment(ObstructionExperimentConfig()).run()
