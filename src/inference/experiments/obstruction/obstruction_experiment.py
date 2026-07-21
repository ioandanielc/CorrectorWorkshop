"""
obstruction_experiment.py
-------------------------
SPH particle initialization around an obstacle, corrected by the whole-cloud
model12 corrector.

Scenario creation (obstruction.py, unchanged):
  - Real particles: regular grid excluding the obstacle interior, plus small
    noise to break the grid symmetry
  - Ghost particles: eroded fill (erode_by=rd) so real particles can sit on
    the contour

Correction: k passes; each pass concatenates the (fixed) ghosts to the current
real particles, runs one whole-cloud apply, and keeps only the real rows — the
ghosts represent the solid and are re-pinned every pass.

Output: initial vs corrected figure (detailed + clean rows) + report, to
artifacts/inference/experiments/obstruction/runs/.

Run:
    .venv\\Scripts\\python.exe src/inference/experiments/obstruction/obstruction_experiment.py
"""
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[3]))

from inference.correctors.base import Experiment
from inference.correctors.wholecloud.wholecloud_corrector import (WholeCloudCorrector2D,
                                                                  WholeCloudCorrector2DConfig)
from inference.experiments.obstruction.obstruction import fill_obstruction, gear_mask


@dataclass
class ObstructionExperimentConfig:
    corrector_config: str = 'src/configs/experiments/obstruction/wholecloud.yaml'
    rd:          float = 0.012
    domain:      float = 1.0
    cx:          float = 0.5
    cy:          float = 0.5
    noise_scale: float = 0.3   # noise_std = noise_scale * rd
    k:           int   = 5     # correction passes (ghosts re-pinned each pass)
    seed:        int   = 42
    device:      str   = 'cpu'


def grid_outside(rd, domain, exclude):
    xs = np.arange(0, domain + 0.5 * rd, rd)
    ys = np.arange(0, domain + 0.5 * rd, rd)
    gx, gy = np.meshgrid(xs, ys)
    pts    = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    return pts[~exclude(pts)]


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
    Grid + ghost-fill initialization against a gear obstacle, corrected by
    the whole-cloud corrector; saves an initial-vs-corrected figure + report.
    """

    def __init__(self, cfg: ObstructionExperimentConfig):
        self.cfg = cfg

        corrector_cfg          = WholeCloudCorrector2DConfig.from_yaml(cfg.corrector_config)
        corrector_cfg.rd_test  = cfg.rd       # scale = rd_train / rd_test
        corrector_cfg.box      = cfg.domain   # bounded scene; PBC wrap couples the
                                              # borders — same behaviour the tiled
                                              # corrector had, acceptable at this rd
        corrector_cfg.device   = cfg.device
        self.corrector = WholeCloudCorrector2D(corrector_cfg)

        self.mask   = gear_mask(cfg.cx, cfg.cy, r=0.18, n_teeth=8, k=2)
        self.ghosts = fill_obstruction(self.mask, cfg.rd, erode_by=cfg.rd)
        print(f'Ghost particles: {len(self.ghosts)}')

    def run(self) -> None:
        cfg = self.cfg

        rng        = np.random.default_rng(cfg.seed)
        real_grid  = grid_outside(cfg.rd, cfg.domain, self.mask)
        n_real     = len(real_grid)
        noise      = rng.normal(0, cfg.noise_scale * cfg.rd, real_grid.shape).astype(np.float32)
        initial    = np.clip(real_grid + noise, 0.0, cfg.domain)
        print(f'Real particles: {n_real}')

        current = initial.copy()
        for i in range(cfg.k):
            all_pts = np.concatenate([current, self.ghosts])   # ghosts re-pinned
            current = self.corrector.apply(all_pts, k=1)[:n_real]
            print(f'  pass {i+1}/{cfg.k}: mean_nn={min_nn(current).mean():.4f}')

        self._save(initial, current, n_real)

    def _save(self, initial, corrected, n_real) -> None:
        cfg = self.cfg
        tag = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        out_dir = Path('artifacts/inference/experiments/obstruction/runs') / f'wholecloud_{tag}'
        out_dir.mkdir(parents=True, exist_ok=True)

        states = [('Initial', initial), (f'Corrected (k={cfg.k})', corrected)]
        all_nn = np.concatenate([min_nn(p) for _, p in states])
        vmin, vmax = np.percentile(all_nn, 2), np.percentile(all_nn, 98)

        fig, axes = plt.subplots(2, 2, figsize=(11, 11))
        fig.suptitle(f'Obstacle initialization, wholecloud corrector  '
                     f'(gear: r=0.18, 8 teeth, k=2  |  rd={cfg.rd})',
                     fontsize=12, fontweight='bold')

        sc = None
        for col, (title, pts) in enumerate(states):
            nn = min_nn(pts)

            # row 0: detailed (ghosts + nn colour + boundary)
            ax = axes[0, col]
            ax.scatter(self.ghosts[:, 0], self.ghosts[:, 1],
                       c='#aaaaaa', s=4, lw=0, alpha=0.45, zorder=2)
            sc = ax.scatter(pts[:, 0], pts[:, 1], c=nn, cmap='viridis', s=5, lw=0,
                            vmin=vmin, vmax=vmax, alpha=0.9, zorder=3)
            draw_boundary(ax, self.mask, cfg.domain)
            ax.text(0.02, 0.02, f'mean nn={nn.mean():.4f}',
                    transform=ax.transAxes, fontsize=8, color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='#333', alpha=0.6))

            # row 1: clean (black dots only)
            axes[1, col].scatter(pts[:, 0], pts[:, 1], c='black', s=5, lw=0, alpha=0.85)

            for row in (0, 1):
                ax = axes[row, col]
                ax.set_title(title if row == 0 else '', fontsize=10)
                ax.set_aspect('equal')
                ax.set_xlim(0.0, cfg.domain); ax.set_ylim(0.0, cfg.domain)
                ax.set_xticks([]); ax.set_yticks([])

        plt.tight_layout(rect=[0, 0, 0.92, 1])
        cbar_ax = fig.add_axes([0.93, 0.55, 0.015, 0.4])
        cb = fig.colorbar(sc, cax=cbar_ax)
        cb.set_label('min-NN distance', fontsize=9)
        cb.ax.axhline(cfg.rd, color='red', lw=1.0, linestyle='--')

        fig.savefig(out_dir / 'obstruction.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        with open(out_dir / 'report.txt', 'w') as f:
            f.write(f'corrector : wholecloud k={cfg.k}  (ghosts re-pinned each pass)\n')
            f.write(f'config    : {cfg.corrector_config}  rd={cfg.rd}  domain={cfg.domain}\n')
            f.write(f'particles : {n_real} real + {len(self.ghosts)} ghost\n')
            f.write(f'mean_nn   : {min_nn(initial).mean():.5f} -> {min_nn(corrected).mean():.5f}  (rd={cfg.rd})\n')
            f.write(f'ill%      : {(min_nn(initial) < cfg.rd).mean()*100:.2f} -> '
                    f'{(min_nn(corrected) < cfg.rd).mean()*100:.2f}\n')
        print(f'Saved -> {out_dir}')


if __name__ == '__main__':
    ObstructionExperiment(ObstructionExperimentConfig()).run()
