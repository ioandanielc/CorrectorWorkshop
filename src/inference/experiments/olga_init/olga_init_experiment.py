"""
src/inference/experiments/olga_init/olga_init_experiment.py
---------------------------------------------
Corrector for Olga's MD initialization (3D, LJ units).

A generative model proposes an initial particle state for the LS1 MD solver
(N=5096, cubic box L=18, T=1.4, rho=0.874) as an (N, 6) npy: x y z vx vy vz.
The proposed positions contain a small number of far-too-close pairs
(min distance ~0.08) that spike the potential energy and cost ~100 extra MD
timesteps of equilibration. This experiment applies a 3D corrector to the
positions (velocities pass through untouched), writes the corrected (N, 6)
array for LS1, and reports RDF + minimum-distance statistics before/after.

The corrector variant (grid / kdtree / grid_then_kdtree) and the checkpoint
are selected by the config file — one file per variant in
src/configs/experiments/olga_init/.

Usage
-----
    .venv\\Scripts\\python.exe src/inference/experiments/olga_init/olga_init_experiment.py src/configs/experiments/olga_init/n50_grid.yaml
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[3]))

from inference.correctors.base import Experiment
from inference.correctors.grid.corrector import GridCorrector3D, GridCorrector3DConfig
from inference.correctors.kdtree.kdtree_corrector import KDTreeCorrector3D, KDTreeCorrector3DConfig


# ── PBC pair statistics (3D, cKDTree — fine at N~5000) ───────────────────────

def pair_stats(pos: np.ndarray, box: float, rd: float):
    """min pair distance, per-particle nearest-neighbour distances, #pairs < rd."""
    tree   = cKDTree(np.mod(pos, box), boxsize=box)
    nn, _  = tree.query(np.mod(pos, box), k=2)
    nn     = nn[:, 1]                          # column 0 is the point itself
    n_viol = len(tree.query_pairs(rd))
    return nn.min(), nn, n_viol


def rdf_3d(pos: np.ndarray, box: float, r_max: float, n_bins: int = 250):
    """Radial distribution function g(r) in a periodic cubic box."""
    pos   = np.mod(pos, box)
    N     = len(pos)
    rho   = N / box**3
    tree  = cKDTree(pos, boxsize=box)
    pairs = tree.query_pairs(r_max, output_type='ndarray')
    delta = pos[pairs[:, 0]] - pos[pairs[:, 1]]
    delta -= box * np.round(delta / box)       # minimum-image convention
    d     = np.linalg.norm(delta, axis=1)

    edges     = np.linspace(0.0, r_max, n_bins + 1)
    counts, _ = np.histogram(d, bins=edges)
    v_shell   = 4.0 / 3.0 * np.pi * (edges[1:]**3 - edges[:-1]**3)
    expected  = 0.5 * N * rho * v_shell        # pairs per shell if uncorrelated
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    return r_centers, counts / expected


# ── experiment ────────────────────────────────────────────────────────────────

class OlgaInitExperiment(Experiment):
    """
    Load the (N, 6) state, correct the positions with the configured
    corrector(s), save the corrected (N, 6) state + RDF comparison + report.
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path) as f:
            self.raw = yaml.safe_load(f)

        d = self.raw['data']
        self.input_path = d['input']
        self.box        = float(d['box'])
        self.rd_test    = float(d['rd_test'])

        exp = self.raw.get('experiment', {})
        self.kind     = exp.get('corrector', 'grid')   # grid | kdtree | grid_then_kdtree
        self.k_grid   = int(exp.get('k_grid', 5))
        self.k_kdtree = int(exp.get('k_kdtree', 10))
        if self.kind not in ('grid', 'kdtree', 'grid_then_kdtree'):
            raise ValueError(f"experiment.corrector must be grid | kdtree | grid_then_kdtree, got '{self.kind}'")

    def _build_steps(self):
        """[(corrector, k), ...] in application order."""
        steps = []
        if self.kind in ('grid', 'grid_then_kdtree'):
            steps.append(('grid',
                          GridCorrector3D(GridCorrector3DConfig.from_yaml(self.config_path)),
                          self.k_grid))
        if self.kind in ('kdtree', 'grid_then_kdtree'):
            steps.append(('kdtree',
                          KDTreeCorrector3D(KDTreeCorrector3DConfig.from_yaml(self.config_path)),
                          self.k_kdtree))
        return steps

    def run(self) -> None:
        arr = np.load(self.input_path)
        if arr.ndim != 2 or arr.shape[1] != 6:
            raise ValueError(f'{self.input_path}: expected (N, 6) x y z vx vy vz, got {arr.shape}')
        pos, vel = arr[:, :3].astype(np.float64), arr[:, 3:]
        N, L, rd = len(pos), self.box, self.rd_test

        tag     = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        out_dir = Path('artifacts/inference/experiments/olga_init/runs') / f'{Path(self.config_path).stem}_{tag}'
        out_dir.mkdir(parents=True)

        min_b, nn_b, viol_b = pair_stats(pos, L, rd)
        print(f'Input : N={N}  box={L}  rd_test={rd}')
        print(f'Before: min_dist={min_b:.4f}  mean_nn={nn_b.mean():.4f}  pairs<rd={viol_b}')

        corrected = pos.copy()
        t0 = time.perf_counter()
        for name, corrector, k in self._build_steps():
            t1 = time.perf_counter()
            corrected = corrector.apply(corrected, k=k)
            print(f'{name:6s} k={k}: {time.perf_counter() - t1:.1f}s')
        runtime   = time.perf_counter() - t0
        corrected = np.mod(corrected, L)               # wrap back into the box

        min_a, nn_a, viol_a = pair_stats(corrected, L, rd)
        disp = np.linalg.norm(corrected - np.mod(pos, L), axis=1)
        disp = np.minimum(disp, np.abs(disp - L))      # ignore wrap jumps in the norm
        print(f'After : min_dist={min_a:.4f}  mean_nn={nn_a.mean():.4f}  pairs<rd={viol_a}')

        # corrected state for LS1, same layout and dtype as the input
        out_npy = out_dir / 'data_corrected.npy'
        np.save(out_npy, np.concatenate([corrected, vel], axis=1).astype(arr.dtype))

        # RDF before/after
        r_max = min(L / 2, 3.5)
        r, g_before = rdf_3d(pos, L, r_max)
        _, g_after  = rdf_3d(corrected, L, r_max)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(r, g_before, lw=1.2, color='#888888', label='before')
        ax.plot(r, g_after,  lw=1.2, color='#1f77b4', label=f'after ({self.kind})')
        ax.axvline(rd, color='#e74c3c', lw=0.8, ls='--', label=f'rd_test={rd}')
        ax.set_xlabel('r')
        ax.set_ylabel('g(r)')
        ax.set_title(f'{Path(self.config_path).stem}: N={N}, box={L}')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / 'rdf.png', dpi=150)
        plt.close(fig)

        with open(out_dir / 'report.txt', 'w') as f:
            f.write(f'config     : {self.config_path}\n')
            f.write(f'input      : {self.input_path}  (N={N}, box={L})\n')
            f.write(f'corrector  : {self.kind}  k_grid={self.k_grid}  k_kdtree={self.k_kdtree}\n')
            f.write(f'rd_test    : {rd}\n')
            f.write(f'runtime    : {runtime:.1f}s\n')
            f.write(f'min_dist   : {min_b:.4f} -> {min_a:.4f}\n')
            f.write(f'mean_nn    : {nn_b.mean():.4f} -> {nn_a.mean():.4f}\n')
            f.write(f'pairs<rd   : {viol_b} -> {viol_a}\n')
            f.write(f'displacement: mean={disp.mean():.4f}  max={disp.max():.4f}\n')

        print(f'Saved -> {out_npy}')
        print(f'Saved -> {out_dir / "rdf.png"}')
        print(f'Saved -> {out_dir / "report.txt"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='src/configs/experiments/olga_init/*.yaml')
    args = parser.parse_args()
    OlgaInitExperiment(args.config).run()
