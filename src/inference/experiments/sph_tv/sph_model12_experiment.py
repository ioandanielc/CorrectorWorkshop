"""
sph_model12_experiment.py
--------------------------
Validates the model12 whole-cloud corrector against the real SPH trajectory:
before/after quality (mean nn-distance + illegal%) per sampled timestep.
model12 trains only on synthetic square-lattice Poisson-disk clouds, so this
is the real-data check. For the full-trajectory KG comparison against the
raw / TV / model9-era series, see kg_sweep.py.

Usage
-----
    .venv\\Scripts\\python.exe src/inference/experiments/sph_tv/sph_model12_experiment.py src/configs/experiments/sph_tv/model12_wholecloud.yaml

    # Single timestep (fast check)
    .venv\\Scripts\\python.exe src/inference/experiments/sph_tv/sph_model12_experiment.py src/configs/experiments/sph_tv/model12_wholecloud.yaml --timestep 300
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

sys.path.insert(0, str(Path(__file__).parents[3]))

from inference.correctors.base import Experiment
from inference.correctors.wholecloud.wholecloud_corrector import (WholeCloudCorrector2D,
                                                                  WholeCloudCorrector2DConfig)
from utils.metrics import nn_dists


class SPHModel12Experiment(Experiment):
    """
    Applies the whole-cloud corrector to sampled timesteps of the SPH
    trajectory and reports mean nn-distance + illegal-pair% before/after.
    """

    def __init__(self, config_path: str, timestep: int = None, stride: int = None):
        self.config_path = config_path
        with open(config_path) as f:
            self.raw = yaml.safe_load(f)

        d = self.raw['data']
        self.data_path = d.get('without_tv', 'artifacts/inference/experiments/sph_tv/data/positions_without.npy')
        self.rd_test   = float(d['rd_test'])
        self.box       = float(d.get('box', 1.0))

        exp = self.raw.get('experiment', {})
        self.k        = int(exp.get('k_wholecloud', 5))
        self.stride   = stride if stride is not None else int(exp.get('stride', 200))
        self.timestep = timestep

    def run(self) -> None:
        pos_without = np.load(self.data_path)
        T, N, _ = pos_without.shape
        rd = self.rd_test

        tag     = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        out_dir = Path('artifacts/inference/experiments/sph_tv/runs') / f'model12_wholecloud_{tag}'
        out_dir.mkdir(parents=True)

        corrector = WholeCloudCorrector2D(WholeCloudCorrector2DConfig.from_yaml(self.config_path))
        print(f'Corrector: wholecloud  k={self.k}  box={self.box}')
        print(f'Data: {self.data_path}  T={T}  N={N}  rd_test={rd}')

        timesteps = [self.timestep % T] if self.timestep is not None else list(range(0, T, self.stride))

        ts_list, nn_before, nn_after, ill_before, ill_after, runtimes = [], [], [], [], [], []
        for i, t in enumerate(timesteps):
            pts  = pos_without[t].astype(np.float32)
            nn_b = nn_dists(pts, box=self.box)

            t0        = time.perf_counter()
            corrected = corrector.apply(pts, k=self.k)
            runtime   = time.perf_counter() - t0

            nn_a = nn_dists(corrected, box=self.box)

            ts_list.append(t)
            nn_before.append(float(nn_b.mean()))
            nn_after.append(float(nn_a.mean()))
            ill_before.append(float((nn_b < rd).mean() * 100))
            ill_after.append(float((nn_a < rd).mean() * 100))
            runtimes.append(runtime)
            print(f'[{i+1}/{len(timesteps)}] t={t:4d}  '
                  f'mean_nn {nn_b.mean():.5f} -> {nn_a.mean():.5f}  '
                  f'illegal% {ill_before[-1]:.2f} -> {ill_after[-1]:.2f}  ({runtime:.2f}s)')

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(ts_list, nn_before, 'o-', color='#888888', lw=1.2, ms=3, label='before')
        ax.plot(ts_list, nn_after,  'o-', color='#1f77b4', lw=1.2, ms=3, label=f'after (wholecloud k={self.k})')
        ax.axhline(rd, color='#e74c3c', lw=0.8, ls='--', label=f'rd_test={rd}')
        ax.set_xlabel('timestep')
        ax.set_ylabel('mean nn distance')
        ax.set_title(f'model12 wholecloud: N={N}')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / 'timeseries.png', dpi=150)
        plt.close(fig)

        with open(out_dir / 'report.txt', 'w') as f:
            f.write(f'config    : {self.config_path}\n')
            f.write(f'data      : {self.data_path}  (T={T}, N={N})\n')
            f.write(f'corrector : wholecloud  k={self.k}  box={self.box}\n')
            f.write(f'rd_test   : {rd}\n')
            f.write(f'timesteps : {ts_list}\n')
            f.write(f'runtime   : mean={np.mean(runtimes):.2f}s  total={sum(runtimes):.1f}s\n\n')
            f.write(f'{"t":>6s}  {"nn_before":>10s}  {"nn_after":>10s}  {"ill%_before":>12s}  {"ill%_after":>12s}\n')
            for t, nb, na, ib, ia in zip(ts_list, nn_before, nn_after, ill_before, ill_after):
                f.write(f'{t:6d}  {nb:10.5f}  {na:10.5f}  {ib:12.2f}  {ia:12.2f}\n')

        print(f'Saved -> {out_dir / "report.txt"}')
        print(f'Saved -> {out_dir / "timeseries.png"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='src/configs/experiments/sph_tv/*.yaml')
    parser.add_argument('--timestep', type=int, default=None, help='Single timestep (skips full sweep)')
    parser.add_argument('--stride',   type=int, default=None, help='Override config stride')
    args = parser.parse_args()
    SPHModel12Experiment(args.config, timestep=args.timestep, stride=args.stride).run()
