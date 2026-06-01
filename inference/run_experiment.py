"""
run_experiment.py
-----------------
Main entry point for SPH inference experiments.

Creates an experiment directory mirroring the training_artifacts structure:

  inference/experiments/exp_YYYY-MM-DD_HH-MM-SS/
  ├── config.yaml       copy of the experiment config
  ├── run.log           timestep-by-timestep metrics
  ├── timeseries.png    mean nn-distance + CV over all sampled timesteps
  └── frames/
      ├── t0000.png
      ├── t0100.png
      └── ...

Usage
-----
    # Run with default 6x6 config
    .venv\\Scripts\\python.exe inference/run_experiment.py inference/configs/grid_6x6.yaml

    # Run 10x10 experiment
    .venv\\Scripts\\python.exe inference/run_experiment.py inference/configs/grid_10x10.yaml

    # Single timestep (fast check)
    .venv\\Scripts\\python.exe inference/run_experiment.py inference/configs/grid_6x6.yaml --timestep 300

    # Override stride
    .venv\\Scripts\\python.exe inference/run_experiment.py inference/configs/grid_6x6.yaml --stride 50
"""
import argparse
import logging
import shutil
import sys
from datetime import datetime
import matplotlib
matplotlib.use('Agg')   # file-only rendering; must be set before any plt import
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.pipeline.corrector import Corrector, CorrectorConfig
from inference.pipeline.pbc import min_nn_pbc
from inference.visualization.comparison import plot_comparison_frame
from inference.visualization.timeseries import plot_timeseries


def _setup_logger(log_path: Path) -> logging.Logger:
    log = logging.getLogger('experiment')
    log.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s  %(message)s', '%H:%M:%S')
    fh  = logging.FileHandler(log_path)
    sh  = logging.StreamHandler(sys.stdout)
    for h in (fh, sh):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config',       help='Path to experiment YAML config')
    parser.add_argument('--timestep',   type=int, default=None,
                        help='Single timestep (skips full run)')
    parser.add_argument('--stride',     type=int, default=None,
                        help='Override config stride')
    args = parser.parse_args()

    cfg = CorrectorConfig.from_yaml(args.config)
    if args.stride is not None:
        cfg.stride = args.stride

    # ── create experiment directory ──────────────────────────────────────────
    tag     = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_dir = Path('inference/experiments') / f'exp_{tag}'
    exp_dir.mkdir(parents=True)
    frames_dir = exp_dir / 'frames'
    frames_dir.mkdir()

    shutil.copy(args.config, exp_dir / 'config.yaml')
    log = _setup_logger(exp_dir / 'run.log')
    log.info(f'Config: {args.config}')
    log.info(f'Grid: {cfg.grid_size}x{cfg.grid_size}  '
             f'scale={cfg.rd_train/cfg.rd_test:.2f}  '
             f'k_values={cfg.k_values}')

    # ── load data + corrector ────────────────────────────────────────────────
    pos_without = np.load(Path(cfg.checkpoint).parent.parent /
                          '..' / 'inference' / 'sph_data' / 'positions_without.npy'
                          if False else 'inference/sph_data/positions_without.npy')
    pos_with    = np.load('inference/sph_data/positions.npy')
    T, N, _     = pos_without.shape
    log.info(f'Data loaded  T={T}  N={N}')

    corrector = Corrector(cfg)
    log.info('Corrector ready')

    # ── timesteps ────────────────────────────────────────────────────────────
    if args.timestep is not None:
        timesteps = [args.timestep % T]
    else:
        timesteps = list(range(0, T, cfg.stride))

    ts_list          = []
    metrics_by_k     = {k: {'mean': [], 'cv': []} for k in cfg.k_values}
    metrics_tv       = {'mean': [], 'cv': []}

    for i, t in enumerate(timesteps):
        log.info(f'[{i+1}/{len(timesteps)}]  t={t}')
        pts_wo = pos_without[t].astype(np.float32)
        pts_wi = pos_with[t].astype(np.float32)

        corrected_by_k = {}
        for k in cfg.k_values:
            corr = corrector.apply(pts_wo, k=k)
            corrected_by_k[k] = corr
            nn = min_nn_pbc(corr)
            metrics_by_k[k]['mean'].append(float(nn.mean()))
            metrics_by_k[k]['cv'].append(float(nn.std() / nn.mean()))
            log.info(f'  K={k}: mean_nn={nn.mean():.5f}')

        nn_tv = min_nn_pbc(pts_wi)
        metrics_tv['mean'].append(float(nn_tv.mean()))
        metrics_tv['cv'].append(float(nn_tv.std() / nn_tv.mean()))
        log.info(f'  TV:  mean_nn={nn_tv.mean():.5f}')
        ts_list.append(t)

        save = frames_dir / f't{t:04d}.png'
        plot_comparison_frame(
            pts_wo, corrected_by_k, pts_wi,
            timestep=t,
            grid=cfg.grid_size,
            scale=cfg.rd_train / cfg.rd_test,
            save_path=str(save),
        )

    # ── timeseries plot ──────────────────────────────────────────────────────
    if len(ts_list) > 1:
        plot_timeseries(ts_list, metrics_by_k, metrics_tv,
                        save_path=str(exp_dir / 'timeseries.png'), show=False)
        print(f'Timeseries saved -> {exp_dir / "timeseries.png"}')

    log.info(f'Done. Results in {exp_dir}')


if __name__ == '__main__':
    main()
