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
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.pipeline.corrector import Corrector, CorrectorConfig
from inference.pipeline.pbc import min_nn_pbc
from inference.visualization.comparison import plot_comparison_frame


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


def _plot_timeseries(ts, metrics_by_k, metrics_tv, save_path):
    """Plot mean nn and CV over time for all K values vs TV."""
    C_TV  = '#e67e22'
    BLUES = ['#aed6f1', '#5dade2', '#2471a3', '#1a3a5c']

    k_values = sorted(metrics_by_k.keys())
    colors   = {k: BLUES[min(i, len(BLUES)-1)] for i, k in enumerate(k_values)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.13, wspace=0.30)

    for k in k_values:
        ax1.plot(ts, metrics_by_k[k]['mean'], color=colors[k], lw=1.8, label=f'K={k}')
        ax2.plot(ts, metrics_by_k[k]['cv'],   color=colors[k], lw=1.8, label=f'K={k}')

    ax1.plot(ts, metrics_tv['mean'], color=C_TV, lw=1.8, ls='--', label='TV')
    ax2.plot(ts, metrics_tv['cv'],   color=C_TV, lw=1.8, ls='--', label='TV')

    k_best = max(k_values)
    k3 = np.array(metrics_by_k[k_best]['mean'])
    tv = np.array(metrics_tv['mean'])
    ax1.fill_between(ts, k3, tv, where=(k3 >= tv), alpha=0.12,
                     color=colors[k_best], label=f'K={k_best} ahead')

    for ax, ylabel, title in [
        (ax1, 'mean min-nn distance',    'Mean NN-distance over time'),
        (ax2, 'CV = std/mean (min-nn)',  'Distribution uniformity  (lower = better)'),
    ]:
        ax.set_xlabel('timestep', fontsize=9)
        ax.set_ylabel(ylabel,     fontsize=9)
        ax.set_title(title,       fontsize=10)
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(ts[0], ts[-1])

    fig.suptitle('SPH corrector vs TV  —  trajectory summary', fontsize=10)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Timeseries saved -> {save_path}')


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
        _plot_timeseries(ts_list, metrics_by_k, metrics_tv,
                         exp_dir / 'timeseries.png')

    log.info(f'Done. Results in {exp_dir}')


if __name__ == '__main__':
    main()
