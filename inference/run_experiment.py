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
import time
from datetime import datetime
import matplotlib
matplotlib.use('Agg')   # file-only rendering; must be set before any plt import
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.pipeline.corrector import Corrector, CorrectorConfig
from inference.pipeline.pbc import min_nn_pbc
from inference.pipeline.tv_corrector import TVCorrector, FastTVCorrector
from inference.visualization.comparison import plot_comparison_frame
from inference.visualization.timeseries import plot_timeseries
from inference.visualization.ml_vs_tv import plot_ml_vs_tv, save_report


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
    log.info(f'Grid: {cfg.n_cells}x{cfg.n_cells}  '
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

    # ── TV algorithm corrector ───────────────────────────────────────────────
    with open(args.config) as _f:
        _raw = yaml.safe_load(_f)
    _tv_cfg = _raw.get('tv', {})
    _tv_kw = dict(
        h_factor = float(_tv_cfg.get('h_factor', 1.3)),
        nmax     = int(_tv_cfg.get('nmax', 10)),
        dt       = float(_tv_cfg.get('dt', 0.2)),
    )
    tv_naive     = TVCorrector(**_tv_kw)
    tv_corrector = FastTVCorrector(**_tv_kw)   # fast version used in main loop
    log.info(f'TV corrector (fast): h_factor={tv_corrector.h_factor:.4f}  nmax={tv_corrector.nmax}'
             f'  dt={tv_corrector.dt}')

    # ── timesteps ────────────────────────────────────────────────────────────
    if args.timestep is not None:
        timesteps = [args.timestep % T]
    else:
        timesteps = list(range(0, T, cfg.stride))

    ts_list          = []
    metrics_by_k     = {k: {'mean': [], 'cv': []} for k in cfg.k_values}
    metrics_tv       = {'mean': [], 'cv': []}
    metrics_tv_algo  = {'mean': [], 'cv': []}
    timings_k        = {k: [] for k in cfg.k_values}   # collect per-timestep for median
    timings_tv_fast  = []
    t_naive_ref      = None

    for i, t in enumerate(timesteps):
        log.info(f'[{i+1}/{len(timesteps)}]  t={t}')
        pts_wo = pos_without[t].astype(np.float32)
        pts_wi = pos_with[t].astype(np.float32)

        corrected_by_k = {}
        time_by_k      = {}
        for k in cfg.k_values:
            t0   = time.perf_counter()
            corr = corrector.apply(pts_wo, k=k)
            time_by_k[k] = time.perf_counter() - t0
            if i > 0:               # skip warmup at t=0
                timings_k[k].append(time_by_k[k] * 1000)
            corrected_by_k[k] = corr
            nn = min_nn_pbc(corr)
            metrics_by_k[k]['mean'].append(float(nn.mean()))
            metrics_by_k[k]['cv'].append(float(nn.std() / nn.mean()))
            log.info(f'  K={k}: mean_nn={nn.mean():.5f}  ({time_by_k[k]:.2f}s)')

        nn_tv = min_nn_pbc(pts_wi)
        metrics_tv['mean'].append(float(nn_tv.mean()))
        metrics_tv['cv'].append(float(nn_tv.std() / nn_tv.mean()))
        log.info(f'  TV sim:  mean_nn={nn_tv.mean():.5f}')

        t0      = time.perf_counter()
        pts_tva = tv_corrector.apply(pts_wo)
        t_tva   = time.perf_counter() - t0
        nn_tva  = min_nn_pbc(pts_tva)
        metrics_tv_algo['mean'].append(float(nn_tva.mean()))
        metrics_tv_algo['cv'].append(float(nn_tva.std() / nn_tva.mean()))
        if i > 0:
            timings_tv_fast.append(t_tva * 1000)
        log.info(f'  TV fast  (n={tv_corrector.nmax}): mean_nn={nn_tva.mean():.5f}'
                 f'  ({t_tva:.3f}s)')

        # On first timestep, also time the naive O(N²) implementation for reference
        if i == 0:
            t0_naive = time.perf_counter()
            tv_naive.apply(pts_wo)
            t_naive_ref = (time.perf_counter() - t0_naive) * 1000
            log.info(f'  TV naive (n={tv_naive.nmax}): O(N²) reference'
                     f'  ({t_naive_ref/1000:.3f}s)  speedup={t_naive_ref/t_tva/1000:.0f}×')
        ts_list.append(t)

        save = frames_dir / f't{t:04d}.png'
        plot_comparison_frame(
            pts_wo, corrected_by_k, pts_wi,
            timestep=t,
            grid=cfg.n_cells,
            scale=cfg.rd_train / cfg.rd_test,
            rd=cfg.rd_test,
            time_by_k=time_by_k,
            save_path=str(save),
        )

    # ── timeseries plot ──────────────────────────────────────────────────────
    if len(ts_list) > 1:
        plot_timeseries(ts_list, metrics_by_k, metrics_tv,
                        metrics_tv_algo=metrics_tv_algo,
                        tv_algo_label=f'TV algo (n={tv_corrector.nmax})',
                        save_path=str(exp_dir / 'timeseries.png'), show=False)
        print(f'Timeseries saved -> {exp_dir / "timeseries.png"}')

    # ── ML vs TV comparison figure + report ─────────────────────────────────
    if len(ts_list) > 1 and timings_k[cfg.k_values[0]]:
        k_vals = cfg.k_values
        # Median timings (ms), skip warmup at t=0
        timings_ms = {f'K={k} (GPU)': float(np.median(timings_k[k])) for k in k_vals}
        timings_ms['TV fast (CPU)']  = float(np.median(timings_tv_fast)) if timings_tv_fast else 36.0
        if t_naive_ref is not None:
            timings_ms['TV naive (CPU)'] = float(t_naive_ref)

        plot_ml_vs_tv(
            ts_list, metrics_by_k, metrics_tv, metrics_tv_algo,
            timings_ms=timings_ms,
            rd=cfg.rd_test,
            save_path=str(exp_dir / 'ml_vs_tv.png'),
        )

        save_report(
            str(exp_dir / 'report.txt'),
            ts_list, metrics_by_k, metrics_tv, metrics_tv_algo,
            timings_ms=timings_ms,
            rd=cfg.rd_test,
        )

    log.info(f'Done. Results in {exp_dir}')


if __name__ == '__main__':
    main()
