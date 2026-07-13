"""
apply_corrector.py
------------------
Apply the trained ML corrector to every timestep of the no-TV SPH trajectory
and save the corrected positions.

Output
------
  inference/corrected/positions_corrected_K{k}.npy   shape (T, N, 2) float32
  inference/corrected/positions_corrected_K{k}.txt   human-readable, one block per timestep

Usage
-----
    .venv\\Scripts\\python.exe inference/apply_corrector.py
    .venv\\Scripts\\python.exe inference/apply_corrector.py --k 3
    .venv\\Scripts\\python.exe inference/apply_corrector.py --config inference/configs/grid_6x6.yaml --k 5
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.pipeline.corrector import GridCorrector2D, GridCorrector2DConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='inference/configs/grid_6x6.yaml')
    parser.add_argument('--k',      type=int, default=5,
                        help='Number of corrector passes (default 5)')
    args = parser.parse_args()

    cfg = GridCorrector2DConfig.from_yaml(args.config)

    pos_without = np.load('inference/sph_data/positions_without.npy')
    T, N, D = pos_without.shape
    print(f'Loaded positions_without.npy  T={T}  N={N}  D={D}')
    print(f'Config: {args.config}  K={args.k}  device={cfg.device}  '
          f'grid={cfg.n_cells}x{cfg.n_cells}  '
          f'scale={cfg.rd_train/cfg.rd_test:.2f}')

    corrector = GridCorrector2D(cfg)
    print('Corrector ready. Running...\n')

    corrected = np.empty_like(pos_without)

    t_start = time.perf_counter()
    for t in range(T):
        corrected[t] = corrector.apply(pos_without[t], k=args.k)
        if (t + 1) % 100 == 0 or t == T - 1:
            elapsed = time.perf_counter() - t_start
            rate    = (t + 1) / elapsed
            eta     = (T - t - 1) / rate
            print(f'  t={t+1:4d}/{T}  {elapsed:.1f}s elapsed  '
                  f'{rate:.1f} frames/s  ETA {eta:.1f}s')

    total = time.perf_counter() - t_start
    print(f'\nDone. {T} timesteps in {total:.1f}s  ({T/total:.1f} frames/s)')

    # ── save ──────────────────────────────────────────────────────────────────
    out_dir = Path('inference/corrected')
    out_dir.mkdir(exist_ok=True)

    stem    = f'positions_corrected_K{args.k}'
    npy_out = out_dir / f'{stem}.npy'
    txt_out = out_dir / f'{stem}.txt'

    np.save(npy_out, corrected)
    print(f'Saved -> {npy_out}  ({corrected.nbytes / 1e6:.1f} MB)')

    with open(txt_out, 'w') as f:
        f.write(f'# ML-corrected SPH positions\n')
        f.write(f'# Source: inference/sph_data/positions_without.npy\n')
        f.write(f'# Model : n100_p050  (train_run_2026-05-28_14-53-45)\n')
        f.write(f'# Config: {args.config}\n')
        f.write(f'# Tiling: {cfg.n_cells}x{cfg.n_cells} grid  '
                f'ghost_factor={cfg.ghost_factor}  '
                f'scale={cfg.rd_train/cfg.rd_test:.2f}\n')
        f.write(f'# Passes: K={args.k}\n')
        f.write(f'# Shape : T={T}  N={N}  D={D}\n')
        f.write(f'# Format: one block per timestep, N lines of "x y"\n')
        f.write(f'#\n')
        for t in range(T):
            f.write(f'# t={t}\n')
            np.savetxt(f, corrected[t], fmt='%.8f')

    print(f'Saved -> {txt_out}  ({txt_out.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
