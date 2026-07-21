"""
kg_sweep.py

Persist the full-trajectory KG comparison: for every timestep of the SPH
trajectory, compute mean|KG| (SPH kernel-gradient asymmetry), mean nn-distance
and illegal fraction for four precomputed series:

  raw          data/positions_without.npy                       (non-TV input)
  tv           data/positions.npy                               (TV baseline)
  model9_K5    ../apply_corrector/runs/positions_corrected_K5.npy (model9 grid, K=5)
  model12_wc   for_sim/positions_model12_corrected.npy          (whole-cloud sparse, k=5)

No corrector is run — this only measures existing artifacts. Output:
artifacts/inference/experiments/sph_tv/runs/kg_sweep_<timestamp>/
  metrics.csv   one row per timestep, wide format (t, <series>_{kg,nn,ill} ...)
  report.txt    provenance + disordered-regime (t >= 300) aggregates

Run:
    .venv\\Scripts\\python.exe src/inference/experiments/sph_tv/kg_sweep.py
    .venv\\Scripts\\python.exe src/inference/experiments/sph_tv/kg_sweep.py --stride 10
"""
import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src'))

from utils.metrics import kg_norm, nn_dists

RD_TEST  = 0.02
H_FACTOR = 2.0   # h = h_factor * dx, dx = 0.02 -> support 3h = 0.12 (see model12_grid.yaml)
BOX      = 1.0   # unit torus

EXP = ROOT / 'artifacts/inference/experiments'
SERIES = {  # name -> trajectory file, all (T, N, 2) in the unit periodic box
    'raw':        EXP / 'sph_tv/data/positions_without.npy',
    'tv':         EXP / 'sph_tv/data/positions.npy',
    'model9_K5':  EXP / 'apply_corrector/runs/positions_corrected_K5.npy',
    'model12_wc': EXP / 'sph_tv/for_sim/positions_model12_corrected.npy',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    device = torch.device(args.device)

    trajs = {name: np.load(path, mmap_mode='r') for name, path in SERIES.items()}
    T = min(t.shape[0] for t in trajs.values())
    ts = list(range(0, T, args.stride))

    out_dir = (EXP / 'sph_tv/runs' /
               f'kg_sweep_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    out_dir.mkdir(parents=True, exist_ok=True)

    names = list(SERIES)
    cols = ['t'] + [f'{n}_{m}' for n in names for m in ('kg', 'nn', 'ill')]
    rows = []
    t0 = time.perf_counter()
    # incremental write: a killed run keeps every row computed so far
    with open(out_dir / 'metrics.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, t in enumerate(ts):
            clouds = [np.asarray(trajs[n][t], dtype=np.float32) for n in names]
            with torch.no_grad():   # one batched call for all series
                kgs = kg_norm(torch.tensor(np.stack(clouds), device=device),
                              h_factor=H_FACTOR, box=BOX).cpu().numpy()
            row = [t]
            for n, p, kg_val in zip(names, clouds, kgs):
                nn = nn_dists(p, box=BOX)
                row += [float(kg_val), float(nn.mean()), float((nn < RD_TEST).mean())]
            rows.append(row)
            w.writerow(row)
            f.flush()
            if i % 100 == 0:
                print(f'  t={t:4d}/{T}  ({(time.perf_counter()-t0)/(i+1)*1000:.0f} ms/step)',
                      flush=True)

    # disordered-regime aggregates (t >= 300; ordered/low-KG frames excluded)
    a = np.array(rows)
    dis = a[a[:, 0] >= 300]
    with open(out_dir / 'report.txt', 'w') as f:
        f.write(f'kg_sweep  {datetime.now():%Y-%m-%d %H:%M}  stride={args.stride}  '
                f'T={T}  rd_test={RD_TEST}  h_factor={H_FACTOR}  box={BOX}\n\n')
        for n, path in SERIES.items():
            f.write(f'  {n:11}  {path.relative_to(ROOT)}\n')
        f.write(f'\nDisordered regime (t >= 300, {len(dis)} timesteps), means:\n')
        f.write(f'{"series":>11} {"mean|KG|":>9} {"mean_nn":>9} {"ill%":>7}\n')
        for j, n in enumerate(names):
            f.write(f'{n:>11} {dis[:, 1+3*j].mean():>9.4f} '
                    f'{dis[:, 2+3*j].mean():>9.5f} {100*dis[:, 3+3*j].mean():>6.1f}%\n')
        wc = dis[:, 1 + 3*names.index('model12_wc')]
        f.write(f'\nmodel12_wc KG over t>=300: min {wc.min():.4f}  '
                f'p10 {np.percentile(wc, 10):.4f}  (floor estimate)\n')

    print(f'\nSaved -> {out_dir}  ({time.perf_counter()-t0:.0f}s)')
    with open(out_dir / 'report.txt') as f:
        print(f.read())


if __name__ == '__main__':
    main()
