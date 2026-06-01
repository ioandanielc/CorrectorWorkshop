"""
sph_timeseries.py
-----------------
2x2 figure:
  (0,0)  Mean nn-distance over time for four approaches
  (0,1)  CV = std/mean of nn-distances over time (uniformity)
  (1,0)  nn-distance histogram at t=0 (initial snapshot)
  (1,1)  Radial distribution function at t=0 (initial snapshot)

Approaches compared:
  - nonTV original     (no correction, no TV)
  - nonTV + model K=1
  - nonTV + model K=3
  - TV variant

Usage:
    .venv\\Scripts\\python.exe inference/sph_timeseries.py
    .venv\\Scripts\\python.exe inference/sph_timeseries.py --stride 25
"""
import argparse, sys, importlib
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.data_processor import DataProcessor
from utils.config import load_model_config
from inference.sph_viz import min_nn_pbc, apply_corrector, compute_rdf

CKPT_DEFAULT = Path("training_artifacts/train_run_2026-05-28_14-53-45/model_final.pt")
CFG_DEFAULT  = Path("configs/model_configs/model_config_9_n100_p050.yaml")
DATA_WITHOUT = Path("inference/sph_data/positions_without.npy")
DATA_WITH    = Path("inference/sph_data/positions.npy")
RD_TRAIN     = 0.076
RD_TEST      = 0.02
SCALE        = RD_TRAIN / RD_TEST
DOMAIN       = 1.0

C_ORIG  = '#888888'
C_K1    = '#5dade2'
C_K3    = '#1a5276'
C_TV    = '#e67e22'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stride',     type=int, default=50,
                        help='Sample every STRIDE timesteps (default: 50)')
    parser.add_argument('--checkpoint', default=str(CKPT_DEFAULT))
    parser.add_argument('--config',     default=str(CFG_DEFAULT))
    parser.add_argument('--without',    default=str(DATA_WITHOUT))
    parser.add_argument('--with_tv',    default=str(DATA_WITH))
    parser.add_argument('--save',       default='inference/sph_timeseries.png')
    args = parser.parse_args()

    device = torch.device('cpu')

    model_cfg = load_model_config(args.config)
    m         = importlib.import_module('models.fixed_rd.model9')
    model     = m.CorrectorModel(model_cfg, input_dim=2, initialization='xavier_uniform')
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model.eval()
    rd_t      = torch.tensor(RD_TRAIN, dtype=torch.float32)
    processor = DataProcessor()

    pos_wo = np.load(args.without)
    pos_wi = np.load(args.with_tv)
    T, N, _ = pos_wo.shape
    timesteps = list(range(0, T, args.stride))
    print(f'T={T}  N={N}  sampling {len(timesteps)} timesteps (stride={args.stride})')

    results = {k: [] for k in ('t', 'orig_mean', 'orig_cv',
                                'k1_mean', 'k1_cv',
                                'k3_mean', 'k3_cv',
                                'tv_mean',  'tv_cv')}

    for i, t in enumerate(timesteps):
        print(f'  [{i+1}/{len(timesteps)}] t={t}', end='  ', flush=True)
        pts_wo = pos_wo[t].astype(np.float32)
        pts_wi = pos_wi[t].astype(np.float32)

        nn_orig = min_nn_pbc(pts_wo)
        nn_tv   = min_nn_pbc(pts_wi)

        corr_k1 = apply_corrector(pts_wo, model, rd_t, SCALE, processor, device, k=1)
        corr_k3 = apply_corrector(pts_wo, model, rd_t, SCALE, processor, device, k=3)
        nn_k1   = min_nn_pbc(corr_k1)
        nn_k3   = min_nn_pbc(corr_k3)

        results['t'].append(t)
        for key, nn in [('orig', nn_orig), ('k1', nn_k1),
                         ('k3', nn_k3),    ('tv', nn_tv)]:
            results[f'{key}_mean'].append(float(nn.mean()))
            results[f'{key}_cv'].append(float(nn.std() / nn.mean()))

        print(f'orig={nn_orig.mean():.4f}  K1={nn_k1.mean():.4f}  '
              f'K3={nn_k3.mean():.4f}  TV={nn_tv.mean():.4f}')

    ts = np.array(results['t'])

    # ── t=0 snapshot: histogram + RDF ─────────────────────────────────────────
    print('\nComputing t=0 snapshot for histogram and RDF...')
    pts_wo0 = pos_wo[0].astype(np.float32)
    pts_wi0 = pos_wi[0].astype(np.float32)
    corr_k1_0 = apply_corrector(pts_wo0, model, rd_t, SCALE, processor, device, k=1)
    corr_k3_0 = apply_corrector(pts_wo0, model, rd_t, SCALE, processor, device, k=3)

    nn0 = {
        'orig': min_nn_pbc(pts_wo0),
        'k1':   min_nn_pbc(corr_k1_0),
        'k3':   min_nn_pbc(corr_k3_0),
        'tv':   min_nn_pbc(pts_wi0),
    }
    r_max = RD_TEST * 5.0
    rdf0 = {
        'orig': compute_rdf(pts_wo0,   r_max),
        'k1':   compute_rdf(corr_k1_0, r_max),
        'k3':   compute_rdf(corr_k3_0, r_max),
        'tv':   compute_rdf(pts_wi0,   r_max),
    }

    # ── 1x4 figure ─────────────────────────────────────────────────────────────
    fig, (ax_mean, ax_cv, ax_hist, ax_rdf) = plt.subplots(1, 4, figsize=(24, 5.5))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.13, wspace=0.32)

    # ── (0,0) mean nn over time ────────────────────────────────────────────────
    ax_mean.plot(ts, results['orig_mean'], color=C_ORIG, lw=1.5, ls=':',
                 label='nonTV  (no correction)')
    ax_mean.plot(ts, results['k1_mean'],   color=C_K1,  lw=1.8,
                 label='nonTV + model  K=1')
    ax_mean.plot(ts, results['k3_mean'],   color=C_K3,  lw=2.2,
                 label='nonTV + model  K=3')
    ax_mean.plot(ts, results['tv_mean'],   color=C_TV,  lw=1.8, ls='--',
                 label='TV variant')
    k3 = np.array(results['k3_mean']); tv = np.array(results['tv_mean'])
    ax_mean.fill_between(ts, k3, tv, where=(k3 >= tv),
                         alpha=0.13, color=C_K3, label='K=3 ahead of TV')
    ax_mean.set_xlabel('timestep', fontsize=9)
    ax_mean.set_ylabel('mean min-nn distance', fontsize=9)
    ax_mean.set_title('Mean nearest-neighbour distance over time', fontsize=10)
    ax_mean.legend(fontsize=8, framealpha=0.9)
    ax_mean.grid(True, alpha=0.3); ax_mean.set_xlim(ts[0], ts[-1])

    # ── (0,1) CV over time ────────────────────────────────────────────────────
    ax_cv.plot(ts, results['orig_cv'], color=C_ORIG, lw=1.5, ls=':',
               label='nonTV  (no correction)')
    ax_cv.plot(ts, results['k1_cv'],   color=C_K1,  lw=1.8,
               label='nonTV + model  K=1')
    ax_cv.plot(ts, results['k3_cv'],   color=C_K3,  lw=2.2,
               label='nonTV + model  K=3')
    ax_cv.plot(ts, results['tv_cv'],   color=C_TV,  lw=1.8, ls='--',
               label='TV variant')
    k3_cv = np.array(results['k3_cv']); tv_cv = np.array(results['tv_cv'])
    ax_cv.fill_between(ts, tv_cv, k3_cv, where=(k3_cv <= tv_cv),
                       alpha=0.13, color=C_K3, label='K=3 more uniform than TV')
    ax_cv.set_xlabel('timestep', fontsize=9)
    ax_cv.set_ylabel('CV = std / mean  (min-nn)', fontsize=9)
    ax_cv.set_title('Distribution uniformity over time  (lower = more uniform)', fontsize=10)
    ax_cv.legend(fontsize=8, framealpha=0.9)
    ax_cv.grid(True, alpha=0.3); ax_cv.set_xlim(ts[0], ts[-1])

    # ── (1,0) nn histogram at t=0 ─────────────────────────────────────────────
    all0 = np.concatenate(list(nn0.values()))
    lo0  = max(0, all0.min() * 0.97); hi0 = all0.max() * 1.03
    bins0 = np.linspace(lo0, hi0, 60)
    for key, color, ls, lw, lbl in [
        ('orig', C_ORIG, ':',  1.5, 'nonTV original'),
        ('k1',   C_K1,  '-',  1.6, 'nonTV + model  K=1'),
        ('k3',   C_K3,  '-',  2.0, 'nonTV + model  K=3'),
        ('tv',   C_TV,  '--', 1.6, 'TV variant'),
    ]:
        ax_hist.hist(nn0[key], bins=bins0, color=color, histtype='step',
                     lw=lw, ls=ls, label=lbl, density=True)
    ax_hist.set_xlabel('min nearest-neighbour distance', fontsize=9)
    ax_hist.set_ylabel('density', fontsize=9)
    ax_hist.set_title('nn-distance distribution at t = 0', fontsize=10)
    ax_hist.legend(fontsize=8, framealpha=0.9)
    ax_hist.grid(True, alpha=0.3)

    # ── (1,1) RDF at t=0 ──────────────────────────────────────────────────────
    for key, color, ls, lw, lbl in [
        ('orig', C_ORIG, ':',  1.5, 'nonTV original'),
        ('k1',   C_K1,  '-',  1.6, 'nonTV + model  K=1'),
        ('k3',   C_K3,  '-',  2.0, 'nonTV + model  K=3'),
        ('tv',   C_TV,  '--', 1.6, 'TV variant'),
    ]:
        r, g = rdf0[key]
        ax_rdf.plot(r, g, color=color, lw=lw, ls=ls, label=lbl, alpha=0.85)
    ax_rdf.axhline(1.0, color='#bbb', lw=0.8, ls=':')
    ax_rdf.set_xlabel('r', fontsize=9)
    ax_rdf.set_ylabel('g(r)', fontsize=9)
    ax_rdf.set_title('Radial distribution function at t = 0', fontsize=10)
    ax_rdf.legend(fontsize=8, framealpha=0.9)
    ax_rdf.grid(True, alpha=0.3)
    ax_rdf.set_xlim(0, r_max); ax_rdf.set_ylim(bottom=0)

    fig.suptitle(
        f'SPH trajectory analysis: nonTV + corrector vs TV   '
        f'(N=2500,  scale={SCALE:.1f},  6x6 grid ~107 pts/tile,  stride={args.stride})',
        fontsize=10,
    )

    fig.savefig(args.save, dpi=150, bbox_inches='tight')
    print(f'\nSaved -> {args.save}')


if __name__ == '__main__':
    main()
