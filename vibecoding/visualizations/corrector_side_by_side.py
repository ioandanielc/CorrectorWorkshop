"""
Side-by-side corrector comparisons, two figures.

  fig1  one N=49 synthetic cloud through every architecture (K=5), with the
        displacement field drawn as arrows. This is where the uniform-translation
        failure becomes obvious by eye: the degenerate arms produce a field of
        parallel arrows of identical length, leaving relative geometry untouched —
        which is exactly why |KG| and illegal% read "unchanged" for them.

  fig2  SPH trajectory timestep 1000, N=2500: raw / Transport Velocity / model12.
        Only model12 appears because it is the only architecture here with a sparse
        path; the dense-only baselines cannot process a 2500-point cloud at all.

    .venv\\Scripts\\python.exe vibecoding/visualizations/corrector_side_by_side.py
"""
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from training.datagen import PoissonDiskDataset
from utils.metrics import mean_kg, nn_dists

OUT = Path(__file__).resolve().parent / 'outputs'
RUNS = ROOT / 'artifacts/training'

ARMS = [   # label, run dir, tag shown in the panel title
    ('model12',  'train_run_2026-08-10_10-40-18'),
    ('GNS',      'train_run_2026-08-10_11-16-59'),
    ('DGCNN',    'train_run_2026-08-10_10-51-22'),
    ('PointNet', 'train_run_2026-08-10_11-01-18'),
]
RD, CUTOFF, BOX, K = 0.14, 0.286, 1.0, 5


def load_model(run):
    cfg = None
    for p in (RUNS / run / 'configs').glob('*.yaml'):
        c = yaml.safe_load(open(p))
        if isinstance(c, dict) and 'model_file' in c:
            cfg = c
    if cfg is None:
        raise FileNotFoundError(f'no model config in {run}/configs')
    mod = importlib.import_module(cfg['model_file'].replace('/', '.'))
    net = mod.CorrectorModel(cfg, input_dim=2, initialization='xavier_uniform')
    net.load_state_dict(torch.load(RUNS / run / 'model_best.pt', map_location='cpu'))
    return net.eval(), cfg


def apply_k(net, cfg, x, k=K):
    kw = {'box': BOX} if getattr(net, 'uses_box', False) else {}
    rd_t = torch.tensor(float(cfg.get('cutoff_rd', RD)))
    with torch.no_grad():
        for _ in range(k):
            d = net(x, rd=rd_t, **kw) if getattr(net, 'uses_rd', False) else net(x)
            x = x + d
    return x


def draw_cloud(ax, pts, rd, box, title, ref=None):
    """Points coloured by nn-distance; violating points ringed; drift arrows vs ref."""
    nn = nn_dists(pts, box=box)
    if ref is not None:
        d = pts - ref
        d -= box * np.round(d / box)                    # shortest path on the torus
        ax.quiver(ref[:, 0], ref[:, 1], d[:, 0], d[:, 1], angles='xy',
                  scale_units='xy', scale=1, width=0.004,
                  color='#999999', alpha=0.8, zorder=1)
    sc = ax.scatter(pts[:, 0] % box, pts[:, 1] % box, c=nn, cmap='viridis',
                    vmin=0.5 * rd, vmax=1.3 * rd, s=90, zorder=3,
                    edgecolors='white', linewidths=0.6)
    bad = nn < rd
    if bad.any():
        ax.scatter(pts[bad, 0] % box, pts[bad, 1] % box, facecolors='none',
                   edgecolors='#e74c3c', s=220, linewidths=1.6, zorder=4)
    ax.set_xlim(0, box); ax.set_ylim(0, box); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    return sc


def figure_one_cloud():
    ds = PoissonDiskDataset(dim=2, cardinality=49, rd=RD, seed=7,
                            noise_scale_min=0.0, noise_scale_max=0.6 * RD, periodic=True)
    noisy = ds.noise_sample(ds.generate_sample(1)).astype(np.float32)
    x0 = torch.tensor(noisy)
    base = noisy[0]

    fig, axes = plt.subplots(1, len(ARMS) + 1, figsize=(4.0 * (len(ARMS) + 1), 4.4))
    sc = draw_cloud(axes[0], base, RD, BOX,
                    f'input\nillegal {100 * (nn_dists(base, BOX) < RD).mean():.0f}%  '
                    f'|KG| {mean_kg(base, box=BOX):.3f}')
    for ax, (label, run) in zip(axes[1:], ARMS):
        net, cfg = load_model(run)
        out = apply_k(net, cfg, x0)[0].numpy() % BOX
        disp = out - base
        disp -= BOX * np.round(disp / BOX)
        drift = np.linalg.norm(disp.mean(0)) / (np.linalg.norm(disp, axis=1).mean() + 1e-12)
        draw_cloud(ax, out, RD, BOX,
                   f'{label}  (K={K})\n'
                   f'illegal {100 * (nn_dists(out, BOX) < RD).mean():.0f}%  '
                   f'|KG| {mean_kg(out, box=BOX):.3f}\ndrift {100 * drift:.0f}%',
                   ref=base)
    fig.colorbar(sc, ax=axes, fraction=0.012, pad=0.01, label='nn distance')
    fig.suptitle('One N=49 cloud through each architecture — arrows are the displacement field; '
                 'red rings mark points closer than rd', fontsize=11)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / 'side_by_side_n49.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'-> {OUT / "side_by_side_n49.png"}')


def figure_sph(t=1000, crop=0.35):
    exp = ROOT / 'artifacts/inference/experiments'
    series = [('raw', exp / 'sph_tv/data/positions_without.npy'),
              ('Transport Velocity', exp / 'sph_tv/data/positions.npy'),
              ('model12 (whole-cloud, k=5)', exp / 'sph_tv/for_sim/positions_model12_corrected.npy')]
    rd_test = 0.02

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.6),
                             gridspec_kw={'height_ratios': [1, 1]})
    for j, (label, path) in enumerate(series):
        pts = np.asarray(np.load(path, mmap_mode='r')[t], dtype=np.float32) % 1.0
        nn = nn_dists(pts, box=1.0)
        kg = mean_kg(pts, box=1.0)

        for i, ax in enumerate(axes[:, j]):
            m = np.ones(len(pts), bool) if i == 0 else (
                (pts[:, 0] < crop) & (pts[:, 1] < crop))
            s = 2.0 if i == 0 else 44.0
            ax.scatter(pts[m, 0], pts[m, 1], c=nn[m], cmap='viridis',
                       vmin=0.6 * rd_test, vmax=1.2 * rd_test, s=s,
                       edgecolors='none' if i == 0 else 'white', linewidths=0.4)
            lim = 1.0 if i == 0 else crop
            ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f'{label}\nmean nn {nn.mean():.5f}   |KG| {kg:.4f}', fontsize=11)
            else:
                ax.set_title(f'zoom {crop}x{crop}', fontsize=9)

    fig.suptitle(f'SPH trajectory, timestep {t}, N=2500 — colour is nn distance '
                 f'(rd_test = {rd_test}); model12 is the only architecture with a '
                 f'sparse path able to process the whole cloud', fontsize=12)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'side_by_side_sph_t{t}.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'-> {OUT / f"side_by_side_sph_t{t}.png"}')


def figure_displacement_decomposition(n_clouds=64):
    """Split each architecture's displacement into bulk drift + local restructuring.

    Every loss term is translation-invariant, so the bulk component does no work on
    the objective — it is pure waste, and it is invisible to |KG| and illegal%. The
    two rows separate what each model is actually doing.
    """
    ds = PoissonDiskDataset(dim=2, cardinality=49, rd=RD, seed=1234,
                            noise_scale_min=0.0, noise_scale_max=0.6 * RD, periodic=True)
    batch = ds.noise_sample(ds.generate_sample(n_clouds)).astype(np.float32)
    x = torch.tensor(batch)
    base = batch[0]

    fig, axes = plt.subplots(2, len(ARMS), figsize=(4.0 * len(ARMS), 8.4))
    for j, (label, run) in enumerate(ARMS):
        net, cfg = load_model(run)
        with torch.no_grad():
            kw = {'box': BOX} if getattr(net, 'uses_box', False) else {}
            rd_t = torch.tensor(float(cfg.get('cutoff_rd', RD)))
            disp = (net(x, rd=rd_t, **kw) if getattr(net, 'uses_rd', False) else net(x))
        bulk = disp.mean(dim=1, keepdim=True)                  # (B,1,2) translation
        local = disp - bulk
        drift = (bulk.norm(dim=-1).mean() / (disp.norm(dim=-1).mean() + 1e-12)).item()

        d0, b0, l0 = disp[0].numpy(), bulk[0, 0].numpy(), local[0].numpy()
        for i, (field, name) in enumerate(((d0, 'total'), (l0, 'local (bulk removed)'))):
            ax = axes[i, j]
            ax.quiver(base[:, 0], base[:, 1], field[:, 0], field[:, 1],
                      angles='xy', scale_units='xy', scale=1, width=0.006,
                      color='#1f77b4' if i else '#444444')
            ax.scatter(base[:, 0], base[:, 1], s=14, c='#cccccc', zorder=0)
            ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1); ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f'{label}\ndrift {100 * drift:.0f}% of motion '
                             f'(|bulk| {np.linalg.norm(b0):.3f})', fontsize=10)
            else:
                ax.set_title(f'{name}   mean |local| '
                             f'{np.linalg.norm(l0, axis=1).mean():.4f}', fontsize=9)

    fig.suptitle('Displacement decomposed, one pass, identical cloud and axis scale. '
                 'Top: total motion. Bottom: after removing the bulk translation — '
                 'what the model actually contributes, since every loss term is '
                 'translation-invariant.', fontsize=11)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / 'displacement_decomposition.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'-> {OUT / "displacement_decomposition.png"}')


if __name__ == '__main__':
    figure_one_cloud()
    figure_sph()
    figure_displacement_decomposition()
