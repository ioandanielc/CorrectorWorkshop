"""
score_arm.py
------------
Scores one trained checkpoint into rows of `paper/results.csv` — the single
backbone every ablation exhibit reads from.

A training run directory already carries everything needed (`configs/` holds the
dataset/model/loss YAMLs the run used), so an arm is identified by its run dir:

    .venv\\Scripts\\python.exe src/inference/experiments/ablations/score_arm.py ^
        --run-dir artifacts/training/train_run_2026-08-10_10-40-18 ^
        --arm model12_n49 --deploy-n 16,25,49,100,196 --trajectory

Two regimes:

  synthetic   Held-out clouds at any cardinality. Cross-N holds DENSITY fixed and
              grows the box, so local geometry is identical to training and only
              cardinality/extent change — the protocol that separates size from
              density. Reports violation / illegal% / nn / |KG| pre, K=1 and K=5.

  trajectory  The real N=2500 SPH trajectory through WholeCloudCorrector2D.
              Needs a model with forward_sparse, so it is opt-in (--trajectory)
              and unavailable to the dense-only baselines.

All distances are minimum-image. Note this differs from the trainer's inline eval
block, which uses non-periodic `cdist`: on a torus that inflates nn distances and
hides wrap-seam violations. Numbers here are therefore NOT directly comparable to
the trainer log lines.
"""
import argparse
import csv
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src'))

from training.datagen import PoissonDiskDataset
from utils.metrics import _pbc_rel, kg_norm, nn_dists

EVAL_SEED   = 1234    # fixed: every arm sees identical evaluation clouds
EVAL_CLOUDS = 64
MODEL_CHUNK = 16      # clouds per forward call (dense edges are (B, N, N, .))

FIELDS = [
    'arm', 'regime', 'train_N', 'deploy_N', 'noise_ratio', 'n_params',
    'viol_pre', 'viol_k1', 'viol_k5',
    'ill_pct_pre', 'ill_pct_k1', 'ill_pct_k5',
    'nn_pre', 'nn_k1', 'nn_k5',
    'kg_pre', 'kg_k1', 'kg_k5',
    'disp_k1', 'disp_k5', 'viol_red_k1', 'viol_red_k5',
    'knn_keep_k5', 'uniform_frac_k1', 'nn_cv_pre', 'nn_cv_k5',
    'checkpoint',
]


def load_run_configs(run_dir: Path) -> dict:
    """Classify a run's config snapshot by content, not filename (names vary per arm)."""
    out = {}
    for path in sorted((run_dir / 'configs').glob('*.yaml')):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        if 'model_file' in cfg:
            out['model'] = cfg
        elif 'points_per_cloud' in cfg:
            out['dataset'] = cfg
        elif 'batch_size' in cfg:
            out['train'] = cfg
        elif 'params' in cfg:
            out['loss'] = cfg
    missing = {'model', 'dataset', 'loss'} - set(out)
    if missing:
        raise ValueError(f'{run_dir}/configs is missing {sorted(missing)}')
    return out


def build_model(model_cfg, dim, checkpoint, device):
    module = importlib.import_module(model_cfg['model_file'].replace('/', '.'))
    model = module.CorrectorModel(model_cfg, input_dim=dim, initialization='xavier_uniform')
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    return model.to(device).eval()


def eval_cloud_batch(N_eval, N_train, rd_train, noise_ratio, dim):
    """Held-out clouds at cardinality N_eval with the TRAINING density.

    Generated on the unit torus, then scaled by box = n_eval / n_train: that keeps
    the point spacing at the training value (1 / n_train) and grows only the box.
    """
    n_train, n_eval = round(N_train ** 0.5), round(N_eval ** 0.5)
    box = n_eval / n_train
    rd_unit = rd_train / box            # rd/spacing ratio preserved on the unit torus
    ds = PoissonDiskDataset(dim=dim, cardinality=N_eval, rd=rd_unit, seed=EVAL_SEED,
                            noise_scale_min=0.0, noise_scale_max=noise_ratio * rd_unit,
                            periodic=True)
    noisy = ds.noise_sample(ds.generate_sample(EVAL_CLOUDS))
    return (noisy * box).astype(np.float32), box


def stats(x, rd, box, h_factor):
    """Periodic quality metrics for a batch of clouds. x: (B, N, D) torch."""
    N = x.shape[1]
    _, pw = _pbc_rel(x, box)
    eye = torch.eye(N, dtype=torch.bool, device=x.device)
    pd = pw[:, ~eye]
    nn = pw.masked_fill(eye.unsqueeze(0), float('inf')).min(dim=-1).values
    return dict(
        viol=torch.relu(rd - pd).mean().item(),
        ill_pct=(pd < rd).float().mean().item() * 100,
        ill_per_cloud=(pd < rd).float().sum(dim=-1).mean().item() / 2,
        nn=nn.mean().item(),
        # spread of nn distance relative to its mean: how UNIFORM the spacing is, i.e.
        # how close the output is to a clean lattice. Distinct from knn_keep, which asks
        # whether the INPUT arrangement survived — an SPH restart wants a well-conditioned
        # distribution, not a faithful copy of the broken input.
        nn_cv=(nn.std() / (nn.mean() + 1e-12)).item(),
        kg=kg_norm(x, h_factor, box).mean().item() if x.shape[-1] == 2 else float('nan'),
    )


def knn_preservation(a, b, box, k=6):
    """Mean Jaccard overlap of each particle's k-NN set before vs after correction.

    1.0 = every particle keeps exactly the same neighbours, i.e. the corrector
    relaxed positions without rewiring who-neighbours-whom. For an SPH restart that
    is the desirable behaviour: the physical state is tied to the arrangement, so a
    corrector that reshuffles neighbourhoods perturbs the simulation more than one
    that nudges within them.
    """
    def sets(x):
        _, pw = _pbc_rel(x, box)
        N = x.shape[1]
        eye = torch.eye(N, dtype=torch.bool, device=x.device)
        idx = pw.masked_fill(eye, float('inf')).topk(k, dim=-1, largest=False).indices
        m = torch.zeros(x.shape[0], N, N, dtype=torch.bool, device=x.device)
        return m.scatter_(2, idx, True)

    ma, mb = sets(a), sets(b)
    return ((ma & mb).sum(-1).float() / (ma | mb).sum(-1).float()).mean().item()


def uniform_fraction(disp):
    """Share of the motion that is bulk translation rather than local restructuring.

    Every loss term depends only on relative positions, so a uniform translation is
    invisible to them — a degenerate solution that leaves |KG| and illegal% looking
    untouched while the cloud drifts. 0 = purely local work, 1 = pure drift.
    """
    mean_vec = disp.mean(dim=1, keepdim=True)
    return (mean_vec.norm(dim=-1).mean() / (disp.norm(dim=-1).mean() + 1e-12)).item()


def apply_k(model, x, k, rd_model, box):
    """k correction passes, chunked so dense edge tensors stay in memory."""
    kwargs = {'box': box} if getattr(model, 'uses_box', False) else {}
    out = []
    with torch.no_grad():
        for start in range(0, x.shape[0], MODEL_CHUNK):
            xc = x[start:start + MODEL_CHUNK]
            for _ in range(k):
                d = (model(xc, rd=rd_model, **kwargs)
                     if getattr(model, 'uses_rd', False) else model(xc))
                xc = xc + d
            out.append(xc)
    return torch.cat(out, dim=0)


def score_synthetic(model, cfgs, arm, deploy_ns, device, n_params, checkpoint):
    ds_cfg, m_cfg, l_cfg = cfgs['dataset'], cfgs['model'], cfgs['loss']
    N_train  = ds_cfg['points_per_cloud']
    rd_train = float(ds_cfg['rd'])
    dim      = ds_cfg['dim']
    noise_ratio = float(ds_cfg['noise_scale_max']) / rd_train
    h_factor = float(l_cfg['params'].get('h_factor', 2.0))
    cutoff   = float(m_cfg.get('cutoff_rd', rd_train))
    rd_model = torch.tensor(cutoff, dtype=torch.float32, device=device)

    rows = []
    for N_eval in deploy_ns:
        pts, box = eval_cloud_batch(N_eval, N_train, rd_train, noise_ratio, dim)
        x = torch.tensor(pts, device=device)

        pre = stats(x, rd_train, box, h_factor)
        k1  = apply_k(model, x, 1, rd_model, box)
        k5  = apply_k(model, x, 5, rd_model, box)
        s1, s5 = (stats(t, rd_train, box, h_factor) for t in (k1, k5))

        def red(s):
            return (pre['ill_per_cloud'] - s['ill_per_cloud']) / (pre['ill_per_cloud'] + 1e-9) * 100

        rows.append({
            'arm': arm, 'regime': 'synthetic', 'train_N': N_train, 'deploy_N': N_eval,
            'noise_ratio': round(noise_ratio, 4), 'n_params': n_params,
            'viol_pre': pre['viol'], 'viol_k1': s1['viol'], 'viol_k5': s5['viol'],
            'ill_pct_pre': pre['ill_pct'], 'ill_pct_k1': s1['ill_pct'], 'ill_pct_k5': s5['ill_pct'],
            'nn_pre': pre['nn'], 'nn_k1': s1['nn'], 'nn_k5': s5['nn'],
            'kg_pre': pre['kg'], 'kg_k1': s1['kg'], 'kg_k5': s5['kg'],
            'disp_k1': (k1 - x).norm(dim=-1).mean().item(),
            'disp_k5': (k5 - x).norm(dim=-1).mean().item(),
            'viol_red_k1': red(s1), 'viol_red_k5': red(s5),
            'knn_keep_k5': knn_preservation(x, k5, box),
            'uniform_frac_k1': uniform_fraction(k1 - x),
            'nn_cv_pre': pre['nn_cv'], 'nn_cv_k5': s5['nn_cv'],
            'checkpoint': str(checkpoint),
        })
        print(f'  synthetic N={N_eval:4d} (box {box:.3f})  '
              f'viol_red {red(s5):6.1f}%  ill {pre["ill_pct"]:.1f}->{s5["ill_pct"]:.1f}%  '
              f'|KG| {pre["kg"]:.4f}->{s5["kg"]:.4f}  '
              f'nn_CV {100 * pre["nn_cv"]:.1f}->{100 * s5["nn_cv"]:.1f}%  '
              f'knn_keep {rows[-1]["knn_keep_k5"]:.3f}  '
              f'drift {100 * rows[-1]["uniform_frac_k1"]:.0f}%')
    return rows


def score_trajectory(cfgs, arm, model_config_path, checkpoint, device,
                     n_params, stride, k, rd_test=0.02, box=1.0):
    from inference.correctors.wholecloud.wholecloud_corrector import (
        WholeCloudCorrector2D, WholeCloudCorrector2DConfig)

    ds_cfg = cfgs['dataset']
    traj_path = ROOT / 'artifacts/inference/experiments/sph_tv/data/positions_without.npy'
    traj = np.load(traj_path, mmap_mode='r')

    corrector = WholeCloudCorrector2D(WholeCloudCorrector2DConfig(
        checkpoint=str(checkpoint), model_config=str(model_config_path),
        rd_train=float(ds_cfg['rd']), rd_test=rd_test, box=box, device=device))

    h_factor = float(cfgs['loss']['params'].get('h_factor', 2.0))
    ts = [t for t in range(0, traj.shape[0], stride) if t >= 300]   # disordered regime
    pre_acc, post_acc = [], []
    for t in ts:
        pts = np.asarray(traj[t], dtype=np.float32)
        cor = corrector.apply(pts, k=k)
        for acc, p in ((pre_acc, pts), (post_acc, cor)):
            nn = nn_dists(p, box=box)
            xt = torch.tensor(p[None], dtype=torch.float32)
            acc.append((float(kg_norm(xt, h_factor, box)[0]), float(nn.mean()),
                        float((nn < rd_test).mean()) * 100))
    pre  = np.array(pre_acc).mean(axis=0)
    post = np.array(post_acc).mean(axis=0)
    print(f'  trajectory ({len(ts)} steps, t>=300)  '
          f'|KG| {pre[0]:.4f}->{post[0]:.4f}  nn {pre[1]:.5f}->{post[1]:.5f}  '
          f'ill {pre[2]:.1f}->{post[2]:.1f}%')

    return [{
        'arm': arm, 'regime': 'trajectory', 'train_N': ds_cfg['points_per_cloud'],
        'deploy_N': int(traj.shape[1]),
        'noise_ratio': round(float(ds_cfg['noise_scale_max']) / float(ds_cfg['rd']), 4),
        'n_params': n_params,
        'viol_pre': '', 'viol_k1': '', 'viol_k5': '',
        'ill_pct_pre': pre[2], 'ill_pct_k1': '', 'ill_pct_k5': post[2],
        'nn_pre': pre[1], 'nn_k1': '', 'nn_k5': post[1],
        'kg_pre': pre[0], 'kg_k1': '', 'kg_k5': post[0],
        'disp_k1': '', 'disp_k5': '', 'viol_red_k1': '', 'viol_red_k5': '',
        'checkpoint': str(checkpoint),
    }]


def append_rows(out_csv: Path, rows):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not out_csv.exists()
    with open(out_csv, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True, help='artifacts/training/train_run_*')
    ap.add_argument('--arm', required=True, help='name this row set carries in results.csv')
    ap.add_argument('--checkpoint', default='model_best.pt',
                    help='file inside the run dir, or a path relative to the repo root '
                         '(lets a shipped checkpoint be scored against a run\'s configs)')
    ap.add_argument('--deploy-n', default='', help='comma-separated; default = train N only')
    ap.add_argument('--trajectory', action='store_true', help='also score the real SPH trajectory')
    ap.add_argument('--stride', type=int, default=50)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default=str(ROOT / 'paper/results.csv'))
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    cfgs = load_run_configs(run_dir)
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = next((c for c in (run_dir / args.checkpoint, ROOT / args.checkpoint)
                     if c.exists()), run_dir / args.checkpoint)
    device = torch.device(args.device)

    model = build_model(cfgs['model'], cfgs['dataset']['dim'], ckpt, device)
    n_params = sum(p.numel() for p in model.parameters())

    deploy_ns = ([int(v) for v in args.deploy_n.split(',')] if args.deploy_n
                 else [cfgs['dataset']['points_per_cloud']])

    print(f'arm={args.arm}  train_N={cfgs["dataset"]["points_per_cloud"]}  '
          f'params={n_params:,}  ckpt={ckpt.name}')

    rows = score_synthetic(model, cfgs, args.arm, deploy_ns, device, n_params, ckpt)

    if args.trajectory:
        model_cfg_path = next(p for p in (run_dir / 'configs').glob('*.yaml')
                              if 'model_file' in yaml.safe_load(open(p)))
        rows += score_trajectory(cfgs, args.arm, model_cfg_path, ckpt, args.device,
                                 n_params, args.stride, args.k)

    append_rows(Path(args.out), rows)
    print(f'-> {args.out}  (+{len(rows)} rows)')


if __name__ == '__main__':
    main()
