"""
sph_viz.py
----------
Comparison figure: non-TV + corrector model  vs  TV variant.

Layout  (2 rows x 3 cols):
  row 0  |  scatter (nonTV + K=1)  |  scatter (TV)  |  hist overlay (K=1 vs TV)
  row 1  |  scatter (nonTV + K=3)  |  scatter (TV)  |  hist overlay (K=3 vs TV)

Scatter: points coloured by min-nn distance, shared colormap range across all panels.
Hist:    overlaid semi-transparent bars (blue = model, orange = TV).
         Vertical dashed lines mark the mean nn-distance of each distribution.
         No rd reference anywhere.

Importable:
    from inference.sph_viz import plot_comparison, apply_corrector, min_nn_pbc

CLI:
    .venv\\Scripts\\python.exe inference/sph_viz.py --timestep 0
    .venv\\Scripts\\python.exe inference/sph_viz.py --timestep 300 --save inference/t300.png
"""
import argparse, sys, importlib
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.data_processor import DataProcessor
from utils.config import load_model_config

# ── defaults ───────────────────────────────────────────────────────────────────
# 6x6 experiment — N=100 model (n100_p050, rd_train=0.076, scale=3.8)
CKPT_6x6     = Path("training_artifacts/train_run_2026-05-28_14-53-45/model_final.pt")
CFG_6x6      = Path("configs/model_configs/model_config_9_n100_p050.yaml")
RD_TRAIN_6x6 = 0.076

# 10x10 experiment — N=50 model (sparse, rd_train=0.05, scale=2.5)
CKPT_10x10   = Path("training_artifacts/train_run_2026-05-26_17-34-01/model_final.pt")
CFG_10x10    = Path("configs/model_configs/model_config_9.yaml")
RD_TRAIN_10x10 = 0.05

CKPT_DEFAULT = CKPT_6x6
CFG_DEFAULT  = CFG_6x6
DATA_WITHOUT = Path("inference/sph_data/positions_without.npy")
DATA_WITH    = Path("inference/sph_data/positions.npy")
RD_TEST      = 0.02
DOMAIN       = 1.0

C_MODEL = '#2471a3'   # blue  — nonTV + model
C_TV    = '#e67e22'   # orange — TV variant


# ── PBC nearest-neighbour distance ────────────────────────────────────────────
def min_nn_pbc(pts, domain=DOMAIN):
    diff = pts[:, None] - pts[None, :]
    diff = diff - domain * np.round(diff / domain)
    D    = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(D, np.inf)
    return D.min(axis=1)


# ── tiled PBC corrector ────────────────────────────────────────────────────────
def _build_tile(points, tile_lo, tile_hi, ghost_w, domain=DOMAIN):
    ext_lo = tile_lo - ghost_w
    ext_hi = tile_hi + ghost_w
    pts_list, idx_list, core_list = [], [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            shifted = points + np.array([dx * domain, dy * domain])
            in_ext  = np.all((shifted >= ext_lo) & (shifted < ext_hi), axis=1)
            if not in_ext.any():
                continue
            pts_list.append(shifted[in_ext])
            idx_list.append(np.where(in_ext)[0])
            if dx == 0 and dy == 0:
                in_core = np.all((points >= tile_lo) & (points < tile_hi), axis=1)
                core_list.append(in_core[in_ext])
            else:
                core_list.append(np.zeros(in_ext.sum(), dtype=bool))
    return np.vstack(pts_list), np.concatenate(core_list), np.concatenate(idx_list)


def apply_corrector(points, model, rd_t, scale, processor, device,
                    grid=6, rd_test=RD_TEST, domain=DOMAIN, k=1):
    """Apply k passes of the tiled PBC corrector with coordinate rescaling.

    grid=6, ghost_width=rd gives ~107 points per tile (69 core + 37 ghost),
    matching the training distribution N~100.  ghost_width=rd is the minimum
    value that guarantees all violations are visible (correctness proof: any
    pair with d_PBC < rd is captured if ghost_width >= rd).
    """
    cell    = domain / grid
    ghost_w = rd_test          # minimum provably correct; 2*rd was unnecessarily large
    pts     = points.copy()
    for _ in range(k):
        displacements = np.zeros_like(pts)
        counts        = np.zeros(len(pts), dtype=int)
        for i in range(grid):
            for j in range(grid):
                lo = np.array([i * cell, j * cell])
                hi = np.array([(i+1)*cell, (j+1)*cell])
                ext, is_core, orig_idx = _build_tile(pts, lo, hi, ghost_w, domain)
                if is_core.sum() == 0:
                    continue
                pts_s         = ext * scale
                x_inv, _, rev = processor.make_invariant(pts_s[None])
                x_t           = torch.tensor(x_inv, dtype=torch.float32, device=device)
                with torch.no_grad():
                    disp_inv = model(x_t, rd=rd_t).cpu().numpy()[0]
                corrected = rev(x_inv + disp_inv)[0]
                disp_orig = (corrected - pts_s) / scale
                for idx_k in range(len(ext)):
                    if is_core[idx_k]:
                        displacements[orig_idx[idx_k]] += disp_orig[idx_k]
                        counts[orig_idx[idx_k]] += 1
        counts = np.maximum(counts, 1)
        pts = (pts + displacements / counts[:, None]) % domain
    return pts


# ── RDF ───────────────────────────────────────────────────────────────────────
def compute_rdf(pts, r_max, n_bins=80, domain=DOMAIN):
    """2D PBC radial distribution function g(r)."""
    N   = len(pts)
    rho = N / domain**2
    diff = pts[:, None] - pts[None, :]
    diff = diff - domain * np.round(diff / domain)
    D    = np.linalg.norm(diff, axis=-1)
    d_pairs = D[np.triu_indices(N, k=1)]
    d_pairs = d_pairs[d_pairs <= r_max]
    bins        = np.linspace(0, r_max, n_bins + 1)
    counts, _   = np.histogram(d_pairs, bins=bins)
    r_centers   = (bins[:-1] + bins[1:]) / 2
    dr          = bins[1] - bins[0]
    g_r = (2 * counts / N) / (2 * np.pi * r_centers * dr * rho)
    return r_centers, g_r


# ── panel drawing ──────────────────────────────────────────────────────────────
def _draw_scatter(ax, pts, nn, vmin, vmax, title):
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=nn,
                    cmap='plasma_r', vmin=vmin, vmax=vmax,
                    s=3, alpha=0.65, linewidths=0, rasterized=True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # no aspect='equal' — let gridspec control heights so rows stay aligned
    ax.set_xlabel('x', fontsize=8); ax.set_ylabel('y', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=8.5, pad=3)
    return sc


def _cv(nn):
    return float(nn.std() / nn.mean())


def _draw_hist_overlay(ax, nn_orig, nn_model, nn_tv, title, shared_xlim=None):
    all_nn = np.concatenate([nn_orig, nn_model, nn_tv])
    lo = max(0.0, all_nn.min() - (all_nn.max() - all_nn.min()) * 0.05)
    hi = all_nn.max() + (all_nn.max() - all_nn.min()) * 0.05
    if shared_xlim is not None:
        lo, hi = shared_xlim
    bins = np.linspace(lo, hi, 55)

    ax.hist(nn_orig,  bins=bins, color='#888', alpha=0.45, histtype='stepfilled',
            label=f'nonTV orig   mean={nn_orig.mean():.4f}  CV={_cv(nn_orig):.3f}')
    ax.hist(nn_model, bins=bins, color=C_MODEL, alpha=0.55, histtype='stepfilled',
            label=f'nonTV+model  mean={nn_model.mean():.4f}  CV={_cv(nn_model):.3f}')
    ax.hist(nn_tv,    bins=bins, color=C_TV,    alpha=0.45, histtype='stepfilled',
            label=f'TV           mean={nn_tv.mean():.4f}  CV={_cv(nn_tv):.3f}')

    for nn, c in [(nn_orig, '#666'), (nn_model, C_MODEL), (nn_tv, C_TV)]:
        ax.axvline(nn.mean(), color=c, lw=1.3, ls='--')

    ax.set_xlabel('min nearest-neighbour distance', fontsize=8)
    ax.set_ylabel('particle count', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(lo, hi)
    ax.set_title(title, fontsize=8.5, pad=3)
    ax.legend(fontsize=7, loc='upper left', framealpha=0.85)


def _draw_rdf_overlay(ax, r_orig, g_orig, r_m, g_m, r_tv, g_tv, title, rd=RD_TEST):
    ax.plot(r_orig, g_orig, color='#888', lw=1.3, alpha=0.75, ls=':', label='nonTV orig')
    ax.plot(r_m,    g_m,    color=C_MODEL, lw=1.6, alpha=0.85, label='nonTV + model')
    ax.plot(r_tv,   g_tv,   color=C_TV,    lw=1.6, alpha=0.85, label='TV')
    ax.axhline(1.0, color='#aaa', lw=0.8, ls=':', zorder=0)
    ax.set_xlabel('r', fontsize=8)
    ax.set_ylabel('g(r)', fontsize=8)
    ax.set_xlim(0, r_m[-1])
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=8.5, pad=3)
    ax.legend(fontsize=7.5, framealpha=0.85)


# ── main public function ───────────────────────────────────────────────────────
def plot_comparison(
    pts_nonTV,
    corr_k1,
    corr_k2,
    corr_k3,
    corr_k5,
    pts_tv,
    domain=DOMAIN,
    timestep=None,
    scale=None,
    grid=6,
    save_path=None,
    show=True,
):
    """4-row x 5-col: nonTV orig | model Kx | TV | nn-hist | RDF  (rows: K=1,2,3,5)"""
    if scale is None:
        scale = RD_TRAIN_6x6 / RD_TEST
    step_str = f't = {timestep}' if timestep is not None else ''

    print('  Computing nn distances...')
    nn_raw = min_nn_pbc(pts_nonTV, domain)
    nn_k1  = min_nn_pbc(corr_k1,  domain)
    nn_k2  = min_nn_pbc(corr_k2,  domain)
    nn_k3  = min_nn_pbc(corr_k3,  domain)
    nn_k5  = min_nn_pbc(corr_k5,  domain)
    nn_tv  = min_nn_pbc(pts_tv,   domain)

    all_nn = np.concatenate([nn_raw, nn_k1, nn_k2, nn_k3, nn_k5, nn_tv])
    vmin, vmax = np.percentile(all_nn, 1), np.percentile(all_nn, 99)
    xlim = (max(0.0, all_nn.min() * 0.97), all_nn.max() * 1.03)

    r_max = RD_TEST * 4.0
    print('  Computing RDFs...')
    r_orig, g_orig = compute_rdf(pts_nonTV, r_max, domain=domain)
    r_tv,   g_tv   = compute_rdf(pts_tv,   r_max, domain=domain)
    r_k1,   g_k1   = compute_rdf(corr_k1,  r_max, domain=domain)
    r_k2,   g_k2   = compute_rdf(corr_k2,  r_max, domain=domain)
    r_k3,   g_k3   = compute_rdf(corr_k3,  r_max, domain=domain)
    r_k5,   g_k5   = compute_rdf(corr_k5,  r_max, domain=domain)

    fig, axes = plt.subplots(
        4, 5, figsize=(27, 16),
        gridspec_kw={
            'width_ratios': [1, 1, 1, 1.4, 1.4],
            'hspace': 0.44,
            'wspace': 0.30,
        }
    )
    fig.subplots_adjust(left=0.04, right=0.99, top=0.92, bottom=0.10)

    rows = [
        (corr_k1, nn_k1, r_k1, g_k1, 'K = 1'),
        (corr_k2, nn_k2, r_k2, g_k2, 'K = 2'),
        (corr_k3, nn_k3, r_k3, g_k3, 'K = 3'),
        (corr_k5, nn_k5, r_k5, g_k5, 'K = 5'),
    ]

    sc_last = None
    for row_idx, (corr, nn_corr, r_m, g_m, k_label) in enumerate(rows):
        ax_raw, ax_model, ax_tv, ax_hist, ax_rdf = axes[row_idx]

        _draw_scatter(ax_raw,   pts_nonTV, nn_raw,  vmin, vmax,
                      f'nonTV (original)   {step_str}')
        sc_last = _draw_scatter(ax_model, corr, nn_corr, vmin, vmax,
                      f'nonTV + model  {k_label}   {step_str}')
        _draw_scatter(ax_tv,    pts_tv,    nn_tv,   vmin, vmax,
                      f'TV   {step_str}')
        _draw_hist_overlay(ax_hist, nn_raw, nn_corr, nn_tv,
                      f'nn-dist: {k_label} vs TV   {step_str}',
                      shared_xlim=xlim)
        _draw_rdf_overlay(ax_rdf, r_orig, g_orig, r_m, g_m, r_tv, g_tv,
                      f'RDF: {k_label} vs TV   {step_str}')

    cbar_ax = fig.add_axes([0.04, 0.03, 0.56, 0.018])
    cbar = fig.colorbar(sc_last, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('min nearest-neighbour distance', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f'SPH corrector comparison   {step_str}'
        f'   |   N = 2,500   |   {grid}x{grid} grid'
        f'   |   scale s = {scale:.1f}',
        fontsize=10, y=0.975,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved -> {save_path}')
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestep',   type=int, default=None)
    parser.add_argument('--stride',     type=int, default=100)
    parser.add_argument('--outdir',     default=None,
                        help='Output dir (default: inference/frames_{grid}x{grid})')
    parser.add_argument('--save',       default=None)
    parser.add_argument('--checkpoint', default=None,
                        help='Override checkpoint (default: auto from --grid)')
    parser.add_argument('--config',     default=None,
                        help='Override model config (default: auto from --grid)')
    parser.add_argument('--grid',       type=int, default=6,
                        help='Tiling grid size (default: 6 for 6x6)')
    parser.add_argument('--rd-train',   type=float, default=None,
                        help='rd the model was trained at (default: auto from --grid)')
    parser.add_argument('--without',    default=str(DATA_WITHOUT))
    parser.add_argument('--with_tv',    default=str(DATA_WITH))
    args = parser.parse_args()

    # resolve grid-dependent defaults
    if args.grid == 10:
        ckpt     = Path(args.checkpoint or CKPT_10x10)
        cfg      = args.config or str(CFG_10x10)
        rd_train = args.rd_train or RD_TRAIN_10x10
    else:
        ckpt     = Path(args.checkpoint or CKPT_6x6)
        cfg      = args.config or str(CFG_6x6)
        rd_train = args.rd_train or RD_TRAIN_6x6

    scale  = rd_train / RD_TEST
    outdir = Path(args.outdir or f'inference/frames_{args.grid}x{args.grid}')
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cpu')
    model_cfg = load_model_config(cfg)
    m         = importlib.import_module('models.fixed_rd.model9')
    model     = m.CorrectorModel(model_cfg, input_dim=2, initialization='xavier_uniform')
    model.load_state_dict(torch.load(str(ckpt), map_location='cpu'))
    model.eval()
    rd_t      = torch.tensor(rd_train, dtype=torch.float32)
    processor = DataProcessor()

    print(f'Grid={args.grid}x{args.grid}  rd_train={rd_train}  scale={scale:.2f}  ckpt={ckpt.name}')

    pos_without = np.load(args.without)
    pos_with    = np.load(args.with_tv)
    T, N, _ = pos_without.shape
    print(f'Loaded  T={T}  N={N}')

    timesteps = [args.timestep % T] if args.timestep is not None else list(range(0, T, args.stride))

    for i, t in enumerate(timesteps):
        print(f'\n[{i+1}/{len(timesteps)}]  t = {t}')
        pts_wo = pos_without[t].astype(np.float32)
        pts_wi = pos_with[t].astype(np.float32)

        corr_k1 = apply_corrector(pts_wo, model, rd_t, scale, processor, device,
                                   grid=args.grid, k=1)
        corr_k2 = apply_corrector(pts_wo, model, rd_t, scale, processor, device,
                                   grid=args.grid, k=2)
        corr_k3 = apply_corrector(pts_wo, model, rd_t, scale, processor, device,
                                   grid=args.grid, k=3)
        corr_k5 = apply_corrector(pts_wo, model, rd_t, scale, processor, device,
                                   grid=args.grid, k=5)

        for k, arr in [(1,corr_k1),(2,corr_k2),(3,corr_k3),(5,corr_k5)]:
            print(f'  K={k}: mean nn = {min_nn_pbc(arr).mean():.5f}')
        print(f'  TV:  mean nn = {min_nn_pbc(pts_wi).mean():.5f}')

        save = args.save if (args.timestep is not None and args.save) \
               else outdir / f'sph_viz_t{t:04d}.png'

        plot_comparison(pts_wo, corr_k1, corr_k2, corr_k3, corr_k5, pts_wi,
                        timestep=t, scale=scale, grid=args.grid,
                        save_path=save, show=False)


if __name__ == '__main__':
    main()
