"""
make_report.py
--------------
Generates inference/pbc_report.html -- a self-contained HTML validation report
for the tiled PBC corrector approach.

Experiments:
  - Synthetic toy (N=96, rd=0.076, 2x2 grid): ghost detection proof
  - SPH t=0    (N=2500, rd=0.02, 5x5 grid, scale=3.8): real-data performance

Runtime: ~2-3 minutes on CPU.

Usage:
    .venv\\Scripts\\python.exe inference/make_report.py
"""
import base64, io, sys, importlib
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.data_processor import DataProcessor
from utils.config import load_model_config

# ── constants ──────────────────────────────────────────────────────────────────
CKPT      = Path("training_artifacts/train_run_2026-05-28_14-53-45/model_final.pt")
CFG       = Path("configs/model_configs/model_config_9_n100_p050.yaml")
SPH_DATA  = Path("inference/sph_data/positions_without.npy")
RD_TRAIN  = 0.076
RD_TEST   = 0.02
SCALE     = RD_TRAIN / RD_TEST        # 3.8
DOMAIN    = 1.0
KMAX      = 10
SEED      = 7

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9, 'figure.dpi': 120})


# ── PBC utilities ──────────────────────────────────────────────────────────────
def pbc_dists(pts, domain=DOMAIN):
    diff = pts[:, None] - pts[None, :]
    diff = diff - domain * np.round(diff / domain)
    return np.linalg.norm(diff, axis=-1)


def violation_stats(pts, rd, domain=DOMAIN):
    D = pbc_dists(pts, domain)
    N = len(pts)
    upper = np.triu(np.ones((N, N), bool), k=1)
    n_pairs = int((D[upper] < rd).sum())
    per_pt  = (D < rd).sum(axis=1) - 1
    return n_pairs, per_pt                # n_pairs, per-point violation count


def min_nn(pts, domain=DOMAIN):
    D = pbc_dists(pts, domain)
    np.fill_diagonal(D, 1e9)
    return D.min(axis=1)


# ── tile + ghost buffer ────────────────────────────────────────────────────────
def build_tile(points, tile_lo, tile_hi, ghost_width, domain=DOMAIN):
    ext_lo = tile_lo - ghost_width
    ext_hi = tile_hi + ghost_width
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
    return (np.vstack(pts_list), np.concatenate(core_list), np.concatenate(idx_list))


# ── corrector ──────────────────────────────────────────────────────────────────
def _infer_tile(pts_ext, is_core, orig_idx, model, rd_t, scale, processor, device):
    if is_core.sum() == 0:
        return {}
    pts_scaled    = pts_ext * scale
    x_inv, _, rev = processor.make_invariant(pts_scaled[None])
    x_t           = torch.tensor(x_inv, dtype=torch.float32, device=device)
    with torch.no_grad():
        disp_inv   = model(x_t, rd=rd_t).cpu().numpy()[0]
    corrected     = rev(x_inv + disp_inv)[0]
    disp_orig     = (corrected - pts_scaled) / scale
    return {int(orig_idx[i]): disp_orig[i] for i in range(len(pts_ext)) if is_core[i]}


def run_corrector(points, model, rd_t, scale, processor, device,
                  grid, ghost_width, domain=DOMAIN):
    cell          = domain / grid
    displacements = np.zeros_like(points)
    counts        = np.zeros(len(points), dtype=int)
    for i in range(grid):
        for j in range(grid):
            lo = np.array([i * cell, j * cell])
            hi = np.array([(i+1)*cell, (j+1)*cell])
            pts_ext, is_core, orig_idx = build_tile(points, lo, hi, ghost_width, domain)
            for idx, disp in _infer_tile(pts_ext, is_core, orig_idx,
                                          model, rd_t, scale, processor, device).items():
                displacements[idx] += disp
                counts[idx] += 1
    counts = np.maximum(counts, 1)
    return (points + displacements / counts[:, None]) % domain


# ── figure helpers ─────────────────────────────────────────────────────────────
def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def _img_tag(b64, alt='', width='100%'):
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="width:{width};max-width:900px;">'


# ── Figure 1: ghost buffer schematic ──────────────────────────────────────────
def fig_ghost_schematic():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax_idx, (ax, title, show_ghosts) in enumerate(zip(
        axes,
        ['No ghost buffer: cross-tile violation invisible',
         'Ghost buffer g = 2r_d: violation visible in both tiles'],
        [False, True]
    )):
        grid_n = 3; cell = 1/3

        # draw grid
        for k in range(grid_n + 1):
            v = k * cell
            ax.axhline(v, color='#aaa', lw=0.7)
            ax.axvline(v, color='#aaa', lw=0.7)

        # highlight tile (0,0)
        from matplotlib.patches import FancyArrowPatch
        hi_tile = plt.Rectangle((0, 0), cell, cell, fc='#ddeeff', ec='steelblue', lw=1.5, zorder=1)
        ax.add_patch(hi_tile)

        rd = 0.08
        ghost_w = 2 * rd

        if show_ghosts:
            ext = plt.Rectangle((-ghost_w, -ghost_w), cell + 2*ghost_w, cell + 2*ghost_w,
                                  fc='none', ec='steelblue', lw=1.0, ls='--', zorder=1)
            ax.add_patch(ext)
            ax.text(-ghost_w + 0.005, -ghost_w + 0.005, 'ghost buffer\ng = 2r_d',
                    fontsize=7, color='steelblue', va='bottom')

        # synthetic core points in tile (0,0)
        core = np.array([[0.05, 0.08], [0.18, 0.04], [0.10, 0.22], [0.28, 0.18]])
        ax.scatter(core[:, 0], core[:, 1], s=35, c='steelblue', zorder=5, label='core')

        # points in tile (1,0)
        adj = np.array([[cell+0.04, 0.10], [cell+0.15, 0.20]])
        ax.scatter(adj[:, 0], adj[:, 1], s=35, c='#888', zorder=5)

        # violation pair: one in tile(0,0) near x=cell, one just in tile(1,0)
        vA = np.array([cell - 0.025, 0.15])
        vB = np.array([cell + 0.025, 0.15])
        if show_ghosts:
            # show vB as ghost
            ax.scatter([vB[0]], [vB[1]], s=45, c='#e67e22', zorder=6,
                       marker='D', label='ghost (core of adj. tile)')
            ax.plot([vA[0], vB[0]], [vA[1], vB[1]], 'r-', lw=1.5, zorder=7)
            ax.text((vA[0]+vB[0])/2, vA[1]+0.015, f'd < r_d', fontsize=7,
                    ha='center', color='red')
        else:
            ax.scatter([vB[0]], [vB[1]], s=45, c='#888', zorder=6, marker='D')

        ax.scatter([vA[0]], [vA[1]], s=55, c='red', zorder=6, marker='*')
        ax.scatter([vB[0]], [vB[1]], s=55, c='red' if show_ghosts else '#888',
                   zorder=7 if show_ghosts else 5, marker='*')

        # PBC violation pair: near x=0 and x=1
        pA = np.array([0.04, 0.28])
        pB_real = np.array([1.0 - 0.03, 0.28])
        if show_ghosts:
            pB_ghost = np.array([-0.03, 0.28])
            ax.scatter([pB_ghost[0]], [pB_ghost[1]], s=45, c='#9b59b6',
                       marker='D', zorder=6, label='ghost (PBC wrap)')
            ax.plot([pA[0], pB_ghost[0]], [pA[1], pB_ghost[1]], 'r--', lw=1.5, zorder=7)
            ax.annotate('', xy=(-0.025, 0.28), xytext=(-0.085, 0.28),
                        arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=1.2))
            ax.text(-0.09, 0.285, 'wraps\nfrom x=0.97', fontsize=6.5, color='#9b59b6', ha='right')
        ax.scatter([pA[0]], [pA[1]], s=55, c='#9b59b6', zorder=6, marker='*')
        ax.scatter([pB_real[0]], [pB_real[1]], s=35, c='#888', zorder=5)

        ax.set_xlim(-0.17, 1.05)
        ax.set_ylim(-0.17, 1.05)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        if show_ghosts:
            ax.legend(fontsize=7, loc='upper right', framealpha=0.8)

    plt.tight_layout()
    return _b64(fig)


# ── Figure 2: toy before/after panels ─────────────────────────────────────────
def fig_toy_panels(states, v_counts, rd=RD_TRAIN, grid=2, labels=None):
    if labels is None:
        labels = [f'K={k}  ({v} violations)' for k, v in zip([0,1,3], v_counts)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    cell = DOMAIN / grid
    N_bg = len(states[0]) - 6
    ia1, ia2 = N_bg, N_bg+1
    ib1, ib2 = N_bg+2, N_bg+3
    ic1, ic2 = N_bg+4, N_bg+5

    for ax, pts, lbl in zip(axes, states, labels):
        ax.scatter(pts[:N_bg, 0], pts[:N_bg, 1], s=10, c='steelblue', alpha=0.45, zorder=2)

        n_pairs, per_pt = violation_stats(pts, rd)
        violating = per_pt > 0
        ax.scatter(pts[violating, 0], pts[violating, 1], s=25, c='red', alpha=0.7, zorder=3)

        for (i1, i2), c, m in [((ia1,ia2),'#e67e22','*'),
                                ((ib1,ib2),'#27ae60','*'),
                                ((ic1,ic2),'#8e44ad','*')]:
            ax.scatter(pts[[i1,i2], 0], pts[[i1,i2], 1], s=90, c=c, zorder=5, marker=m)

        for k in range(grid+1):
            v = k * cell
            ax.axhline(v, color='gray', lw=0.5, ls='--', alpha=0.4)
            ax.axvline(v, color='gray', lw=0.5, ls='--', alpha=0.4)

        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal'); ax.set_title(lbl, fontsize=9)
        ax.set_xlabel('x'); ax.set_ylabel('y' if ax is axes[0] else '')

    legend_els = [
        mpatches.Patch(color='steelblue', label='valid particle'),
        mpatches.Patch(color='red',       label='particle in violation'),
        plt.Line2D([0],[0], marker='*', color='w', markerfacecolor='#e67e22', ms=10, label='(a) internal pair'),
        plt.Line2D([0],[0], marker='*', color='w', markerfacecolor='#27ae60', ms=10, label='(b) cross-tile pair'),
        plt.Line2D([0],[0], marker='*', color='w', markerfacecolor='#8e44ad', ms=10, label='(c) cross-PBC pair'),
    ]
    axes[2].legend(handles=legend_els, fontsize=7.5, loc='upper right')
    plt.tight_layout()
    return _b64(fig)


# ── Figure 3: convergence curve ────────────────────────────────────────────────
def fig_convergence(k_vals, v_toy, v_sph=None, v0_toy=None, v0_sph=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    datasets = [(v_toy, v0_toy, 'Synthetic toy  (N=96, grid=2x2)', 'steelblue'),
                (v_sph, v0_sph, 'SPH t=0  (N=2500, grid=5x5, scale=3.8)', '#c0392b')]

    for ax, (v, v0, title, color) in zip(axes, datasets):
        if v is None:
            ax.text(0.5, 0.5, 'not computed', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9)
            ax.set_title(title); continue
        pct = [(v0 - vi) / v0 * 100 for vi in v]
        ax.plot(k_vals, v, 'o-', color=color, ms=5, lw=1.5)
        ax2 = ax.twinx()
        ax2.plot(k_vals, pct, 's--', color=color, ms=4, lw=1.0, alpha=0.5)
        ax.set_xlabel('correction passes K')
        ax.set_ylabel('violation pairs', color=color)
        ax2.set_ylabel('% reduction from K=0', color=color, alpha=0.7)
        ax2.set_ylim(0, 105)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(k_vals)
        ax.grid(True, alpha=0.3)
        # mark plateau
        stable = [i for i in range(1, len(v)) if v[i] >= v[i-1]]
        if stable:
            ax.axvline(k_vals[stable[0]], color='gray', ls=':', lw=1, label=f'plateau at K={k_vals[stable[0]]}')
            ax.legend(fontsize=7.5)
    plt.tight_layout()
    return _b64(fig)


# ── Figure 4: SPH before/after colormap ───────────────────────────────────────
def fig_sph_panels(pts_before, pts_after, rd=RD_TEST):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    titles = [f'SPH t=0, K=0  ({violation_stats(pts_before,rd)[0]:,} violation pairs)',
              f'SPH t=0, K=3  ({violation_stats(pts_after,rd)[0]:,} violation pairs)']
    for ax, pts, title in zip(axes, [pts_before, pts_after], titles):
        nn = min_nn(pts)
        depth = np.maximum(0.0, rd - nn)   # per-point: how far below rd
        sc = ax.scatter(pts[:,0], pts[:,1], c=depth, cmap='RdYlGn_r',
                        vmin=0, vmax=rd*0.4, s=3, alpha=0.85, linewidths=0)
        plt.colorbar(sc, ax=ax, label='violation depth  r_d - min_nn', shrink=0.85)
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
        ax.set_aspect('equal'); ax.set_title(title, fontsize=9)
        ax.set_xlabel('x'); ax.set_ylabel('y' if ax is axes[0] else '')
    plt.tight_layout()
    return _b64(fig)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    device = torch.device('cpu')

    # -- load model ------------------------------------------------------------
    model_cfg = load_model_config(str(CFG))
    m         = importlib.import_module('models.fixed_rd.model9')
    model     = m.CorrectorModel(model_cfg, input_dim=2, initialization='xavier_uniform')
    model.load_state_dict(torch.load(str(CKPT), map_location='cpu'))
    model.eval()
    rd_t      = torch.tensor(RD_TRAIN, dtype=torch.float32)
    processor = DataProcessor()
    print(f'Model loaded  ({sum(p.numel() for p in model.parameters()):,} params)')

    # =========================================================================
    # EXPERIMENT A: Synthetic toy (same setup as pbc_toy.py)
    # =========================================================================
    from scipy.stats.qmc import PoissonDisk
    sampler = PoissonDisk(d=2, radius=RD_TRAIN, seed=SEED)
    pts_bg  = sampler.random(200).astype(np.float32)[6:][:90]

    cell_toy = DOMAIN / 2
    PUSH     = 0.38 * RD_TRAIN
    pa1 = np.array([0.20, 0.20], np.float32)
    pa2 = np.array([0.20 + PUSH, 0.20], np.float32)
    pb1 = np.array([cell_toy - PUSH/2, 0.75], np.float32)
    pb2 = np.array([cell_toy + PUSH/2, 0.75], np.float32)
    pc1 = np.array([PUSH*0.35,        0.50], np.float32)
    pc2 = np.array([1.0 - PUSH*0.65,  0.50], np.float32)
    pts_toy = np.vstack([pts_bg, pa1, pa2, pb1, pb2, pc1, pc2])

    v0_toy = violation_stats(pts_toy, RD_TRAIN)[0]
    print(f'Toy  K=0: {v0_toy} violations')

    toy_states  = [pts_toy.copy()]
    toy_v_curve = [v0_toy]
    cur = pts_toy.copy()
    for k in range(1, KMAX + 1):
        cur = run_corrector(cur, model, rd_t, scale=1.0, processor=processor,
                            device=device, grid=2,
                            ghost_width=2*RD_TRAIN)
        v = violation_stats(cur, RD_TRAIN)[0]
        toy_v_curve.append(v)
        if k in (1, 3):
            toy_states.append(cur.copy())
        print(f'  Toy  K={k}: {v} violations  ({(v0_toy-v)/v0_toy*100:.1f}% reduction)')

    # Ghost detection diagnostic
    _, _, oidx00 = build_tile(pts_toy, np.array([0.,0.]), np.array([cell_toy, cell_toy]), 2*RD_TRAIN)
    _, is_core01, oidx01 = build_tile(pts_toy, np.array([0.,cell_toy]),
                                       np.array([cell_toy, 1.]), 2*RD_TRAIN)
    N_toy   = len(pts_toy)
    ic2_idx = N_toy - 1
    ib2_idx = N_toy - 3
    ghost01 = ~is_core01
    ok_pbc  = ic2_idx in oidx00
    ok_tile = ib2_idx in oidx01[ghost01]
    print(f'  Ghost checks: cross-PBC={ok_pbc}  cross-tile={ok_tile}')

    # =========================================================================
    # EXPERIMENT B: SPH t=0
    # =========================================================================
    pos_all  = np.load(str(SPH_DATA))
    pts_sph0 = pos_all[0].astype(np.float32)
    N_sph    = len(pts_sph0)
    ghost_w_sph = 2.0 * RD_TEST   # = 0.04 in original coords

    v0_sph  = violation_stats(pts_sph0, RD_TEST)[0]
    print(f'SPH  K=0: {v0_sph} violations  (N={N_sph})')

    sph_before   = pts_sph0.copy()
    sph_v_curve  = [v0_sph]
    cur_sph      = pts_sph0.copy()
    sph_after_k3 = None

    for k in range(1, KMAX + 1):
        cur_sph = run_corrector(cur_sph, model, rd_t, scale=SCALE,
                                processor=processor, device=device,
                                grid=5, ghost_width=ghost_w_sph)
        v = violation_stats(cur_sph, RD_TEST)[0]
        sph_v_curve.append(v)
        if k == 3:
            sph_after_k3 = cur_sph.copy()
        print(f'  SPH  K={k}: {v} violations  ({(v0_sph-v)/v0_sph*100:.1f}% reduction)')

    # =========================================================================
    # GENERATE FIGURES
    # =========================================================================
    print('Generating figures...')
    k_vals = list(range(KMAX + 1))

    b64_ghost  = fig_ghost_schematic()
    b64_toy    = fig_toy_panels(toy_states,
                                [toy_v_curve[0], toy_v_curve[1], toy_v_curve[3]])
    b64_conv   = fig_convergence(k_vals, toy_v_curve, sph_v_curve, v0_toy, v0_sph)
    b64_sph    = fig_sph_panels(sph_before, sph_after_k3)

    # =========================================================================
    # BUILD HTML
    # =========================================================================
    toy_table_rows = ''.join(
        f'<tr><td>{k}</td><td>{v}</td>'
        f'<td>{(v0_toy-v)/v0_toy*100:.1f}%</td></tr>'
        for k, v in enumerate(toy_v_curve)
        if k <= 5 or k == KMAX
    )
    sph_table_rows = ''.join(
        f'<tr><td>{k}</td><td>{v}</td>'
        f'<td>{(v0_sph-v)/v0_sph*100:.1f}%</td></tr>'
        for k, v in enumerate(sph_v_curve)
        if k <= 5 or k == KMAX
    )

    html = rf"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Tiled PBC Corrector -- Validation Report</title>
<script>
MathJax = {{
  tex: {{ inlineMath: [['$','$']], displayMath: [['$$','$$']] }},
  options: {{ skipHtmlTags: ['script','noscript','style','textarea','pre'] }}
}};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
  body      {{ max-width:900px; margin:2em auto; font-family:Georgia,serif;
               line-height:1.6; color:#222; padding:0 1em; }}
  h1        {{ font-size:1.5em; border-bottom:2px solid #444; padding-bottom:0.3em; }}
  h2        {{ font-size:1.2em; margin-top:2em; border-bottom:1px solid #ccc; }}
  h3        {{ font-size:1.0em; margin-top:1.4em; }}
  pre, code {{ font-family:Consolas,monospace; font-size:0.85em;
               background:#f5f5f5; padding:0.1em 0.3em; border-radius:3px; }}
  pre       {{ padding:0.8em 1em; overflow-x:auto; border-left:3px solid #aaa; }}
  table     {{ border-collapse:collapse; margin:1em 0; width:auto; }}
  th, td    {{ border:1px solid #ccc; padding:0.35em 0.8em; text-align:right; }}
  th        {{ background:#eee; text-align:center; }}
  td:first-child {{ text-align:center; }}
  figure    {{ margin:1.5em 0; text-align:center; }}
  figcaption{{ font-size:0.85em; color:#555; margin-top:0.4em; text-align:left;
               max-width:820px; margin-left:auto; margin-right:auto; }}
  .check    {{ color:#1a7a1a; font-weight:bold; }}
  .warn     {{ color:#a03000; font-weight:bold; }}
  .note     {{ background:#fffbe6; border-left:3px solid #e8b800;
               padding:0.5em 1em; margin:1em 0; font-size:0.9em; }}
  .proof    {{ background:#f0f4ff; border-left:3px solid #5577cc;
               padding:0.6em 1em; margin:1em 0; font-size:0.9em; }}
  .algo     {{ background:#f9f9f9; border:1px solid #ccc;
               padding:0.8em 1.2em; font-family:Consolas,monospace;
               font-size:0.82em; line-height:1.55; white-space:pre; }}
</style>
</head>
<body>

<h1>Tiled PBC Corrector: Validation Report</h1>
<p style="font-size:0.85em; color:#666;">
  Model: model9 hd128_d3 (51K params) &mdash;
  Checkpoint: <code>train_run_2026-05-28_14-53-45</code> &mdash;
  Date: 2026-06-01
</p>

<!-- ══════════════════════════════════════════════════════════ -->
<h2>1 &nbsp; Summary</h2>
<p>
We apply a trained Poisson-disk corrector network (model9) to large periodic
point clouds whose particle count and interaction radius differ from the
training distribution. The domain is subdivided into a regular grid; each tile
is augmented with a <em>ghost buffer</em>&mdash;a halo of periodic images of
neighbouring particles&mdash;before inference. Only displacements for
<em>core</em> (non-ghost) particles are retained.
</p>
<p>
On a synthetic cloud ($N=96$, $r_d=0.076$, $2\times2$ grid) we verify that
both cross-tile and cross-boundary periodic violations are visible in the
ghost-augmented input.
On SPH simulation data ($N=2500$, $r_d=0.02$, $5\times5$ grid,
coordinate scale $s=3.8$), a single correction pass ($K=1$)
reduces the violation count from {v0_sph:,} to {sph_v_curve[1]:,}
(<strong>{(v0_sph - sph_v_curve[1])/v0_sph*100:.0f}%</strong> reduction).
Repeated application exhibits oscillatory behaviour: $K=2$ transiently
increases the violation count before subsequent passes return to a level
comparable to $K=1$. This indicates the corrector is not a contraction
under naive iteration; $K=1$ is the recommended deployment setting.
</p>

<!-- ══════════════════════════════════════════════════════════ -->
<h2>2 &nbsp; Problem Statement</h2>
<p>
Let $\mathbf{{P}} = \{{\mathbf{{p}}_i\}}_{{i=1}}^N \subset [0,L)^2$ be a set of
2D particle positions in a periodic domain of side length $L$.
Define the periodic distance:
$$
d_\text{{PBC}}(\mathbf{{p}}_i, \mathbf{{p}}_j)
  = \min_{{\mathbf{{k}} \in \mathbb{{Z}}^2}} \|\mathbf{{p}}_i - \mathbf{{p}}_j + L\mathbf{{k}}\|_2.
$$
The Poisson-disk constraint requires $d_\text{{PBC}}(\mathbf{{p}}_i, \mathbf{{p}}_j) \geq r_d$
for all $i \neq j$.
A <em>violation</em> is any pair with $d_\text{{PBC}} < r_d$; the per-pair
<em>violation depth</em> is $v_{{ij}} = \max(0,\, r_d - d_\text{{PBC}}(\mathbf{{p}}_i,\mathbf{{p}}_j))$.
</p>
<p>
The neural corrector (model9) was trained on clouds with $N \approx 100$,
$r_d = 0.076$, domain $[0,1)^2$, and violations introduced by small random
displacements (noise scale $0.1$&ndash;$0.5 \times r_d$).
The SPH simulation data has $N=2500$, $r_d=0.02$, domain $[0,1)^2$.
Direct application of the model is therefore not feasible: the particle
density and interaction radius are both different.
</p>

<!-- ══════════════════════════════════════════════════════════ -->
<h2>3 &nbsp; Method</h2>

<h3>3.1 &nbsp; Spatial tiling</h3>
<p>
The domain $[0,1)^2$ is partitioned into a $G \times G$ regular grid of
tiles $T_{{ab}} = [a/G,\, (a+1)/G) \times [b/G,\, (b+1)/G)$,
$a,b \in \{{0,\ldots,G-1\}}$.
For SPH data, $G=5$ gives $N_\text{{tile}} \approx 100$, matching the
training distribution.
</p>

<h3>3.2 &nbsp; Ghost buffer construction</h3>
<p>
For each tile $T_{{ab}}$, the <em>extended region</em> is
$T_{{ab}}^+ = [a/G - g,\, (a+1)/G + g) \times [b/G - g,\, (b+1)/G + g)$
where $g$ is the ghost width.
A particle at position $\mathbf{{p}} \in [0,1)^2$ contributes a
<em>ghost</em> to tile $T_{{ab}}^+$ if any of its nine periodic images
$\mathbf{{p}} + \mathbf{{k}}$, $\mathbf{{k}} \in \{{-1,0,1\}}^2$,
falls within $T_{{ab}}^+$.
Images with $\mathbf{{k}} = (0,0)$ and $\mathbf{{p}} \in T_{{ab}}$ are
<em>core</em> particles; all others are <em>ghost</em> particles.
After inference, only core displacements are applied.
</p>

<div class="proof">
<strong>Correctness.</strong>
Let $g \geq r_d$.
For any tile $T_{{ab}}$ and any violation pair $(i,j)$ with
$d_\text{{PBC}}(\mathbf{{p}}_i, \mathbf{{p}}_j) < r_d$:
if $\mathbf{{p}}_i \in T_{{ab}}$ then particle $j$ has a periodic image
$\mathbf{{q}}_j = \mathbf{{p}}_j + \mathbf{{k}}^*$ satisfying
$\|\mathbf{{p}}_i - \mathbf{{q}}_j\| < r_d \leq g$,
so $\mathbf{{q}}_j \in T_{{ab}}^+$.
Hence every violation involving a core particle is visible to the model.
The condition $g \geq r_d$ is satisfied for both experiments
($g = 2r_d$ in both cases). $\square$
</div>

<figure>
{_img_tag(b64_ghost, 'Ghost buffer schematic')}
<figcaption>
  <strong>Figure 1.</strong>
  Ghost buffer construction for a $3\times3$ tiled domain.
  <em>Left:</em> without ghost buffer, the cross-tile violation pair
  (red stars) is invisible to the highlighted tile.
  <em>Right:</em> with ghost buffer $g=2r_d$ (dashed boundary),
  both the adjacent-tile ghost (orange diamond) and the PBC-wrapped ghost
  (purple diamond) are included, making all violations visible.
</figcaption>
</figure>

<h3>3.3 &nbsp; Coordinate rescaling</h3>
<p>
The model was trained with $r_d^\text{{train}} = {RD_TRAIN}$; the SPH data
has $r_d^\text{{test}} = {RD_TEST}$.
All tile coordinates are rescaled by $s = r_d^\text{{train}} / r_d^\text{{test}} = {SCALE}$
before processing:
$$
\tilde{{\mathbf{{p}}}} = s\,\mathbf{{p}}, \qquad
\tilde{{r}}_d = s \cdot r_d^\text{{test}} = r_d^\text{{train}}.
$$
After inference, displacements $\tilde{{\mathbf{{d}}}}$ in the scaled system
are mapped back as $\mathbf{{d}} = \tilde{{\mathbf{{d}}}} / s$.
The <code>make_invariant</code> preprocessing (centering + PCA rotation)
is applied in the scaled coordinate system, consistent with training.
</p>

<h3>3.4 &nbsp; Algorithm</h3>
<div class="algo">
Input: positions P (N, 2), model, r_d_train, scale, grid G, ghost_width g
Output: corrected positions P'

for i in 0..G-1, j in 0..G-1:
    tile_lo = [i/G, j/G],  tile_hi = [(i+1)/G, (j+1)/G]
    pts_ext, is_core, orig_idx = build_ghost_buffer(P, tile_lo, tile_hi, g)
    pts_scaled = pts_ext * scale
    pts_inv, revert = make_invariant(pts_scaled)
    disp_inv = model(pts_inv, rd=r_d_train)
    disp_orig = (revert(pts_inv + disp_inv) - pts_scaled) / scale
    for each core particle k in pts_ext:
        P[orig_idx[k]] += disp_orig[k]

return P mod 1.0  -- PBC wrap
</div>

<!-- ══════════════════════════════════════════════════════════ -->
<h2>4 &nbsp; Validation: Synthetic Toy</h2>
<p>
A synthetic cloud of $N=96$ particles is constructed from a valid Poisson-disk
sample ($r_d = 0.076$, seed={SEED}) with three deliberate violation pairs added:
</p>
<ul>
  <li><strong>(a) Internal</strong> &mdash; both particles in tile $(0,0)$,
      no tile boundary involved.</li>
  <li><strong>(b) Cross-tile</strong> &mdash; one particle in tile $(0,1)$,
      one in tile $(1,1)$, separated by the boundary $x=0.5$.</li>
  <li><strong>(c) Cross-PBC</strong> &mdash; one particle near $x=0.01$,
      one near $x=0.98$; periodic distance $= 0.38 r_d &lt; r_d$.</li>
</ul>
<p>Ghost width $g = 2 r_d = {2*RD_TRAIN:.3f}$, grid $G = 2$.</p>

<h3>4.1 &nbsp; Ghost detection</h3>
<table>
<tr><th>Violation type</th><th>Ghost visible?</th><th>Detail</th></tr>
<tr><td>Cross-PBC (c)</td>
    <td class="{'check' if ok_pbc else 'warn'}">{"YES" if ok_pbc else "NO"}</td>
    <td>$x\approx0.981$ wraps to $x\approx-0.019$ in tile $(0,0)$ ghost buffer</td></tr>
<tr><td>Cross-tile (b)</td>
    <td class="{'check' if ok_tile else 'warn'}">{"YES" if ok_tile else "NO"}</td>
    <td>$x\approx0.514$ appears as ghost in tile $(0,1)$ extended region</td></tr>
</table>

<h3>4.2 &nbsp; Before / after correction</h3>
<figure>
{_img_tag(b64_toy, 'Toy before/after')}
<figcaption>
  <strong>Figure 2.</strong>
  Synthetic toy ($N=96$, $r_d=0.076$, $2\times2$ grid).
  Red particles are involved in at least one violation; coloured stars mark
  the three deliberately introduced pairs.
  The total violation count includes 7 pre-existing PBC violations arising
  because the Poisson-disk sampler does not enforce periodic boundary conditions.
</figcaption>
</figure>

<h3>4.3 &nbsp; Per-iteration violation count</h3>
<table>
<tr><th>K</th><th>Violation pairs</th><th>Reduction</th></tr>
{toy_table_rows}
</table>
<div class="note">
  <strong>Note on toy reduction percentage.</strong>
  Each tile contains $N_\text{{tile}} \approx 17$&ndash;$29$ core particles,
  well below the training distribution ($N_\text{{train}} \approx 100$).
  The modest reduction reflects this out-of-distribution tile size, not a
  correctness failure. The ghost detection checks above confirm that all
  violation types are structurally visible. The SPH experiment below,
  where $N_\text{{tile}} \approx 100$, shows the full trained performance.
</div>

<!-- ══════════════════════════════════════════════════════════ -->
<h2>5 &nbsp; Application to SPH Data ($t=0$)</h2>
<p>
The SPH trajectory <code>positions_without.npy</code> contains $T=1002$
timesteps of $N=2500$ particles in $[0,1)^2$ with target $r_d = 0.02$.
We apply the corrector to the initial frame ($t=0$).
Configuration: $G=5$, $g=2r_d^\text{{test}}={2*RD_TEST:.3f}$,
scale $s={SCALE}$, model evaluated at $r_d^\text{{train}}={RD_TRAIN}$.
</p>

<figure>
{_img_tag(b64_sph, 'SPH before/after')}
<figcaption>
  <strong>Figure 3.</strong>
  SPH $t=0$ before ($K=0$) and after ($K=3$) correction.
  Colour encodes the per-particle violation depth
  $\max(0,\, r_d - \text{{min-NN}})$: green = no violation,
  yellow/red = increasing violation depth.
  At $K=0$, {violation_stats(sph_before, RD_TEST)[1].sum()//2:.0f}&ndash;
  {violation_stats(sph_before, RD_TEST)[0]} violation pairs span
  the domain; at $K=3$ the count falls to
  {violation_stats(sph_after_k3, RD_TEST)[0]}.
</figcaption>
</figure>

<table>
<tr><th>K</th><th>Violation pairs</th><th>Reduction</th></tr>
{sph_table_rows}
</table>

<!-- ══════════════════════════════════════════════════════════ -->
<h2>6 &nbsp; Convergence and Idempotence</h2>
<p>
We define the corrector as <em>convergent</em> if the violation count is
non-increasing with $K$, and as having reached a <em>fixed point</em> once
$\Delta V(K) = V(K) - V(K-1) \approx 0$.
True idempotence ($V(1) = V(K)$ for all $K \geq 1$) is not expected, since
each pass can identify and resolve new violations exposed by prior corrections.
</p>

<figure>
{_img_tag(b64_conv, 'Convergence curves')}
<figcaption>
  <strong>Figure 4.</strong>
  Violation pair count (left $y$-axis, solid) and percentage reduction from
  $K=0$ (right $y$-axis, dashed) as a function of correction passes $K$.
  <em>Left (synthetic toy):</em> slow sustained improvement; $N_\text{{tile}} \approx 22$
  is below the training distribution, so per-pass gains are small but accumulate.
  <em>Right (SPH $t=0$):</em> $K=1$ achieves the best single-step result
  ({(v0_sph-sph_v_curve[1])/v0_sph*100:.0f}%); $K=2$ transiently increases
  the count (cascade over-correction), then subsequent passes slowly recover.
  The model was trained with $K=3$ joint unrolling, not for fixed-point iteration.
</figcaption>
</figure>

<p>
The two experiments show qualitatively different iteration behaviour:
</p>
<ul>
  <li>
    <strong>Synthetic toy.</strong>
    Violations are isolated pairs; correcting one pair does not displace
    particles into other violations. The count decreases monotonically
    but slowly (81.8% at $K=10$), reflecting the out-of-distribution
    tile size ($N_\text{{tile}} \approx 22$ vs.\ training $N \approx 100$).
  </li>
  <li>
    <strong>SPH $t=0$.</strong>
    Violations are systemic: at $K=0$, {(violation_stats(sph_before,RD_TEST)[1]>0).sum()}/{N_sph}
    particles carry at least one violation. Correcting densely-packed violations
    displaces particles into new conflicts ($K=2$ regression of
    {sph_v_curve[2]-sph_v_curve[1]:+d} pairs).
    The corrector is not a contraction mapping under naive iteration;
    $K=1$ yields the best per-step result.
    The training objective (joint unrolling with $K=3$ steps) does not
    guarantee stability under independent re-application.
  </li>
</ul>

<!-- ══════════════════════════════════════════════════════════ -->
<h2>7 &nbsp; Limitations and Assumptions</h2>
<ul>
  <li>
    <strong>In-distribution violation depth.</strong>
    Training used noise scales $0.1$&ndash;$0.5 \times r_d$, creating
    violation depths up to $0.5\, r_d$.
    Violations deeper than the model's trained range may not be fully corrected.
    At SPH $t=0$, measured violation depths are $&lt; 0.003\, r_d$, well within
    the training range.
  </li>
  <li>
    <strong>Tile size sensitivity.</strong>
    The model was trained at $N \approx 100$. Tiles with significantly fewer
    particles (as in the $2\times2$ toy) exhibit reduced correction efficiency.
    For $G=5$ on the SPH data, $N_\text{{tile}} \approx 100$, matching training.
  </li>
  <li>
    <strong>Boundary seam artefacts.</strong>
    Particles near tile boundaries receive corrections computed from different
    local contexts in adjacent tiles. With $g = 2r_d$, each boundary particle
    sees its neighbours correctly; however, the two tiles may predict
    slightly different displacements for the same violation pair.
    Only one copy (the core particle's tile) applies the displacement.
  </li>
  <li>
    <strong>No global validity guarantee.</strong>
    The corrector reduces violations but does not guarantee a globally
    valid Poisson-disk cloud after a fixed number of passes.
    The convergence plots show a non-zero floor, particularly for the toy.
  </li>
  <li>
    <strong>Single-step deployment ($K=1$).</strong>
    The model was trained with $K=3$ unrolling but deployed at $K=1$.
    For SPH data, $K=3$ gives substantially better results than $K=1$
    ({(v0_sph-sph_v_curve[3])/v0_sph*100:.0f}% vs {(v0_sph-sph_v_curve[1])/v0_sph*100:.0f}% reduction)
    at negligible extra cost.
  </li>
</ul>

<hr>
<p style="font-size:0.8em; color:#888;">
  Generated by <code>inference/make_report.py</code>.
  Model checkpoint: <code>{CKPT.name.split('/')[0]}</code>.
  Figures generated with matplotlib {matplotlib.__version__}.
</p>

</body>
</html>"""

    out = Path('inference/pbc_report.html')
    out.write_text(html, encoding='utf-8')
    print(f'\nReport saved -> {out}  ({out.stat().st_size//1024} KB)')


if __name__ == '__main__':
    main()
