"""
pbc_toy.py
----------
Proves the 2D PBC tiling + ghost-buffer approach for the Poisson-disk corrector.

Synthetic test in [0,1]^2 at rd = rd_train = 0.076 (no rescaling needed).

Introduces three deliberate violations:
  (a) INTERNAL  -- both points inside tile (0,0), no boundary
  (b) CROSS-TILE -- one point in tile (0,0), one in tile (1,0), near x=0.5
  (c) CROSS-PBC  -- one point near x=0, one near x=1 (PBC distance < rd)

Checks that each violation is visible in the relevant tile's ghost buffer,
then runs K=1 correction and reports before/after violation counts.

Usage:
    .venv\\Scripts\\python.exe inference/pbc_toy.py
    .venv\\Scripts\\python.exe inference/pbc_toy.py --checkpoint <path>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_processor import DataProcessor
from utils.config import load_model_config

# -- constants ----------------------------------------------------------------
DEFAULT_CKPT = Path("weights/n100_p050.pt")
DEFAULT_CFG  = Path("configs/model_configs/model_config_9_n100_p050.yaml")
RD           = 0.076
GHOST_WIDTH  = 2.0 * RD     # ghost buffer around each tile
GRID         = 2            # 2x2 tiling
DOMAIN       = 1.0
PUSH         = 0.38 * RD    # pair separation for introduced violations (~0.4*rd < rd)
SEED         = 7


# -- PBC utilities -------------------------------------------------------------
def pbc_dists(pts, domain=DOMAIN):
    """Return (N, N) matrix of PBC pairwise distances."""
    diff = pts[:, None] - pts[None, :]        # (N, N, 2)
    diff = diff - domain * np.round(diff / domain)
    return np.linalg.norm(diff, axis=-1)


def count_violations(pts, rd=RD, domain=DOMAIN):
    D = pbc_dists(pts, domain)
    N = len(pts)
    upper = np.triu(np.ones((N, N), bool), k=1)
    return int((D[upper] < rd).sum())


# -- tile + ghost buffer -------------------------------------------------------
def build_tile(points, tile_lo, tile_hi, ghost_width, domain=DOMAIN):
    """
    Collect core + ghost points for one tile via all 9 periodic images.

    Returns
    -------
    pts_ext  : (M, 2) -- positions in extended-tile coordinates
    is_core  : (M,)   -- True if point belongs to this tile (not a ghost)
    orig_idx : (M,)   -- index into `points` for each extended point
    """
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

    return (np.vstack(pts_list),
            np.concatenate(core_list),
            np.concatenate(idx_list))


# -- model inference on one tile -----------------------------------------------
def correct_tile(pts_ext, is_core, orig_idx, model, rd_t, processor, device):
    """Run model on tile, return {orig_index: displacement} for core points."""
    if is_core.sum() == 0:
        return {}

    x_np          = pts_ext[None]                              # (1, M, 2)
    x_inv, _, rev = processor.make_invariant(x_np)
    x_t           = torch.tensor(x_inv, dtype=torch.float32, device=device)

    with torch.no_grad():
        disp_inv = model(x_t, rd=rd_t).cpu().numpy()[0]       # (M, 2)

    corrected = rev(x_inv + disp_inv)[0]                      # (M, 2)
    disp_orig = corrected - pts_ext

    return {int(orig_idx[i]): disp_orig[i]
            for i in range(len(pts_ext)) if is_core[i]}


def run_corrector(points, model, rd_t, processor, device,
                  grid=GRID, ghost_width=GHOST_WIDTH, domain=DOMAIN):
    """One K=1 pass of the tiled PBC corrector."""
    cell = domain / grid
    displacements = np.zeros_like(points)
    counts        = np.zeros(len(points), dtype=int)

    for i in range(grid):
        for j in range(grid):
            lo = np.array([i * cell, j * cell])
            hi = np.array([(i+1)*cell, (j+1)*cell])
            pts_ext, is_core, orig_idx = build_tile(points, lo, hi, ghost_width, domain)
            for idx, disp in correct_tile(pts_ext, is_core, orig_idx,
                                          model, rd_t, processor, device).items():
                displacements[idx] += disp
                counts[idx] += 1

    counts = np.maximum(counts, 1)
    return (points + displacements / counts[:, None]) % domain


# -- main ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    parser.add_argument("--config",     default=str(DEFAULT_CFG))
    args = parser.parse_args()

    device = torch.device("cpu")

    # -- load model ------------------------------------------------------------
    model_cfg = load_model_config(args.config)
    import importlib
    m = importlib.import_module("models.fixed_rd.model9")
    model = m.CorrectorModel(model_cfg, input_dim=2, initialization="xavier_uniform")
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    rd_t      = torch.tensor(RD, dtype=torch.float32)
    processor = DataProcessor()
    print(f"Loaded: {args.checkpoint}")

    # -- build base cloud (valid Poisson disk) ---------------------------------
    from scipy.stats.qmc import PoissonDisk
    sampler = PoissonDisk(d=2, radius=RD, seed=SEED)
    pts = sampler.random(200).astype(np.float32)   # over-sample, will trim
    # keep only points not too close to positions we'll place violations at
    # reserved positions (we'll overwrite indices 0-5 below)
    pts = pts[6:][:90]   # take 90 background points (indices 6+ in sampler output)
    print(f"Base cloud: {len(pts)} valid points  |  initial violations: {count_violations(pts)}")

    # -- introduce 3 deliberate violations -------------------------------------
    cell = DOMAIN / GRID   # 0.5

    # (a) INTERNAL -- both in tile (0,0), well away from x=0.5 and y=0.5
    pa1 = np.array([0.20, 0.20], dtype=np.float32)
    pa2 = np.array([0.20 + PUSH, 0.20], dtype=np.float32)

    # (b) CROSS-TILE -- straddles x=0.5; pa3 in tile (0,j), pa4 in tile (1,j)
    pb1 = np.array([cell - PUSH/2, 0.75], dtype=np.float32)
    pb2 = np.array([cell + PUSH/2, 0.75], dtype=np.float32)

    # (c) CROSS-PBC -- PBC distance across x=0 / x=1 boundary
    # x_left + (1 - x_right) = PUSH  ->  distance = PUSH
    x_left  = PUSH * 0.35
    x_right = 1.0 - PUSH * 0.65
    pc1 = np.array([x_left,  0.50], dtype=np.float32)
    pc2 = np.array([x_right, 0.50], dtype=np.float32)

    pts = np.vstack([pts, pa1, pa2, pb1, pb2, pc1, pc2])
    ia1, ia2 = len(pts)-6, len(pts)-5
    ib1, ib2 = len(pts)-4, len(pts)-3
    ic1, ic2 = len(pts)-2, len(pts)-1

    v_before = count_violations(pts)
    print(f"After introducing violations: {v_before} violation pairs  (3 intended)")
    print(f"  (a) internal  dist={np.linalg.norm(pa2-pa1):.4f}  (rd={RD})")
    print(f"  (b) cross-tile dist={np.linalg.norm(pb2-pb1):.4f}")
    pbc_c = min(abs(pc1[0]-pc2[0]), DOMAIN - abs(pc1[0]-pc2[0]))
    print(f"  (c) cross-PBC  PBC-dist={pbc_c:.4f}")

    # -- ghost-buffer diagnostics ----------------------------------------------
    print("\n-- Ghost-buffer diagnostics --")
    for i in range(GRID):
        for j in range(GRID):
            lo = np.array([i*cell, j*cell])
            hi = np.array([(i+1)*cell, (j+1)*cell])
            ext, is_core, oidx = build_tile(pts, lo, hi, GHOST_WIDTH)

            # violations visible inside this tile (Euclidean in extended coords)
            D_tile  = np.linalg.norm(ext[:, None] - ext[None, :], axis=-1)
            n_viol  = int(np.triu(D_tile < RD, k=1).sum())

            print(f"  tile({i},{j}) [{lo[0]:.2f},{hi[0]:.2f}]x[{lo[1]:.2f},{hi[1]:.2f}]  "
                  f"{is_core.sum():3d} core + {(~is_core).sum():3d} ghost  "
                  f"{n_viol} visible violation pairs")

    # explicit cross-boundary checks
    lo00 = np.array([0.0, 0.0]); hi00 = np.array([cell, cell])
    lo10 = np.array([cell, 0.0]); hi10 = np.array([DOMAIN, cell])
    lo01 = np.array([0.0, cell]); hi01 = np.array([cell, DOMAIN])

    _, _, oidx00 = build_tile(pts, lo00, hi00, GHOST_WIDTH)
    ext_t, is_core_t, oidx_t = build_tile(pts, lo01, hi01, GHOST_WIDTH)

    ghost_mask = ~is_core_t

    ok_pbc  = ic2 in oidx00                                   # pc2 (x~0.97) wraps into tile(0,0)
    ok_tile = ib2 in oidx_t[ghost_mask] if ib2 in oidx_t else False  # pb2 appears as ghost in tile(0,1)

    print(f"\n  cross-PBC  ghost in tile(0,0): {ok_pbc}"
          f"   [pc2 at x={pc2[0]:.3f} wraps to x~{pc2[0]-1:.3f}]")
    print(f"  cross-tile ghost in tile(0,1): {ok_tile}"
          f"   [pb2 at x={pb2[0]:.3f} appears as ghost]")

    # -- run corrector K=1 and K=3 --------------------------------------------
    K = 3
    current = pts.copy()
    print()
    for k in range(1, K + 1):
        current   = run_corrector(current, model, rd_t, processor, device)
        v_k       = count_violations(current)
        reduction = (v_before - v_k) / max(v_before, 1) * 100
        print(f"  K={k}  violations: {v_k:3d}  ({reduction:.1f}% reduction from start)")
    corrected = current
    v_after   = v_k

    # -- plot ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, p, title in zip(axes,
                             [pts, corrected],
                             ["Before -- violations marked red", "After K=1 correction"]):
        # background points
        ax.scatter(p[:-6, 0], p[:-6, 1], s=12, alpha=0.45, c='steelblue')

        # violation pairs (PBC-aware)
        D = pbc_dists(p)
        N_ = len(p)
        for ii in range(N_):
            for jj in range(ii+1, N_):
                if D[ii, jj] < RD:
                    ax.scatter(p[[ii, jj], 0], p[[ii, jj], 1],
                               s=55, c='red', zorder=5, marker='o')

        # introduced pairs with distinct markers
        colors = ['orange', 'green', 'purple']
        labels = ['(a) internal', '(b) cross-tile', '(c) cross-PBC']
        for (i1, i2), c, lbl in zip([(ia1,ia2),(ib1,ib2),(ic1,ic2)], colors, labels):
            ax.scatter(p[[i1,i2], 0], p[[i1,i2], 1],
                       s=100, c=c, zorder=6, marker='*', label=lbl)

        # tile grid
        for k in range(GRID+1):
            v = k * cell
            ax.axhline(v, color='gray', lw=0.6, ls='--', alpha=0.5)
            ax.axvline(v, color='gray', lw=0.6, ls='--', alpha=0.5)

        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=11)
        if ax is axes[0]:
            ax.legend(fontsize=8, loc='upper right')

    plt.suptitle(
        f"PBC toy  rd={RD}  ghost={GHOST_WIDTH:.3f}  grid={GRID}x{GRID}  "
        f"violations: {v_before} -> {v_after}  ({reduction:.0f}% reduction)",
        fontsize=10,
    )
    out = Path("inference/pbc_toy_result.png")
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print(f"\nPlot saved -> {out}")


if __name__ == "__main__":
    main()
