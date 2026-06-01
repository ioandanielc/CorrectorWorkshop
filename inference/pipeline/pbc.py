"""
pipeline/pbc.py
---------------
Periodic Boundary Condition utilities for 2D particle clouds in [0, L)^2.

All functions are pure numpy — no torch, no model dependencies.

Correctness guarantee for build_ghost_tile:
  If ghost_width >= rd, every pair (i,j) with d_PBC(pi,pj) < rd is jointly
  visible in at least one tile's ghost-augmented neighbourhood. Proof: let
  q_j = closest periodic image of p_j to p_i; ||pi - q_j|| < rd <= ghost_width,
  so q_j lies within ghost_width of the tile containing p_i. QED.
"""
import numpy as np


def pbc_dists(pts: np.ndarray, domain: float = 1.0) -> np.ndarray:
    """(N, N) matrix of pairwise PBC distances. O(N^2) memory."""
    diff = pts[:, None] - pts[None, :]
    diff = diff - domain * np.round(diff / domain)
    return np.linalg.norm(diff, axis=-1)


def min_nn_pbc(pts: np.ndarray, domain: float = 1.0) -> np.ndarray:
    """(N,) per-particle minimum nearest-neighbour distance under PBC."""
    D = pbc_dists(pts, domain)
    np.fill_diagonal(D, np.inf)
    return D.min(axis=1)


def compute_rdf(pts: np.ndarray, r_max: float,
                n_bins: int = 80, domain: float = 1.0):
    """
    2D radial distribution function g(r).

    Returns
    -------
    r_centers : (n_bins,)
    g_r       : (n_bins,)   g(r) -> 1 for uncorrelated, 0 for r < rd
    """
    N   = len(pts)
    rho = N / domain**2
    D   = pbc_dists(pts, domain)
    d_pairs = D[np.triu_indices(N, k=1)]
    d_pairs = d_pairs[d_pairs <= r_max]
    bins        = np.linspace(0, r_max, n_bins + 1)
    counts, _   = np.histogram(d_pairs, bins=bins)
    r_centers   = (bins[:-1] + bins[1:]) / 2
    dr          = bins[1] - bins[0]
    g_r = (2 * counts / N) / (2 * np.pi * r_centers * dr * rho)
    return r_centers, g_r


def build_ghost_tile(
    points: np.ndarray,
    tile_lo: np.ndarray,
    tile_hi: np.ndarray,
    ghost_width: float,
    domain: float = 1.0,
):
    """
    Collect core + ghost particles for one tile via all 9 periodic images.

    Parameters
    ----------
    points     : (N, 2)  all particle positions in [0, domain)^2
    tile_lo    : (2,)    lower-left corner of tile
    tile_hi    : (2,)    upper-right corner of tile (exclusive)
    ghost_width: float   buffer width; must be >= rd for correctness guarantee

    Returns
    -------
    pts_ext  : (M, 2)  positions in extended-tile coordinates
    is_core  : (M,)    True if the particle belongs to this tile (not a ghost)
    orig_idx : (M,)    index into `points` for each extended particle
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
