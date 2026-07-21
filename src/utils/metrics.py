"""
src/utils/metrics.py
--------------------
Shared point-cloud quality metrics — the single home for the SPH
kernel-gradient (KG) primitive and the nn-distance helpers, used by both the
training loss (differentiable torch path) and the inference experiments
(numpy convenience wrappers). Moved here from training/loss.py so inference
code no longer reaches into training internals.

Torch path (differentiable, batched):
    kernel_gradient (B, N, 2) -> (B, N, 2)   KG_i per particle
    kg_norm         (B, N, 2) -> (B,)        per-cloud mean |KG_i|
    mean_kg_norm    (B, N, 2) -> scalar      batch mean (training eval metric)

Numpy path (one (N, D) cloud, periodic box):
    mean_kg, nn_dists, mean_nn, illegal_frac
"""
import numpy as np
import torch
from scipy.spatial import cKDTree

# ── SPH quintic-spline kernel (Morris 1997), 2D ───────────────────────────────
# Differentiable torch port of the reference implementation (periodic KG):
#     KG_i = sum_j  dW/dr(r_ij) * e_ij * V_j,   V_j = dx^2,  dx = box / sqrt(N)
# For a symmetric neighbourhood (e.g. a regular periodic grid) KG_i = 0 — this
# is the SPH consistency condition the TV algorithm relaxes toward.

_SIGMA_2D = 7.0 / (478.0 * torch.pi)   # 2D normalisation constant
_SUPPORT  = 3.0                        # compact support: r < 3h


def _quintic_dw_dr(r, h):
    """Radial derivative dW/dr of the quintic spline (2D normalisation).

    The clamp formulation reproduces the piecewise cases exactly:
    q<1 all three terms, 1<=q<2 two, 2<=q<3 one, q>=3 zero.
    """
    q = r / h
    t3 = torch.clamp(3.0 - q, min=0.0) ** 4
    t2 = torch.clamp(2.0 - q, min=0.0) ** 4
    t1 = torch.clamp(1.0 - q, min=0.0) ** 4
    return (_SIGMA_2D / h ** 3) * (-5.0 * t3 + 30.0 * t2 - 75.0 * t1)


def _pbc_rel(x, box):
    """Minimum-image pairwise displacement vectors and distances. x: (B, N, D)."""
    rel = x.unsqueeze(2) - x.unsqueeze(1)          # (B, N, N, D)
    rel = rel - box * torch.round(rel / box)       # minimum image
    return rel, rel.norm(dim=-1)                   # (B, N, N)


def kernel_gradient(x, h_factor=2.0, box=1.0):
    """KG_i for every particle of every cloud, periodic. x: (B, N, 2) -> (B, N, 2)."""
    B, N, D = x.shape
    assert D == 2, "quintic-spline normalisation implemented for 2D only"
    dx = box / N ** 0.5                            # point spacing
    h  = h_factor * dx

    rel, r = _pbc_rel(x, box)
    eye    = torch.eye(N, dtype=torch.bool, device=x.device)
    safe_r = r.masked_fill(eye, 1.0).clamp_min(1e-12)
    dwdr   = _quintic_dw_dr(safe_r, h).masked_fill(eye, 0.0)
    e_ij   = rel / safe_r.unsqueeze(-1)            # coincident pairs: rel=0 -> 0
    return (dwdr.unsqueeze(-1) * e_ij).sum(dim=2) * dx ** 2


def kg_norm(x, h_factor=2.0, box=1.0):
    """Per-cloud mean |KG_i|. x: (B, N, 2) -> (B,)."""
    return kernel_gradient(x, h_factor, box).norm(dim=-1).mean(dim=-1)


def mean_kg_norm(x, h_factor=2.0, box=1.0):
    """Eval metric: mean |KG_i| over the batch (0 = perfectly symmetric)."""
    return kg_norm(x, h_factor, box).mean()


# ── numpy convenience wrappers (one cloud, periodic box) ──────────────────────

def mean_kg(points, h_factor=2.0, box=1.0):
    """Mean |KG_i| of one (N, 2) numpy cloud."""
    x = torch.tensor(np.asarray(points)[None], dtype=torch.float32)
    return float(mean_kg_norm(x, h_factor=h_factor, box=box))


def nn_dists(points, box=1.0):
    """(N,) nearest-neighbour distances of one (N, D) cloud on the torus."""
    p = np.asarray(points, dtype=np.float32) % box
    d, _ = cKDTree(p, boxsize=box).query(p, k=2)
    return d[:, 1]


def mean_nn(points, box=1.0):
    """Mean nn-distance of one cloud."""
    return float(nn_dists(points, box).mean())


def illegal_frac(points, rd, box=1.0):
    """Fraction of points whose nearest neighbour is closer than rd."""
    return float((nn_dists(points, box) < rd).mean())
