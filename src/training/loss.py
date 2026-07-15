import torch

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


def mean_kg_norm(x, h_factor=2.0, box=1.0):
    """Eval metric: mean |KG_i| over the batch (0 = perfectly symmetric)."""
    return kernel_gradient(x, h_factor, box).norm(dim=-1).mean()


def sph_loss(original_points, displaced_points, relaxation_distance,
             lambda_sph=1.0, lambda2=0.0, h_factor=2.0, box=1.0):
    """Pure SPH symmetry loss: force the kernel-gradient sum to zero.

    L = lambda_sph * mean_i |KG_i|^2  (+ optional lambda2 displacement reg).
    rd is unused by the objective; it is still passed to the model, whose
    attention only activates on pairs closer than rd.
    """
    kg   = kernel_gradient(displaced_points, h_factor, box)
    loss = lambda_sph * (kg ** 2).sum(dim=-1).mean()
    if lambda2:
        loss = loss + lambda2 * (displaced_points - original_points).norm(dim=-1).mean()
    return loss


def rdsph_loss(original_points, displaced_points, relaxation_distance,
               lambda1, lambda1_quad, lambda2, lambda3,
               h_factor=2.0, box=1.0):
    """hybrid_loss with periodic distances + lambda3 * SPH symmetry term.

    Same violation + displacement structure as hybrid_loss, but pair distances
    use the minimum image (training clouds are periodic), and the quintic
    kernel-gradient term pulls neighbourhoods toward symmetry.
    """
    _, pairwise = _pbc_rel(displaced_points, box)
    mask = ~torch.eye(pairwise.shape[1], dtype=torch.bool, device=pairwise.device)

    viol = torch.relu(relaxation_distance - pairwise)[:, mask]
    illegality = lambda1 * viol.mean() + lambda1_quad * (viol ** 2).mean()

    displacement = (displaced_points - original_points).norm(dim=-1).mean()

    kg = kernel_gradient(displaced_points, h_factor, box)
    symmetry = (kg ** 2).sum(dim=-1).mean()

    return illegality + lambda2 * displacement + lambda3 * symmetry


def hybrid_loss(original_points, displaced_points, relaxation_distance,
                lambda1, lambda1_quad, lambda2):
    """
    Active loss function for model9 training.

    Linear illegality + optional quadratic bonus + displacement regularisation.

    Gradient at violation depth d:
        pure linear:  lambda1
        hybrid:       lambda1 + 2 * lambda1_quad * d

    Principled parameterisation (from loss_config_5 / loss_config_packed):
        lambda1      = 1 / rd
        lambda1_quad = 0          (pure linear; quadratic disabled)
        lambda2      = lambda1 / (N-1) / 10
    The last formula keeps the displacement penalty a fixed fraction (1/10) of
    the violation penalty regardless of cloud size N.
    """
    pairwise = torch.cdist(displaced_points, displaced_points)
    mask = ~torch.eye(pairwise.shape[1], dtype=torch.bool, device=pairwise.device)

    viol = torch.relu(relaxation_distance - pairwise)[:, mask]   # (B, N*(N-1))
    illegality = lambda1 * viol.mean() + lambda1_quad * (viol ** 2).mean()

    displacement = (displaced_points - original_points).norm(dim=-1).mean()

    return illegality + lambda2 * displacement
