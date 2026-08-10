import torch

# KG primitive lives in utils/metrics.py (shared with inference experiments);
# kernel_gradient / mean_kg_norm are re-exported here so training-side callers
# keep their historical import path.
from utils.metrics import _pbc_rel, kernel_gradient, mean_kg_norm  # noqa: F401


def sph_loss(original_points, displaced_points, relaxation_distance,
             lambda_sph=1.0, lambda2=0.0, h_factor=2.0, box=1.0):
    """Pure SPH symmetry loss: force the kernel-gradient sum to zero.

    L = lambda_sph * mean_i |KG_i|^2  (+ optional lambda2 displacement reg).
    rd is unused by the objective; it is still passed to the model, whose
    proximity kernel only activates on pairs closer than rd.
    """
    kg   = kernel_gradient(displaced_points, h_factor, box)
    loss = lambda_sph * (kg ** 2).sum(dim=-1).mean()
    if lambda2:
        loss = loss + lambda2 * (displaced_points - original_points).norm(dim=-1).mean()
    return loss


def rdsph_loss(original_points, displaced_points, relaxation_distance,
               lambda1, lambda1_quad, lambda2, lambda3,
               h_factor=2.0, box=1.0):
    """Violation + displacement-reg loss with periodic distances, plus
    lambda3 * SPH symmetry term.

    Pair distances use the minimum image (training clouds are periodic); the
    quintic kernel-gradient term pulls neighbourhoods toward symmetry.
    Parameterisation recipe: lambda1 = 1/rd, lambda1_quad = 0,
    lambda2 = 0.1 * lambda1 / (N-1); lambda3 is the viol<->KG trade-off dial.
    """
    _, pairwise = _pbc_rel(displaced_points, box)
    mask = ~torch.eye(pairwise.shape[1], dtype=torch.bool, device=pairwise.device)

    viol = torch.relu(relaxation_distance - pairwise)[:, mask]
    illegality = lambda1 * viol.mean() + lambda1_quad * (viol ** 2).mean()

    displacement = (displaced_points - original_points).norm(dim=-1).mean()

    kg = kernel_gradient(displaced_points, h_factor, box)
    symmetry = (kg ** 2).sum(dim=-1).mean()

    return illegality + lambda2 * displacement + lambda3 * symmetry
