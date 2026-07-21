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
