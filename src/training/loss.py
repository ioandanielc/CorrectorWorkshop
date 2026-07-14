import torch


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
