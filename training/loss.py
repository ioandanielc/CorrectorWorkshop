import torch


def classic_loss(original_points, displaced_points, relaxation_distance, lambda1, lambda2):
    # displacement magnitude
    displacement = (displaced_points - original_points).norm(dim=-1).mean()

    # pairwise distances: (B, cardinality, cardinality)
    pairwise = torch.cdist(displaced_points, displaced_points)

    # mask diagonal (self-distances)
    mask = ~torch.eye(pairwise.shape[1], dtype=torch.bool, device=pairwise.device)

    # illegality: relu(rd - dist) for each pair, averaged
    illegality = torch.relu(relaxation_distance - pairwise)[:, mask].mean()

    return lambda1 * illegality + lambda2 * displacement


def rd_weighted_loss(original_points, displaced_points, relaxation_distance, rd_min, rd_max, lambda1, lambda2):
    displacement = (displaced_points - original_points).norm(dim=-1).mean()

    pairwise = torch.cdist(displaced_points, displaced_points)
    mask = ~torch.eye(pairwise.shape[1], dtype=torch.bool, device=pairwise.device)
    illegality = torch.relu(relaxation_distance - pairwise)[:, mask].mean()

    weight = rd_max / relaxation_distance

    return weight * (lambda1 * illegality + lambda2 * displacement)


def hybrid_loss(original_points, displaced_points, relaxation_distance,
                lambda1, lambda1_quad, lambda2):
    """
    Linear illegality + quadratic bonus + displacement regularisation.

    The linear term (lambda1) maintains a strong, constant gradient even for
    small violations — fixing the collapse that pure quadratic suffers in late
    training. The quadratic term (lambda1_quad) adds extra gradient proportional
    to violation depth, pushing harder on severe violations early in training
    without weakening the signal on small ones.

    Gradient at violation depth d:
        classic_loss:   lambda1
        hybrid_loss:    lambda1 + 2 * lambda1_quad * d

    Recommended starting point: lambda1=30, lambda1_quad=200, lambda2=1
    """
    pairwise = torch.cdist(displaced_points, displaced_points)
    mask = ~torch.eye(pairwise.shape[1], dtype=torch.bool, device=pairwise.device)

    viol = torch.relu(relaxation_distance - pairwise)[:, mask]   # (B, N*(N-1))
    illegality = lambda1 * viol.mean() + lambda1_quad * (viol ** 2).mean()

    displacement = (displaced_points - original_points).norm(dim=-1).mean()

    return illegality + lambda2 * displacement


def coverage_loss(original_points, displaced_points, relaxation_distance,
                  lambda1, lambda2, lambda3, coverage_target):
    """
    Three-term loss designed for model6 (violation-aware edge network).

    Term 1 — Quadratic illegality (lambda1):
        relu(rd - dist)^2 per pair, averaged.
        Squared vs linear: mild violations get a small penalty, severe violations
        get a disproportionately large one — gradient pushes hardest on the worst pairs.
        Use larger lambda1 than classic_loss to compensate for the smaller magnitude
        (e.g. 500 vs 30).

    Term 2 — Displacement regularisation (lambda2):
        Mean L2 displacement from the noisy input. Prevents the model from making
        unnecessarily large corrections.

    Term 3 — Compactness / anti-overspacing (lambda3):
        relu(nn_dist - coverage_target * rd) per point, averaged.
        Penalises any point whose nearest neighbour in the corrected cloud is more
        than coverage_target * rd away. Without this, the model can trivially satisfy
        the Poisson disk constraint by spreading all points far apart — technically
        legal, but no longer a realistic Poisson disk distribution.
        coverage_target=3.0 gives generous room (won't trigger unless the cloud
        is genuinely over-spaced).

    Config params: lambda1, lambda2, lambda3, coverage_target
    """
    B, N, _ = displaced_points.shape
    pairwise = torch.cdist(displaced_points, displaced_points)  # (B, N, N)
    eye = torch.eye(N, dtype=torch.bool, device=pairwise.device)
    mask = ~eye

    # Term 1: quadratic illegality
    violations = torch.relu(relaxation_distance - pairwise)[:, mask]  # (B, N*(N-1))
    illegality = (violations ** 2).mean()

    # Term 2: displacement from noisy input
    displacement = (displaced_points - original_points).norm(dim=-1).mean()

    # Term 3: compactness — nearest-neighbour distance shouldn't exceed coverage_target * rd
    nn_dist = pairwise.masked_fill(eye.unsqueeze(0), float('inf')).min(dim=-1).values  # (B, N)
    overspacing = torch.relu(nn_dist - coverage_target * relaxation_distance).mean()

    return lambda1 * illegality + lambda2 * displacement + lambda3 * overspacing