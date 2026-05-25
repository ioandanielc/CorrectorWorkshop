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