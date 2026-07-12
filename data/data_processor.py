import numpy as np


def compute_invariant(points, mask=None):
    """
    Center and PCA-rotate a batch of 2D point clouds.

    points : (B, N, 2)
    mask   : (B, N) bool/float, True/1 for real points; None = every point is real.
             Used for padded batches (e.g. tiles with unequal particle counts) so
             padding doesn't skew the centroid/covariance.

    Returns
    -------
    x_inv   : (B, N, 2)  centered, then rotated into descending-eigenvalue PCA frame
    mean    : (B, 2)     per-cloud centroid (over real points only)
    eigvecs : (B, 2, 2)  rotation applied (columns = eigenvectors, descending eigenvalue)
    """
    if mask is None:
        mask = np.ones(points.shape[:2], dtype=np.float32)
    else:
        mask = mask.astype(np.float32)

    n_real = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)    # (B, 1)
    mean = (points * mask[..., None]).sum(axis=1) / n_real       # (B, 2)
    centered = points - mean[:, None]                            # (B, N, 2)
    centered_real = centered * mask[..., None]

    cov = (np.matmul(centered_real.transpose(0, 2, 1), centered_real)
           / n_real[..., None])                                  # (B, 2, 2)
    _, eigvecs = np.linalg.eigh(cov)      # ascending eigenvalues
    eigvecs = eigvecs[:, :, ::-1]         # descending

    x_inv = np.matmul(centered, eigvecs)
    return x_inv, mean, eigvecs


def revert_invariant(x_inv, mean, eigvecs):
    """Undo compute_invariant: rotate back, then re-add the centroid."""
    return np.matmul(x_inv, eigvecs.transpose(0, 2, 1)) + mean[:, None]


class DataProcessor:
    def make_invariant(self, sample):
        x_inv, mean, eigvecs = compute_invariant(sample)

        def revert(s):
            return revert_invariant(s, mean, eigvecs)

        return x_inv, {"centroids": mean[:, None], "eigvecs": eigvecs}, revert
