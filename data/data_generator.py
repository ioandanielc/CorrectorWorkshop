from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import qmc, uniform, norm
from utils.visualizations import plot_poisson_disk


class PoissonDiskDataset():
    def __init__(self, dim, cardinality, rd, seed, noise_scale_min, noise_scale_max, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.cardinality = cardinality
        self.rd = rd

        self.seed = seed
        self.noise_scale_min = noise_scale_min
        self.noise_scale_max = noise_scale_max

    def generate_sample(self, batch_size, *args, **kwargs):
        return np.stack([
            qmc.PoissonDisk(d=self.dim, radius=self.rd,
                            seed=self.seed + i if self.seed is not None else None).random(self.cardinality)
            for i in range(batch_size)
        ], axis=0)

    def noise_sample(self, sample, *args, **kwargs):
        # sample shape: (batch_size, cardinality, dim)
        u = uniform(loc=self.noise_scale_min, scale=self.noise_scale_max - self.noise_scale_min)
        sigmas = u.rvs(size=sample.shape[0], random_state=self.seed)  # one per cloud
        noise = np.stack([
            norm(loc=0.0, scale=sigma).rvs(size=sample.shape[1:], random_state=self.seed)
            for sigma, cloud in zip(sigmas, sample)
        ])
        return sample + noise

    def make_translational_invariant(self, sample):
        centroids = sample.mean(axis=1, keepdims=True)  # (batch_size, 1, dim)
        return sample - centroids

    def make_rotational_invariant(self, sample):
        aligned = []
        for cloud in sample:
            cov = np.cov(cloud, rowvar=False)
            _, eigvecs = np.linalg.eigh(cov)
            eigvecs = eigvecs[:, ::-1]
            aligned.append(cloud @ eigvecs)
        return np.stack(aligned)

    def make_invariant(self, sample):
        sample = self.make_translational_invariant(sample)
        sample = self.make_rotational_invariant(sample)
        return sample



class PackedPoissonDiskDataset(PoissonDiskDataset):
    """Poisson disk dataset parameterised by packedness instead of explicit N.

    packedness ∈ (0, 1] is the fraction of the theoretical maximum 2-D packing
    density (triangular lattice):

        N_target = round(packedness * 2 / (sqrt(3) * rd²))

    This makes the difficulty description rd-independent: packedness=0.85 means
    the same geometric situation regardless of rd or domain size — 85% of the
    maximum number of points that could ever fit.

    At packedness = 1.0 (saturation mode) scipy is asked for more points than
    the domain can hold; it places as many as it can. Because different seeds
    return different counts, generate_sample truncates every cloud in the batch
    to the minimum count seen, ensuring a uniform (B, N, dim) tensor.

    Config keys: rd, packedness, seed, noise_scale_min, noise_scale_max, dim.
    """

    # theoretical max density for 2-D triangular lattice (points per unit area)
    _SQRT3 = 3 ** 0.5

    def __init__(self, dim, rd, packedness, seed, noise_scale_min, noise_scale_max,
                 *args, **kwargs):
        if not (0 < packedness <= 1.0):
            raise ValueError(f"packedness must be in (0, 1], got {packedness}")

        # Theoretical max for 2-D; for dim>2 this is an approximation
        n_target = max(2, round(packedness * 2.0 / (self._SQRT3 * rd ** 2)))
        # At saturation, ask for far more than can fit — scipy caps at domain limit
        n_request = n_target if packedness < 1.0 else 10_000

        super().__init__(
            dim=dim,
            cardinality=n_request,   # passed to parent but overridden in generate_sample
            rd=rd,
            seed=seed,
            noise_scale_min=noise_scale_min,
            noise_scale_max=noise_scale_max,
            *args, **kwargs,
        )
        self.packedness = packedness
        self.n_target = n_target
        self.n_request = n_request

    def generate_sample(self, batch_size, *args, **kwargs):
        clouds = [
            qmc.PoissonDisk(d=self.dim, radius=self.rd,
                            seed=self.seed + i if self.seed is not None else None
                            ).random(self.n_request)
            for i in range(batch_size)
        ]
        # Truncate all clouds to the smallest count (handles saturation variance)
        n_min = min(len(c) for c in clouds)
        clouds = [c[:n_min] for c in clouds]
        return np.stack(clouds, axis=0)   # (B, n_min, dim)

    def noise_sample(self, sample, *args, **kwargs):
        """Add Gaussian noise then clamp to [0, 1]^dim (domain boundary enforcement)."""
        noisy = super().noise_sample(sample, *args, **kwargs)
        return np.clip(noisy, 0.0, 1.0)

    @property
    def points_per_cloud(self):
        """Actual N after one generate_sample call (approximation before first call)."""
        return self.n_target


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rd = 0.05
    packedness_levels = [0.11, 0.4, 0.7, 0.9, 1.0]
    noise = 0.008

    fig, axes = plt.subplots(2, len(packedness_levels),
                             figsize=(4 * len(packedness_levels), 8))
    fig.patch.set_facecolor('#0f1117')

    for col, p in enumerate(packedness_levels):
        gen = PackedPoissonDiskDataset(
            dim=2, rd=rd, packedness=p, seed=42,
            noise_scale_min=noise, noise_scale_max=noise,
        )
        clean = gen.generate_sample(batch_size=1)
        noisy = gen.noise_sample(clean)
        N = clean.shape[1]

        for row, (cloud, label) in enumerate([(clean[0], 'clean'), (noisy[0], 'noisy')]):
            ax = axes[row, col]
            ax.set_facecolor('#141821')
            from scipy.spatial import cKDTree
            tree = cKDTree(cloud)
            dists, _ = tree.query(cloud, k=2)
            colors = ['#f87171' if d < rd else '#4ade80' for d in dists[:, 1]]
            ax.scatter(cloud[:, 0], cloud[:, 1], c=colors, s=8, zorder=3)
            import matplotlib.patches as patches
            for pt, c in zip(cloud, colors):
                ax.add_patch(patches.Circle(pt, rd / 2, fill=False,
                                            linestyle='--', linewidth=0.4,
                                            color=c, alpha=0.25))
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_aspect('equal')
            ax.tick_params(colors='#475569', labelsize=6)
            for spine in ax.spines.values():
                spine.set_color('#1e2a3a')
            if row == 0:
                ax.set_title(f'packedness={p}\nN={N}',
                             color='#e2e8f0', fontsize=10, pad=8)
            if col == 0:
                ax.set_ylabel(label, color='#94a3b8', fontsize=9)

    fig.suptitle(f'PackedPoissonDiskDataset  —  rd={rd}  noise_σ={noise}',
                 color='#f8fafc', fontsize=13, y=1.01)
    fig.tight_layout()
    plt.savefig('packed_examples.png', dpi=120, bbox_inches='tight',
                facecolor='#0f1117')
    print('Saved packed_examples.png')
    plt.show()