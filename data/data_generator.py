import numpy as np
from scipy.stats import qmc, uniform, norm


class PoissonDiskDataset:
    """Synthetic Poisson disk dataset with online (per-batch) generation.

    Generates clouds of `cardinality` points with minimum pairwise distance `rd`
    using scipy's PoissonDisk sampler, then adds Gaussian noise to create
    constraint violations.

    Preprocessing (centering + PCA rotation) is NOT done here — use DataProcessor.
    """

    def __init__(self, dim, cardinality, rd, seed,
                 noise_scale_min, noise_scale_max):
        self.dim             = dim
        self.cardinality     = cardinality
        self.rd              = rd
        self.seed            = seed
        self.noise_scale_min = noise_scale_min
        self.noise_scale_max = noise_scale_max

    def generate_sample(self, batch_size):
        """Return (batch_size, cardinality, dim) array of valid clouds."""
        return np.stack([
            qmc.PoissonDisk(
                d=self.dim, radius=self.rd,
                seed=self.seed + i if self.seed is not None else None,
            ).random(self.cardinality)
            for i in range(batch_size)
        ], axis=0)

    def noise_sample(self, sample):
        """Add per-cloud Gaussian noise. Returns same shape as input."""
        u      = uniform(loc=self.noise_scale_min,
                         scale=self.noise_scale_max - self.noise_scale_min)
        sigmas = u.rvs(size=sample.shape[0], random_state=self.seed)
        noise  = np.stack([
            norm(loc=0.0, scale=sigma).rvs(
                size=sample.shape[1:], random_state=self.seed)
            for sigma in sigmas
        ])
        return sample + noise


class PackedPoissonDiskDataset(PoissonDiskDataset):
    """Poisson disk dataset parameterised by packedness instead of explicit N.

    packedness ∈ (0, 1] is the fraction of the theoretical 2D triangular-lattice
    maximum density:

        N_target = round(packedness × 2 / (√3 × rd²))

    This makes difficulty rd-independent: packedness=0.85 means the same geometric
    situation regardless of rd — 85% of the maximum possible packing.

    At packedness=1.0 (saturation), scipy is asked for more points than can fit
    and places as many as it can. generate_sample truncates all clouds in the batch
    to the minimum count to ensure a uniform (B, N, dim) tensor.

    Config keys: dim, rd, packedness, seed, noise_scale_min, noise_scale_max.
    """

    _SQRT3 = 3 ** 0.5

    def __init__(self, dim, rd, packedness, seed,
                 noise_scale_min, noise_scale_max):
        if not (0 < packedness <= 1.0):
            raise ValueError(f"packedness must be in (0, 1], got {packedness}")

        n_target  = max(2, round(packedness * 2.0 / (self._SQRT3 * rd ** 2)))
        n_request = n_target if packedness < 1.0 else 10_000

        super().__init__(
            dim=dim, cardinality=n_request, rd=rd, seed=seed,
            noise_scale_min=noise_scale_min, noise_scale_max=noise_scale_max,
        )
        self.packedness = packedness
        self.n_target   = n_target
        self.n_request  = n_request

    def generate_sample(self, batch_size):
        clouds = [
            qmc.PoissonDisk(
                d=self.dim, radius=self.rd,
                seed=self.seed + i if self.seed is not None else None,
            ).random(self.n_request)
            for i in range(batch_size)
        ]
        n_min  = min(len(c) for c in clouds)
        return np.stack([c[:n_min] for c in clouds], axis=0)

    def noise_sample(self, sample):
        """Add Gaussian noise and clamp to [0, 1]^dim."""
        return np.clip(super().noise_sample(sample), 0.0, 1.0)

    @property
    def points_per_cloud(self):
        return self.n_target


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from pathlib import Path
    from scipy.spatial import cKDTree

    rd               = 0.05
    packedness_levels = [0.11, 0.4, 0.7, 0.9, 1.0]
    noise             = 0.008

    fig, axes = plt.subplots(2, len(packedness_levels),
                             figsize=(4 * len(packedness_levels), 8))
    fig.patch.set_facecolor('#0f1117')

    for col, p in enumerate(packedness_levels):
        gen   = PackedPoissonDiskDataset(dim=2, rd=rd, packedness=p, seed=42,
                                         noise_scale_min=noise, noise_scale_max=noise)
        clean = gen.generate_sample(batch_size=1)
        noisy = gen.noise_sample(clean)
        N     = clean.shape[1]

        for row, (cloud, label) in enumerate([(clean[0], 'clean'), (noisy[0], 'noisy')]):
            ax   = axes[row, col]
            ax.set_facecolor('#141821')
            tree = cKDTree(cloud)
            dists, _ = tree.query(cloud, k=2)
            colors = ['#f87171' if d < rd else '#4ade80' for d in dists[:, 1]]
            ax.scatter(cloud[:, 0], cloud[:, 1], c=colors, s=8, zorder=3)
            for pt, c in zip(cloud, colors):
                ax.add_patch(patches.Circle(pt, rd / 2, fill=False,
                                             ls='--', lw=0.4, color=c, alpha=0.25))
            ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
            ax.set_aspect('equal')
            ax.tick_params(colors='#475569', labelsize=6)
            for spine in ax.spines.values():
                spine.set_color('#1e2a3a')
            if row == 0:
                ax.set_title(f'packedness={p}\nN={N}', color='#e2e8f0',
                             fontsize=10, pad=8)
            if col == 0:
                ax.set_ylabel(label, color='#94a3b8', fontsize=9)

    fig.suptitle(f'PackedPoissonDiskDataset  rd={rd}  noise_σ={noise}',
                 color='#f8fafc', fontsize=13, y=1.01)
    fig.tight_layout()

    out = Path('analysis/outputs/packed_examples.png')
    plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='#0f1117')
    print(f'Saved {out}')
