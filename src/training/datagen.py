import multiprocessing as mp
import time

import numpy as np
from scipy.stats import qmc

_SQRT3 = 3 ** 0.5
_SQRT2 = 2 ** 0.5


def _packed_n_target(rd, packedness, dim=2):
    """N_target = packedness fraction of the max-packing density at radius rd.

    2D: triangular lattice, N_max = 2 / (sqrt(3) * rd^2)
    3D: FCC lattice,        N_max = sqrt(2) / rd^3
    """
    if dim == 2:
        n_max = 2.0 / (_SQRT3 * rd ** 2)
    elif dim == 3:
        n_max = _SQRT2 / rd ** 3
    else:
        raise ValueError(f"_packed_n_target supports dim 2 or 3, got {dim}")
    return max(2, round(packedness * n_max))


def _gen_one_cloud(dim, rd, n_request, seed):
    """Module-level (picklable) worker for parallel pool generation — see
    VariableRdPackedPoissonDiskDataset._build_pool."""
    return qmc.PoissonDisk(d=dim, radius=rd, seed=seed).random(n_request)


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
        # One generator drives all randomness: the same dataset seed reproduces the
        # same *sequence* of batches, but every call yields fresh clouds and noise.
        # (The previous fixed-seed scheme regenerated the identical batch on every
        # call, so training effectively saw batch_size clouds total.)
        self._rng            = np.random.default_rng(seed)

    def _cloud_seeds(self, batch_size):
        if self.seed is None:
            return [None] * batch_size
        return [int(s) for s in self._rng.integers(0, 2**31 - 1, size=batch_size)]

    def generate_sample(self, batch_size):
        """Return (batch_size, cardinality, dim) array of valid clouds.

        scipy's sampler is stochastic and occasionally places fewer than
        `cardinality` points; such clouds are resampled with a fresh seed.
        """
        clouds = []
        for _ in range(batch_size):
            for _ in range(20):   # resample budget per cloud
                seed  = self._cloud_seeds(1)[0]
                cloud = qmc.PoissonDisk(
                    d=self.dim, radius=self.rd, seed=seed).random(self.cardinality)
                if len(cloud) == self.cardinality:
                    break
            else:
                raise RuntimeError(
                    f"PoissonDisk placed fewer than {self.cardinality} points at "
                    f"rd={self.rd} (dim={self.dim}) in 20 attempts — density too "
                    f"high for reliable sampling")
            clouds.append(cloud)
        return np.stack(clouds, axis=0)

    def noise_sample(self, sample):
        """Add per-cloud Gaussian noise (sigma drawn uniformly per cloud)."""
        sigmas = self._rng.uniform(self.noise_scale_min, self.noise_scale_max,
                                   size=sample.shape[0])
        noise  = self._rng.normal(0.0, 1.0, size=sample.shape) * sigmas[:, None, None]
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

    def __init__(self, dim, rd, packedness, seed,
                 noise_scale_min, noise_scale_max):
        if not (0 < packedness <= 1.0):
            raise ValueError(f"packedness must be in (0, 1], got {packedness}")

        n_target  = _packed_n_target(rd, packedness, dim)
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
            qmc.PoissonDisk(d=self.dim, radius=self.rd, seed=s).random(self.n_request)
            for s in self._cloud_seeds(batch_size)
        ]
        n_min  = min(len(c) for c in clouds)
        return np.stack([c[:n_min] for c in clouds], axis=0)

    def noise_sample(self, sample):
        """Add Gaussian noise and clamp to [0, 1]^dim."""
        return np.clip(super().noise_sample(sample), 0.0, 1.0)

    @property
    def points_per_cloud(self):
        return self.n_target


class VariableRdPackedPoissonDiskDataset:
    """Poisson disk dataset that trains one model across a wide range of rd.

    Sampling Poisson-disk clouds *directly* at large rd in a fixed [0,1]^dim domain is
    physically infeasible (e.g. rd=1.0 fits only ~1-2 points in the unit square, and with
    a whole batch of clouds the worst one frequently places just 1 — which then blows up
    make_invariant's covariance/eigh with a degenerate 1-point cloud). So instead every
    cloud is generated at one small, fixed `rd_gen` (packedness formula clipped to
    [n_min, n_max], same regime as PackedPoissonDiskDataset / dataset_config_packed.yaml —
    always physically safe), then uniformly rescaled by `rd_target / rd_gen` where
    rd_target is drawn log-uniformly from [rd_min, rd_max] on every call. This is exact,
    not an approximation: a Poisson-disk cloud valid at radius rd_gen is, after uniform
    rescaling, valid at radius rd_target too (all pairwise distances scale linearly with
    the coordinates). The model never assumes a bounded domain — make_invariant only
    centers + PCA-rotates — so points landing outside [0,1]^dim after rescaling are fine.

    noise_scale is a *fraction* of rd_target (not an absolute value) since rd spans two
    orders of magnitude across calls.

    Since every cloud is generated at the *same* rd_gen/packedness/N regardless of
    rd_target, generation doesn't need to be repeated every training iteration: a pool of
    `pool_size` base clouds is pre-generated once (in parallel across CPU cores — scipy's
    PoissonDisk sampler is ~6ms/cloud, single-threaded, and doesn't get faster by batching,
    so at batch_size~1000 the naive per-iteration approach is CPU-generation-bound, not
    GPU-bound). generate_sample() then just samples `batch_size` clouds from the pool
    (with replacement) and rescales them — no scipy calls on the training hot path.

    self.last_rd / self.last_n_target expose the values used by the most recent
    generate_sample() call — read these afterward to drive the model + loss for that batch.
    """

    def __init__(self, dim, rd_min, rd_max, packedness, seed,
                 noise_scale_min_frac, noise_scale_max_frac,
                 rd_gen=0.05, n_min=20, n_max=250,
                 pool_size=100_000, n_workers=None):
        if not (0 < packedness <= 1.0):
            raise ValueError(f"packedness must be in (0, 1], got {packedness}")

        self.dim                  = dim
        self.rd_min                = rd_min
        self.rd_max                = rd_max
        self.packedness            = packedness
        self.seed                  = seed
        self.noise_scale_min_frac  = noise_scale_min_frac
        self.noise_scale_max_frac  = noise_scale_max_frac
        self.rd_gen                = rd_gen
        self.n_min                 = n_min
        self.n_max                 = n_max
        self.pool_size             = pool_size
        self._rng                  = np.random.default_rng(seed)
        self.n_request             = int(np.clip(_packed_n_target(rd_gen, packedness, dim), n_min, n_max))
        self._pool                 = self._build_pool(pool_size, n_workers)
        self.last_rd               = None
        self.last_n_target         = None

    def _build_pool(self, pool_size, n_workers):
        seeds = [self.seed + i if self.seed is not None else None for i in range(pool_size)]
        args  = [(self.dim, self.rd_gen, self.n_request, s) for s in seeds]
        print(f"Building rd_gen={self.rd_gen} cloud pool "
              f"({pool_size} clouds, N_request={self.n_request})...", flush=True)
        t0 = time.time()
        with mp.Pool(n_workers or mp.cpu_count()) as pool:
            clouds = pool.starmap(_gen_one_cloud, args)
        n_placed = min(len(c) for c in clouds)
        print(f"Pool built in {time.time() - t0:.1f}s (N={n_placed})", flush=True)
        return np.stack([c[:n_placed] for c in clouds], axis=0)

    def generate_sample(self, batch_size):
        """Sample batch_size base clouds from the pool, rescale to a fresh log-uniform rd_target."""
        idx = self._rng.integers(0, self.pool_size, size=batch_size)
        clean_at_rd_gen = self._pool[idx]

        rd_target = float(10 ** self._rng.uniform(np.log10(self.rd_min), np.log10(self.rd_max)))
        self.last_rd       = rd_target
        self.last_n_target = clean_at_rd_gen.shape[1]
        return clean_at_rd_gen * (rd_target / self.rd_gen)

    def noise_sample(self, sample):
        """Add per-cloud Gaussian noise scaled by the most recent rd_target (no domain clip —
        the rescaled cloud is not bounded to [0, 1]^dim, and the model doesn't assume it is)."""
        if self.last_rd is None:
            raise RuntimeError("generate_sample() must be called before noise_sample()")
        sigmas = self._rng.uniform(self.noise_scale_min_frac * self.last_rd,
                                   self.noise_scale_max_frac * self.last_rd,
                                   size=sample.shape[0])
        noise  = self._rng.normal(0.0, 1.0, size=sample.shape) * sigmas[:, None, None]
        return sample + noise


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
