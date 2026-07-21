import numpy as np
from scipy.stats import qmc


class PoissonDiskDataset:
    """Synthetic Poisson disk dataset with online (per-batch) generation.

    Generates clouds of `cardinality` points with minimum pairwise distance `rd`
    using scipy's PoissonDisk sampler, then adds Gaussian noise to create
    constraint violations. `periodic=True` (the model12/SPH regime) enforces rd
    under minimum-image on the unit torus — see __init__.
    """

    def __init__(self, dim, cardinality, rd, seed,
                 noise_scale_min, noise_scale_max, periodic=False):
        self.dim             = dim
        self.cardinality     = cardinality
        self.rd              = rd
        self.seed            = seed
        self.noise_scale_min = noise_scale_min
        self.noise_scale_max = noise_scale_max
        # periodic=True: rd is enforced under minimum-image on the unit torus
        # and noise wraps mod 1 — for the SPH losses, whose kernel sums are
        # periodic. Clouds then must NOT be passed through make_invariant
        # (rotation would break the box alignment); the trainer skips it.
        # Above ~60% of max density dart throwing jams; there clean clouds are
        # generated as a randomly translated square lattice instead (the only
        # valid configurations at that packing are near-lattice anyway — the
        # noise step provides all the disorder, mirroring the SPH regime).
        self.periodic        = periodic
        if periodic:
            n_side = round(cardinality ** 0.5)
            self._lattice_side = n_side if n_side * n_side == cardinality else None
            spacing = 1.0 / n_side if self._lattice_side else None
            self._use_lattice = self._lattice_side is not None and spacing >= rd
            if self._use_lattice:
                self._lattice_jitter = (spacing - rd) / 2.0
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
        if self.periodic:
            return np.stack([self._periodic_cloud() for _ in range(batch_size)], axis=0)
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

    def _periodic_cloud(self):
        """Valid cloud on the unit torus: rd holds under minimum-image.

        Dense regime (square lattice fits and satisfies rd): randomly
        translated square lattice with a jitter capped so rd still holds —
        reaches packings dart throwing cannot (jams ~60% of max).
        Sparse regime: dart throwing.
        """
        if self.dim == 2 and getattr(self, '_use_lattice', False):
            n = self._lattice_side
            g = (np.arange(n) + 0.5) / n
            pts = np.stack(np.meshgrid(g, g, indexing='ij'), -1).reshape(-1, 2)
            pts = pts + self._rng.uniform(-self._lattice_jitter, self._lattice_jitter,
                                          size=pts.shape)
            pts = pts + self._rng.uniform(size=(1, 2))          # random torus shift
            return np.mod(pts, 1.0)
        pts = np.empty((0, self.dim))
        for _ in range(200):
            for cand in self._rng.uniform(size=(4 * self.cardinality, self.dim)):
                if len(pts) == self.cardinality:
                    return pts
                if len(pts):
                    d = pts - cand
                    d -= np.round(d)                      # minimum image, box = 1
                    if (np.sqrt((d * d).sum(-1)) < self.rd).any():
                        continue
                pts = np.vstack([pts, cand])
            if len(pts) == self.cardinality:
                return pts
        raise RuntimeError(
            f"periodic Poisson sampling stalled at {len(pts)}/{self.cardinality} "
            f"points (rd={self.rd}, dim={self.dim}) — density too high")

    def noise_sample(self, sample):
        """Add per-cloud Gaussian noise (sigma drawn uniformly per cloud)."""
        sigmas = self._rng.uniform(self.noise_scale_min, self.noise_scale_max,
                                   size=sample.shape[0])
        noise  = self._rng.normal(0.0, 1.0, size=sample.shape) * sigmas[:, None, None]
        noisy  = sample + noise
        return np.mod(noisy, 1.0) if self.periodic else noisy
