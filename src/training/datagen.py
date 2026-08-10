import numpy as np
from scipy.stats import qmc


class PoissonDiskDataset:
    """Synthetic Poisson disk dataset with online (per-batch) generation.

    Generates clouds of `cardinality` points with minimum pairwise distance `rd`
    using scipy's PoissonDisk sampler, then adds Gaussian noise to create
    constraint violations. `periodic=True` applies periodic boundary conditions
    (rd enforced under minimum-image on the unit torus) — see __init__.
    """

    def __init__(self, dim, cardinality, rd, seed,
                 noise_scale_min, noise_scale_max, periodic=False):
        self.dim             = dim
        self.cardinality     = cardinality
        self.rd              = rd
        self.seed            = seed
        self.noise_scale_min = noise_scale_min
        self.noise_scale_max = noise_scale_max
        # periodic=True: periodic boundary conditions — rd holds under
        # minimum-image on the unit torus, noise wraps mod 1. model12 trains in
        # this fixed frame — no invariant/rotation transform. Clean-cloud
        # construction: see _periodic_cloud.
        self.periodic        = periodic
        if periodic:
            n_side = round(cardinality ** 0.5)
            self._lattice_side = n_side if n_side * n_side == cardinality else None
            spacing = 1.0 / n_side if self._lattice_side else None
            self._use_lattice = self._lattice_side is not None and spacing >= rd
            if self._use_lattice:
                self._lattice_jitter = (spacing - rd) / 2.0
        # One generator drives all randomness: a given seed fixes the *sequence*
        # of batches, but every call yields fresh clouds and noise (not one
        # repeated batch).
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
        # Periodic (torus) clouds: minimum-image sampler, one per batch item.
        if self.periodic:
            return np.stack([self._periodic_cloud() for _ in range(batch_size)], axis=0)

        # Non-periodic: scipy's PoissonDisk in the open box, one cloud at a time.
        clouds = []
        for _ in range(batch_size):
            # Retry with a fresh seed until the sampler places the full count.
            for _ in range(20):   # resample budget per cloud
                seed  = self._cloud_seeds(1)[0]
                cloud = qmc.PoissonDisk(
                    d=self.dim, radius=self.rd, seed=seed).random(self.cardinality)
                if len(cloud) == self.cardinality:
                    break
            else:                 # all 20 attempts fell short -> too dense to sample
                raise RuntimeError(
                    f"PoissonDisk placed fewer than {self.cardinality} points at "
                    f"rd={self.rd} (dim={self.dim}) in 20 attempts — density too "
                    f"high for reliable sampling")
            clouds.append(cloud)
        return np.stack(clouds, axis=0)   # (batch_size, cardinality, dim)

    def _periodic_cloud(self):
        """Valid cloud on the unit torus: rd holds under minimum-image.

        Dense regime (square lattice fits and satisfies rd): randomly
        translated square lattice with a jitter capped so rd still holds —
        reaches packings dart throwing cannot (jams ~60% of max).
        Sparse regime: dart throwing.
        """
        # Dense regime: a jittered, randomly shifted n×n lattice.
        if self.dim == 2 and getattr(self, '_use_lattice', False):
            n = self._lattice_side
            g = (np.arange(n) + 0.5) / n                        # n cell-centre coords in [0,1)
            pts = np.stack(np.meshgrid(g, g, indexing='ij'), -1).reshape(-1, 2)   # the n×n grid
            pts = pts + self._rng.uniform(-self._lattice_jitter, self._lattice_jitter,
                                          size=pts.shape)        # per-point jitter (rd still holds)
            pts = pts + self._rng.uniform(size=(1, 2))          # random torus shift
            return np.mod(pts, 1.0)                             # wrap back into the box

        # Sparse regime: dart throwing (rejection sampling under minimum image).
        pts = np.empty((0, self.dim))
        for _ in range(200):                                    # bounded outer refill attempts
            # One sweep of candidate darts: accept those that clear rd from every
            # accepted point, returning as soon as the cloud is full.
            for cand in self._rng.uniform(size=(4 * self.cardinality, self.dim)):
                if len(pts) == self.cardinality:
                    return pts
                if len(pts):
                    d = pts - cand
                    d -= np.round(d)                      # minimum image, box = 1
                    if (np.sqrt((d * d).sum(-1)) < self.rd).any():
                        continue                          # too close -> reject this dart
                pts = np.vstack([pts, cand])              # accept
            if len(pts) == self.cardinality:
                return pts
        raise RuntimeError(
            f"periodic Poisson sampling stalled at {len(pts)}/{self.cardinality} "
            f"points (rd={self.rd}, dim={self.dim}) — density too high")

    def noise_sample(self, sample):
        """Add per-cloud Gaussian noise (sigma drawn uniformly per cloud)."""
        sigmas = self._rng.uniform(self.noise_scale_min, self.noise_scale_max,
                                   size=sample.shape[0])          # one sigma per cloud
        noise  = self._rng.normal(0.0, 1.0, size=sample.shape) * sigmas[:, None, None]  # scale each cloud
        noisy  = sample + noise
        return np.mod(noisy, 1.0) if self.periodic else noisy     # wrap on the torus (periodic only)
