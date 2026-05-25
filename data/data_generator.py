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



if __name__ == "__main__":
    gen = PoissonDiskDataset(dim=3, rd=0.05, cardinality=50, seed=42,
                             noise_scale_min=0.001, noise_scale_max=0.01)

    sample = gen.generate_sample(batch_size=1)  # (1, 50, 2)
    sample = gen.noise_sample(sample)  # (1, 50, 2)
    sample = gen.make_invariant(sample)  # (1, 50, 2)
    plot_poisson_disk(sample[0], rd=0.05)