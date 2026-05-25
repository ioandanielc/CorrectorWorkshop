import numpy as np


class DataProcessor:
    def make_translational_invariant(self, sample):
        centroids = sample.mean(axis=1, keepdims=True)  # (batch_size, 1, dim)
        transformed = sample - centroids

        def revert(s):
            return s + centroids

        return transformed, {"centroids": centroids}, revert

    def make_rotational_invariant(self, sample):
        aligned = []
        eigvecs_list = []
        for cloud in sample:
            cov = np.cov(cloud, rowvar=False)
            _, eigvecs = np.linalg.eigh(cov)
            eigvecs = eigvecs[:, ::-1]
            aligned.append(cloud @ eigvecs)
            eigvecs_list.append(eigvecs)
        transformed = np.stack(aligned)
        eigvecs_arr = np.stack(eigvecs_list)  # (batch_size, dim, dim)

        def revert(s):
            reverted = []
            for cloud, evecs in zip(s, eigvecs_arr):
                reverted.append(cloud @ evecs.T)
            return np.stack(reverted)

        return transformed, {"eigvecs": eigvecs_arr}, revert

    def make_invariant(self, sample):
        sample, t_params, revert_translation = self.make_translational_invariant(sample)
        sample, r_params, revert_rotation = self.make_rotational_invariant(sample)

        def revert(s):
            s = revert_rotation(s)
            s = revert_translation(s)
            return s

        return sample, {**t_params, **r_params}, revert
