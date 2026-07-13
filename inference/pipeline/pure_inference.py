"""
pipeline/pure_inference.py
---------------------------
Bare model9 round trip for a single bounded point cloud whose N matches
N_train exactly — no tiling, no PBC ghost buffer, just
scale -> invariant -> model -> revert -> unscale.
"""
import sys
import importlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.config import load_model_config
from data.data_processor import compute_invariant, revert_invariant
from .base import Corrector
from .scaling import compute_scale


class PureInference(Corrector):
    """
    Applies model9 directly to a single bounded cloud (N == N_train, no
    tiling/PBC — see module docstring). points : (N, 2) in any consistent
    coordinate frame, at the constructor's rd_test.
    """

    def __init__(self, checkpoint: str, model_config: str,
                 rd_train: float, rd_test: float, device: str = 'cpu'):
        self.device = torch.device(device)
        self.scale  = compute_scale(rd_train, rd_test)
        self.rd_t   = torch.tensor(rd_train, dtype=torch.float32, device=self.device)

        model_cfg = load_model_config(model_config)
        m = importlib.import_module('models.fixed_rd.model9')
        self.model = m.CorrectorModel(model_cfg, input_dim=2, initialization='xavier_uniform')
        self.model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        self.model.to(self.device).eval()

    def apply(self, points: np.ndarray, k: int = 1) -> np.ndarray:
        pts = points.astype(np.float32)
        for _ in range(k):
            pts_s = (pts * self.scale)[None]   # (1, N, 2)
            x_inv, mean, eigvecs = compute_invariant(pts_s)
            x_t = torch.tensor(x_inv, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                disp = self.model(x_t, rd=self.rd_t).cpu().numpy()
            corrected_s = revert_invariant(x_inv + disp, mean, eigvecs)
            pts = corrected_s[0] / self.scale
        return pts


if __name__ == '__main__':
    inference = PureInference(
        checkpoint   = 'training_artifacts/train_run_2026-05-28_14-53-45/model_final.pt',
        model_config = 'configs/model_configs/model_config_9_n100_p050.yaml',
        rd_train     = 0.076,
        rd_test      = 1.0,   # your test case's minimum distance, in [0,10]-domain units
        device       = 'cpu',
    )

    rng = np.random.default_rng(0)
    points = rng.uniform(0, 10, size=(100, 2)).astype(np.float32)
    corrected = inference.apply(points, k=3)
    print(f'points:    {points.shape}')
    print(f'corrected: {corrected.shape}')
