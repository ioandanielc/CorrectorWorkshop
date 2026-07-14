"""
src/inference/correctors/pure/pure_inference.py
---------------------------
Bare model9 round trip for a single bounded point cloud whose N matches
N_train exactly — no tiling, no PBC ghost buffer, just
scale -> invariant -> model -> revert -> unscale.
"""
import sys
import importlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[3]))
from utils.config import load_model_config
from models.architectures.model9.invariance import compute_invariant, revert_invariant
from ..base import Corrector
from ..common.scaling import compute_scale


@dataclass
class PureInference2DConfig:
    # data — the test case's minimum distance, in its own domain units
    rd_test:      float

    # model — defaults to the primary production checkpoint
    checkpoint:   str   = 'src/models/weights/model9_n100_p050.pt'
    model_config: str   = 'src/configs/training/model/model_config_9_n100_p050.yaml'
    rd_train:     float = 0.076

    device:       str   = 'cpu'


class PureInference2D(Corrector):
    """
    Applies model9 directly to a single bounded cloud (N == N_train, no
    tiling/PBC — see module docstring). points : (N, 2) in any consistent
    coordinate frame, at cfg.rd_test.
    """

    def __init__(self, cfg: PureInference2DConfig):
        self.cfg    = cfg
        self.device = torch.device(cfg.device)
        self.scale  = compute_scale(cfg.rd_train, cfg.rd_test)
        self.rd_t   = torch.tensor(cfg.rd_train, dtype=torch.float32, device=self.device)

        model_cfg = load_model_config(cfg.model_config)
        m = importlib.import_module('models.architectures.model9.model9')
        self.model = m.CorrectorModel(model_cfg, input_dim=2, initialization='xavier_uniform')
        self.model.load_state_dict(torch.load(cfg.checkpoint, map_location='cpu'))
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
    # rd_test = 1.0: minimum distance for a demo cloud in [0, 10]^2
    inference = PureInference2D(PureInference2DConfig(rd_test=1.0))

    rng = np.random.default_rng(0)
    points = rng.uniform(0, 10, size=(100, 2)).astype(np.float32)
    corrected = inference.apply(points, k=3)
    print(f'points:    {points.shape}')
    print(f'corrected: {corrected.shape}')
