"""
apply_single_cloud.py
----------------------
Apply model9 (n100_p050 checkpoint) to a single bounded point cloud whose N
matches N_train exactly — no tiling / PBC ghost buffer needed, just the
scale -> make_invariant -> model -> revert -> unscale round trip used by
trainer.py's eval path.

Usage
-----
    .venv\\Scripts\\python.exe inference/apply_single_cloud.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import load_model_config
from data.data_processor import DataProcessor
from models.fixed_rd.model9 import CorrectorModel

CHECKPOINT   = 'training_artifacts/train_run_2026-05-28_14-53-45/model_final.pt'
MODEL_CONFIG = 'configs/model_configs/model_config_9_n100_p050.yaml'
RD_TRAIN     = 0.076

RD_TEST  = 1.0     # your test case's minimum distance, in [0,10]-domain units
DEVICE   = 'cpu'


def apply(points: np.ndarray, k: int = 1) -> np.ndarray:
    """points: (N, 2) in [0, 10]^2, rd = RD_TEST. Returns corrected (N, 2)."""
    model_cfg = load_model_config(MODEL_CONFIG)
    model = CorrectorModel(model_cfg, input_dim=2, initialization='xavier_uniform')
    model.load_state_dict(torch.load(CHECKPOINT, map_location='cpu'))
    model.to(DEVICE).eval()

    processor = DataProcessor()
    scale     = RD_TRAIN / RD_TEST
    rd_t      = torch.tensor(RD_TRAIN, dtype=torch.float32, device=DEVICE)

    pts = points.astype(np.float32).copy()
    for _ in range(k):
        pts_s = pts * scale                          # rescale to training rd
        batch = pts_s[None]                           # (1, N, 2)

        processed, _, revert = processor.make_invariant(batch)
        x = torch.tensor(processed, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            displacement = model(x, rd=rd_t)           # (1, N, 2), invariant space

        corrected_s = revert((x + displacement).cpu().numpy())  # (1, N, 2), scaled space
        pts = corrected_s[0] / scale                   # back to original [0,10] scale

    return pts


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 10, size=(100, 2)).astype(np.float32)

    corrected = apply(points, k=3)
    print(f'points:    {points.shape}')
    print(f'corrected: {corrected.shape}')
