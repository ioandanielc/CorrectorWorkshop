"""
pipeline/corrector.py
---------------------
Full inference pipeline: tiling + ghost buffer + coordinate scaling + model.

Usage
-----
    from inference.pipeline.corrector import Corrector, CorrectorConfig

    cfg = CorrectorConfig.from_yaml('inference/configs/grid_6x6.yaml')
    corrector = Corrector(cfg)

    pts_corrected = corrector.apply(pts, k=3)   # (N, 2) -> (N, 2)
"""
import sys
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data.data_processor import DataProcessor
from utils.config import load_model_config
from .pbc import build_ghost_tile
from .tiling import TilingConfig, iter_tiles
from .scaling import compute_scale


@dataclass
class CorrectorConfig:
    # model
    checkpoint:  str
    model_config: str
    rd_train:    float

    # data / domain
    rd_test:     float
    domain:      float = 1.0
    device:      str   = 'cpu'

    # tiling
    grid_size:    int   = 6
    ghost_factor: float = 1.0   # ghost_width = ghost_factor * rd_test

    # experiment
    k_values:     List[int] = field(default_factory=lambda: [1, 2, 3, 5])
    stride:       int        = 100

    @classmethod
    def from_yaml(cls, path: str) -> 'CorrectorConfig':
        """
        Load experiment config. The 'tiling' key can be either:
          - a path string to a grid config  (e.g. "inference/configs/grids/grid_6x6.yaml")
          - an inline dict                  (e.g. {grid_size: 6, ghost_factor: 1.0})
        """
        with open(path) as f:
            raw = yaml.safe_load(f)
        m   = raw['model']
        d   = raw['data']
        exp = raw.get('experiment', {})

        # resolve tiling — file reference or inline
        t_raw = raw.get('tiling', {})
        if isinstance(t_raw, str):
            grid_cfg = TilingConfig.from_yaml(t_raw)
            grid_size    = grid_cfg.grid_size
            ghost_factor = grid_cfg.ghost_factor
        else:
            grid_size    = int(t_raw.get('grid_size', 6))
            ghost_factor = float(t_raw.get('ghost_factor', 1.0))

        return cls(
            checkpoint   = m['checkpoint'],
            model_config = m['config'],
            rd_train     = float(m['rd_train']),
            rd_test      = float(d['rd_test']),
            domain       = float(d.get('domain', 1.0)),
            device       = exp.get('device', 'cpu'),
            grid_size    = grid_size,
            ghost_factor = ghost_factor,
            k_values     = list(exp.get('k_values', [1, 2, 3, 5])),
            stride       = int(exp.get('stride', 100)),
        )


class Corrector:
    """
    Applies the trained Poisson-disk corrector to large PBC point clouds
    via tiled inference with ghost buffer and coordinate scaling.
    """

    def __init__(self, cfg: CorrectorConfig):
        self.cfg       = cfg
        self.device    = torch.device(cfg.device)
        self.scale     = compute_scale(cfg.rd_train, cfg.rd_test)
        self.tiling    = TilingConfig(cfg.grid_size, cfg.ghost_factor, cfg.domain)
        self.processor = DataProcessor()
        self.rd_t      = torch.tensor(cfg.rd_train, dtype=torch.float32,
                                      device=self.device)
        self._load_model()

    def _load_model(self):
        model_cfg = load_model_config(self.cfg.model_config)
        m = importlib.import_module('models.fixed_rd.model9')
        self.model = m.CorrectorModel(
            model_cfg, input_dim=2, initialization='xavier_uniform'
        )
        state = torch.load(self.cfg.checkpoint, map_location='cpu')
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def apply(self, points: np.ndarray, k: int = 1) -> np.ndarray:
        """
        Apply k passes of the tiled PBC corrector.

        Parameters
        ----------
        points : (N, 2)  positions in [0, domain)^2
        k      : int     number of correction passes

        Returns
        -------
        (N, 2)  corrected positions, PBC-wrapped to [0, domain)^2
        """
        pts = points.astype(np.float32).copy()
        ghost_w = self.tiling.ghost_width(self.cfg.rd_test)

        for _ in range(k):
            displacements = np.zeros_like(pts)
            counts        = np.zeros(len(pts), dtype=int)

            for tile_lo, tile_hi in iter_tiles(self.tiling):
                ext, is_core, orig_idx = build_ghost_tile(
                    pts, tile_lo, tile_hi, ghost_w, self.cfg.domain
                )
                if is_core.sum() == 0:
                    continue

                # scale -> invariant -> model -> unscale
                pts_s         = ext * self.scale
                x_inv, _, rev = self.processor.make_invariant(pts_s[None])
                x_t           = torch.tensor(x_inv, dtype=torch.float32,
                                             device=self.device)
                with torch.no_grad():
                    disp_inv = self.model(x_t, rd=self.rd_t).cpu().numpy()[0]
                corrected = rev(x_inv + disp_inv)[0]
                disp_orig = (corrected - pts_s) / self.scale

                for idx_k in range(len(ext)):
                    if is_core[idx_k]:
                        displacements[orig_idx[idx_k]] += disp_orig[idx_k]
                        counts[orig_idx[idx_k]] += 1

            counts = np.maximum(counts, 1)
            pts = (pts + displacements / counts[:, None]) % self.cfg.domain

        return pts
