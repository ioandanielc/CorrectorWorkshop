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
        # relative paths are resolved: first against CWD, then against the
        # experiment config's own directory, then two levels up (project root)
        t_raw = raw.get('tiling', {})
        if isinstance(t_raw, str):
            tiling_path = Path(t_raw)
            if not tiling_path.is_absolute():
                config_dir  = Path(path).resolve().parent
                for base in [Path('.'), config_dir, config_dir.parent,
                             config_dir.parent.parent]:
                    candidate = (base / tiling_path).resolve()
                    if candidate.exists():
                        tiling_path = candidate
                        break
            grid_cfg = TilingConfig.from_yaml(str(tiling_path))
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
        pts     = points.astype(np.float32).copy()
        ghost_w = self.tiling.ghost_width(self.cfg.rd_test)
        # Padding sentinel: distance to any real particle >> rd_train -> violation = 0
        # -> weight = 0 -> real particle aggregations are unaffected by padding.
        PAD = np.float32(1e3)

        for _ in range(k):
            # ── Phase 1: ghost tiles + make_invariant (per tile, cheap) ───────
            tiles = []
            for tile_lo, tile_hi in iter_tiles(self.tiling):
                ext, is_core, orig_idx = build_ghost_tile(
                    pts, tile_lo, tile_hi, ghost_w, self.cfg.domain)
                if is_core.sum() == 0:
                    continue
                pts_s         = ext * self.scale
                x_inv, _, rev = self.processor.make_invariant(pts_s[None])
                tiles.append((x_inv, pts_s, is_core, orig_idx, rev))

            if not tiles:
                continue

            # ── Phase 2: pad to max_N, single batched forward pass ────────────
            max_N   = max(t[0].shape[1] for t in tiles)
            n_tiles = len(tiles)
            # Spread padded positions so they don't violate each other (dist > rd_train).
            pad_offsets = np.arange(max_N, dtype=np.float32) * 0.2   # 0.2 >> rd_train
            batch_x = np.empty((n_tiles, max_N, 2), dtype=np.float32)
            batch_x[:, :, 0] = PAD + pad_offsets          # broadcast over tiles
            batch_x[:, :, 1] = PAD
            for i, (x_inv, *_) in enumerate(tiles):
                n = x_inv.shape[1]
                batch_x[i, :n] = x_inv[0]

            x_t = torch.tensor(batch_x, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                disp_batch = self.model(x_t, rd=self.rd_t).cpu().numpy()

            # ── Phase 3: unpad, rev, accumulate (vectorised) ──────────────────
            displacements = np.zeros_like(pts)
            counts        = np.zeros(len(pts), dtype=np.int32)

            for i, (x_inv, pts_s, is_core, orig_idx, rev) in enumerate(tiles):
                n         = x_inv.shape[1]
                disp_inv  = disp_batch[i, :n]          # (n, 2)
                corrected = rev(x_inv + disp_inv)[0]   # (n, 2)
                disp_orig = (corrected - pts_s) / self.scale
                core      = is_core.astype(bool)
                ci        = orig_idx[core]
                np.add.at(displacements, ci, disp_orig[core])
                np.add.at(counts,        ci, 1)

            counts = np.maximum(counts, 1)
            pts    = (pts + displacements / counts[:, None]) % self.cfg.domain

        return pts

    def apply_shifted_grid(
        self,
        points:         np.ndarray,
        shift_fraction: float = 0.5,
    ) -> np.ndarray:
        """
        Two-pass shifted-grid strategy:
          Pass 1 — K=1 on the standard grid   (origin at 0, 0)
          Pass 2 — K=1 on a shifted grid       (origin at shift, shift)

        The shift equals shift_fraction × cell_size, defaulting to
        cell_size/2 (half a cell).  This places Pass-2 tile boundaries
        at the centres of Pass-1 tiles, so particles that were near a
        Pass-1 boundary are now well inside a Pass-2 tile.

        Mechanically, shifting the grid is equivalent to shifting the
        particles before inference and unshifting afterwards — the
        existing corrector is reused without modification.

        Parameters
        ----------
        points         : (N, 2) positions in [0, domain)^2
        shift_fraction : fraction of cell_size to shift (default 0.5)

        Returns
        -------
        (N, 2) corrected positions after both passes
        """
        shift = shift_fraction * self.tiling.cell_size  # scalar offset in both axes

        # Pass 1: standard K=1
        after_p1 = self.apply(points, k=1)

        # Pass 2: shift particles into the shifted-grid frame, correct, unshift
        pts_in  = (after_p1 + shift) % self.cfg.domain
        pts_out = self.apply(pts_in, k=1)
        return (pts_out - shift) % self.cfg.domain
