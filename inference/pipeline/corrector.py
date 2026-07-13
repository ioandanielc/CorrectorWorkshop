"""
pipeline/corrector.py
---------------------
Full inference pipeline: tiling + ghost buffer + coordinate scaling + model.

Usage
-----
    from inference.pipeline.corrector import GridCorrector, GridCorrectorConfig

    cfg = GridCorrectorConfig.from_yaml('inference/configs/grid_6x6.yaml')
    corrector = GridCorrector(cfg)

    pts_corrected = corrector.apply(pts, k=3)   # (N, 2) -> (N, 2)
"""
import sys
import importlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.config import load_model_config
from data.data_processor import compute_invariant, revert_invariant
from .base import Corrector
from .tiling import TilingConfig, iter_tiles
from .pbc import build_ghost_tile
from .scaling import compute_scale


@dataclass
class GridCorrectorConfig:
    # model
    checkpoint:  str
    model_config: str
    rd_train:    float

    # data
    rd_test:     float
    device:      str   = 'cpu'

    # tiling
    n_cells:      int   = 6
    ghost_factor: float = 1.0   # ghost_width = ghost_factor * cell_size

    @classmethod
    def from_yaml(cls, path: str) -> 'GridCorrectorConfig':
        """Load a corrector config. 'tiling' is an inline {n_cells, ghost_factor} dict."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        m   = raw['model']
        d   = raw['data']
        exp = raw.get('experiment', {})

        t_raw        = raw.get('tiling', {})
        n_cells      = int(t_raw.get('n_cells', 6))
        ghost_factor = float(t_raw.get('ghost_factor', 1.0))

        return cls(
            checkpoint   = m['checkpoint'],
            model_config = m['config'],
            rd_train     = float(m['rd_train']),
            rd_test      = float(d['rd_test']),
            device       = exp.get('device', 'cpu'),
            n_cells      = n_cells,
            ghost_factor = ghost_factor,
        )


class GridCorrector(Corrector):
    """
    Applies the trained Poisson-disk corrector to large PBC point clouds
    via tiled inference with ghost buffer and coordinate scaling.

    The point cloud's own domain (box size) is inferred fresh on every
    apply() call, not configured up front: the cloud is centered on its
    own centroid and the domain is taken as its largest axis extent, so
    apply() works regardless of what coordinate frame the input is in.
    """

    def __init__(self, cfg: GridCorrectorConfig):
        self.cfg    = cfg
        self.device = torch.device(cfg.device)
        self.scale  = compute_scale(cfg.rd_train, cfg.rd_test)
        self.rd_t   = torch.tensor(cfg.rd_train, dtype=torch.float32, device=self.device)
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

    def _infer_frame(self, points: np.ndarray):
        """Center on the centroid; domain = largest axis extent of the centered cloud."""
        centroid = points.mean(axis=0)
        centered = points - centroid
        domain = float(max(centered[:, 0].max() - centered[:, 0].min(),
                            centered[:, 1].max() - centered[:, 1].min()))
        return centroid, domain

    def apply(self, points: np.ndarray, k: int = 1) -> np.ndarray:
        """
        Apply k passes of the tiled PBC corrector.

        The cloud's centroid and domain (see class docstring) are computed
        once from `points` here and held fixed across all k passes.

        Parameters
        ----------
        points : (N, 2)  positions in any consistent coordinate frame
        k      : int     number of correction passes

        Returns
        -------
        (N, 2)  corrected positions, in the same coordinate frame as the input
        """
        points = points.astype(np.float32)
        centroid, domain = self._infer_frame(points)

        tiling  = TilingConfig(self.cfg.n_cells, self.cfg.ghost_factor, domain)
        ghost_w = tiling.ghost_width(self.cfg.rd_test)   # warns once if too narrow

        pts = (points - centroid + domain / 2).astype(np.float32) % domain   # shift into [0, domain)
        PAD = np.float32(1e3)   # sentinel: far outside the domain, no violation vs real points

        for _ in range(k):
            # ── build every tile's core + ghost particles ──
            tiles = [build_ghost_tile(pts, lo, hi, ghost_w, domain)
                     for lo, hi in iter_tiles(tiling)]
            n_tiles = len(tiles)
            max_N   = max(len(pts_ext) for pts_ext, _, _ in tiles)

            # ── pack into one padded batch, scaled to the model's training rd ──
            batch_pts_s    = np.full((n_tiles, max_N, 2), PAD, dtype=np.float32)
            batch_is_core  = np.zeros((n_tiles, max_N), dtype=bool)
            batch_orig_idx = np.zeros((n_tiles, max_N), dtype=np.int32)
            real_mask      = np.zeros((n_tiles, max_N), dtype=np.float32)
            for i, (pts_ext, is_core, orig_idx) in enumerate(tiles):
                m = len(pts_ext)
                batch_pts_s[i, :m]    = pts_ext * self.scale
                batch_is_core[i, :m]  = is_core
                batch_orig_idx[i, :m] = orig_idx
                real_mask[i, :m]      = 1.0

            # ── center + PCA-rotate each tile, run the model once for all tiles ──
            x_inv, mean_batch, eigvecs = compute_invariant(batch_pts_s, mask=real_mask)
            x_t = torch.tensor(x_inv, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                disp = self.model(x_t, rd=self.rd_t).cpu().numpy()

            # ── revert to scaled space, take the displacement, unscale ──
            corrected_s = revert_invariant(x_inv + disp, mean_batch, eigvecs)
            disp_orig   = (corrected_s - batch_pts_s) / self.scale

            # ── each particle keeps the displacement from its one core (non-ghost) tile ──
            core_t, core_s = np.where(batch_is_core)
            orig_indices    = batch_orig_idx[core_t, core_s]   # permutation of [0, N)
            displacement    = np.empty_like(pts)
            displacement[orig_indices] = disp_orig[core_t, core_s]
            pts = (pts + displacement) % domain

        return pts - domain / 2 + centroid

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
        existing corrector is reused without modification. apply()
        infers its own frame from whatever it's given, so no PBC wrap
        is needed here between passes.

        Parameters
        ----------
        points         : (N, 2) positions in any consistent coordinate frame
        shift_fraction : fraction of cell_size to shift (default 0.5)

        Returns
        -------
        (N, 2) corrected positions after both passes
        """
        _, domain = self._infer_frame(points)
        shift = shift_fraction * (domain / self.cfg.n_cells)   # scalar offset in both axes

        after_p1 = self.apply(points, k=1)
        pts_out  = self.apply(after_p1 + shift, k=1)
        return pts_out - shift
