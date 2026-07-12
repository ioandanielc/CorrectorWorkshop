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
from utils.config import load_model_config
from data.data_processor import compute_invariant, revert_invariant
from .tiling import TilingConfig
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
        self.cfg    = cfg
        self.device = torch.device(cfg.device)
        self.scale  = compute_scale(cfg.rd_train, cfg.rd_test)
        self.tiling = TilingConfig(cfg.grid_size, cfg.ghost_factor, cfg.domain)
        self.rd_t   = torch.tensor(cfg.rd_train, dtype=torch.float32,
                                   device=self.device)
        self._precompute_tile_geometry()
        self._load_model()

    def _precompute_tile_geometry(self):
        """Precompute tile boundaries and per-image offset list (constant for a given config)."""
        G       = self.tiling.grid_size
        c       = self.tiling.cell_size
        dom     = self.cfg.domain
        ghost_w = self.tiling.ghost_width(self.cfg.rd_test)   # emits warning once if < 1.0

        tile_ids = np.arange(G * G)
        ti = (tile_ids // G).astype(np.float32)
        tj = (tile_ids  % G).astype(np.float32)
        self._tile_lo  = np.stack([ti * c, tj * c], axis=1)   # (n_tiles, 2)
        self._tile_hi  = self._tile_lo + np.float32(c)
        self._ext_lo   = self._tile_lo - np.float32(ghost_w)
        self._ext_hi   = self._tile_hi + np.float32(ghost_w)
        self._G        = G
        self._cell     = np.float32(c)
        self._ghost_w  = np.float32(ghost_w)
        self._dom      = np.float32(dom)
        # 9 PBC image offsets; index 4 = identity (dx=0, dy=0)
        self._img_offsets = [(np.float32(dx * dom), np.float32(dy * dom))
                             for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
        # Flat arrays for vectorised Phase 1a
        self._ox_arr = np.array([o[0] for o in self._img_offsets], dtype=np.float32)  # (9,)
        self._oy_arr = np.array([o[1] for o in self._img_offsets], dtype=np.float32)
        # Combo validity masks (4 combos: [dti=0/dtj=0, dti=0/dtj=1, dti=1/dtj=0, dti=1/dtj=1])
        # True where the combo does NOT need a diff-tile check (i.e. dti/dtj == 0)
        self._dti_no_check = np.array([True, True, False, False], dtype=bool)[:, None, None]
        self._dtj_no_check = np.array([True, False, True, False], dtype=bool)[:, None, None]

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

    def _phase1a(self, pts: np.ndarray):
        """
        Vectorised ghost-tile membership for all 9 PBC images × 4 (dti,dtj) combos at once.

        No deduplication is needed: ghost_width << domain guarantees that no
        (tile, particle) pair is reachable from more than one PBC image.

        Returns
        -------
        t_idx      : (M,) int32   tile index for each pair
        p_idx      : (M,) int32   particle index for each pair
        pos_pairs  : (M, 2) f32   PBC-shifted particle position in [extended tile]
        core_pairs : (M,) bool    True iff pair is the particle's own tile (identity image)
        """
        G       = self._G
        cell    = self._cell
        ghost_w = self._ghost_w
        ox      = self._ox_arr   # (9,)
        oy      = self._oy_arr   # (9,)

        # (9, N) shifted positions for all PBC images simultaneously
        sx = pts[:, 0][None, :] + ox[:, None]   # (9, N)
        sy = pts[:, 1][None, :] + oy[:, None]

        ti_lo_f = (sx - ghost_w) / cell
        ti_hi_f = (sx + ghost_w) / cell
        tj_lo_f = (sy - ghost_w) / cell
        tj_hi_f = (sy + ghost_w) / cell

        valid = (ti_hi_f >= 0) & (ti_lo_f < G) & (tj_hi_f >= 0) & (tj_lo_f < G)  # (9, N)

        ti_lo = np.clip(np.floor(ti_lo_f).astype(np.int32), 0, G - 1)  # (9, N)
        ti_hi = np.clip(np.floor(ti_hi_f).astype(np.int32), 0, G - 1)
        tj_lo = np.clip(np.floor(tj_lo_f).astype(np.int32), 0, G - 1)
        tj_hi = np.clip(np.floor(tj_hi_f).astype(np.int32), 0, G - 1)

        # Stack 4 (dti, dtj) combos → (4, 9, N)
        # combo 0: (0,0)=lo/lo  1: (0,1)=lo/hi  2: (1,0)=hi/lo  3: (1,1)=hi/hi
        ti_c = np.stack([ti_lo, ti_lo, ti_hi, ti_hi])   # (4, 9, N)
        tj_c = np.stack([tj_lo, tj_hi, tj_lo, tj_hi])

        # Combo validity: dti=1 combos (2,3) need ti_hi>ti_lo; dtj=1 combos (1,3) need tj_hi>tj_lo
        combo_valid = (valid[None]
                       & (self._dti_no_check | (ti_hi[None] > ti_lo[None]))
                       & (self._dtj_no_check | (tj_hi[None] > tj_lo[None])))   # (4, 9, N)

        c_idx, img_idx, p_idx = np.where(combo_valid)
        if c_idx.size == 0:
            empty2 = np.empty((0, 2), dtype=np.float32)
            return (np.empty(0, np.int32), np.empty(0, np.int32),
                    empty2, np.empty(0, bool))

        t_idx     = (ti_c[c_idx, img_idx, p_idx] * G
                     + tj_c[c_idx, img_idx, p_idx]).astype(np.int32)
        pos_pairs = np.stack([pts[p_idx, 0] + ox[img_idx],
                              pts[p_idx, 1] + oy[img_idx]], axis=1)

        # Core: identity image (img_idx==4) AND particle is in its own tile
        own_ti = np.floor(pts[:, 0] / cell).astype(np.int32)
        own_tj = np.floor(pts[:, 1] / cell).astype(np.int32)
        core_pairs = ((img_idx == 4)
                      & (ti_c[c_idx, img_idx, p_idx] == own_ti[p_idx])
                      & (tj_c[c_idx, img_idx, p_idx] == own_tj[p_idx]))

        return t_idx, p_idx.astype(np.int32), pos_pairs, core_pairs

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
        n_tiles = self._tile_lo.shape[0]
        PAD     = np.float32(1e3)

        for _ in range(k):
            N = len(pts)

            # ── Phase 1a: vectorised ghost-tile membership ────────────────────
            t_idx, p_idx, pos_pairs, core_pairs = self._phase1a(pts)
            if t_idx.size == 0:
                continue

            # ── Phase 1b: slot assignment + padded batch (CPU) ────────────────
            sort_ord   = np.argsort(t_idx, kind='stable')
            t_sorted   = t_idx[sort_ord]
            tile_start = np.searchsorted(t_sorted, np.arange(n_tiles))
            slot_sorted = np.arange(sort_ord.size) - tile_start[t_sorted]
            slot_idx    = np.empty(sort_ord.size, dtype=np.int32)
            slot_idx[sort_ord] = slot_sorted

            max_N = int(np.bincount(t_idx, minlength=n_tiles).max())
            pad_x = PAD + np.arange(max_N, dtype=np.float32) * 0.2
            batch_pts_s    = np.empty((n_tiles, max_N, 2), dtype=np.float32)
            batch_pts_s[:, :, 0] = pad_x
            batch_pts_s[:, :, 1] = PAD
            batch_is_core  = np.zeros((n_tiles, max_N), dtype=bool)
            batch_orig_idx = np.zeros((n_tiles, max_N), dtype=np.int32)
            real_mask_np   = np.zeros((n_tiles, max_N), dtype=np.float32)

            pts_s = pos_pairs * self.scale
            batch_pts_s   [t_idx, slot_idx] = pts_s
            batch_is_core [t_idx, slot_idx] = core_pairs
            batch_orig_idx[t_idx, slot_idx] = p_idx
            real_mask_np  [t_idx, slot_idx] = 1.0

            # ── Phase 1b cont: center + PCA-rotate each tile (same as DataProcessor.make_invariant) ──
            x_inv, mean_batch, eigvecs = compute_invariant(batch_pts_s, mask=real_mask_np)

            # ── Phase 2: H2D x_inv only → GPU forward → D2H disp ────────────
            x_t = torch.tensor(x_inv, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                disp = self.model(x_t, rd=self.rd_t).cpu().numpy()       # (T, max_N, 2)

            # ── Phase 3: revert to scaled space, take the displacement, unscale ──
            corrected_s = revert_invariant(x_inv + disp, mean_batch, eigvecs)  # (T, max_N, 2)
            disp_orig   = (corrected_s - batch_pts_s) / self.scale             # (T, max_N, 2)

            # Each particle is core in exactly one tile → orig_indices is a permutation
            # of [0,N-1], so direct assignment beats np.add.at
            core_t, core_s = np.where(batch_is_core)
            orig_indices   = batch_orig_idx[core_t, core_s]              # (N,) permutation
            disps_core     = disp_orig[core_t, core_s]                  # (N, 2)

            displacement             = np.empty_like(pts)
            displacement[orig_indices] = disps_core
            pts = (pts + displacement) % self.cfg.domain

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
