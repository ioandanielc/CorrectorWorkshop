"""
src/inference/correctors/grid/corrector.py
---------------------
Full inference pipeline: tiling + ghost buffer + coordinate scaling + model.
GridCorrector2D and GridCorrector3D share the same dimension-generic body
(_GridCorrectorND); only DIM differs.

Usage
-----
    from inference.correctors.grid.corrector import GridCorrector2D, GridCorrector2DConfig

    cfg = GridCorrector2DConfig.from_yaml('src/configs/experiments/sph_tv/model12_grid.yaml')
    corrector = GridCorrector2D(cfg)

    pts_corrected = corrector.apply(pts, k=3)   # (N, 2) -> (N, 2)
"""
import sys
import importlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parents[3]))
from utils.config import load_model_config
from ..base import Corrector
from ..common.tiling import TilingConfig, iter_tiles
from ..common.pbc import build_ghost_tile
from ..common.scaling import compute_scale


@dataclass
class GridCorrector2DConfig:
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

    # visualization
    enhanced_visualization: bool = False   # save a 3-panel figure per apply() call
    viz_dir:                str  = 'artifacts/inference/misc/enhanced_viz'

    @classmethod
    def from_yaml(cls, path: str) -> 'GridCorrector2DConfig':
        """Load a corrector config. 'tiling' is an inline {n_cells, ghost_factor} dict."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        m   = raw['model']
        d   = raw['data']
        exp = raw.get('experiment', {})

        t_raw        = raw.get('tiling', {})
        n_cells      = int(t_raw.get('n_cells', 6))
        ghost_factor = float(t_raw.get('ghost_factor', 1.0))

        v_raw = raw.get('visualization', {})

        return cls(
            checkpoint   = m['checkpoint'],
            model_config = m['config'],
            rd_train     = float(m['rd_train']),
            rd_test      = float(d['rd_test']),
            device       = exp.get('device', 'cpu'),
            n_cells      = n_cells,
            ghost_factor = ghost_factor,
            enhanced_visualization = bool(v_raw.get('enhanced', False)),
            viz_dir      = v_raw.get('dir', cls.viz_dir),
        )


class GridCorrector3DConfig(GridCorrector2DConfig):
    """Same fields as the 2D config; separate type so 3D defaults can diverge later."""


class _GridCorrectorND(Corrector):
    """
    Applies the trained Poisson-disk corrector to large PBC point clouds
    via tiled inference with ghost buffer and coordinate scaling.
    Dimension-generic body — concrete subclasses set DIM.

    The point cloud's own domain (box size) is inferred fresh on every
    apply() call, not configured up front: the cloud is centered on its
    own centroid and the domain is taken as its largest axis extent, so
    apply() works regardless of what coordinate frame the input is in.
    """

    DIM: int = None   # set by concrete subclasses

    def __init__(self, cfg: GridCorrector2DConfig):
        self.cfg    = cfg
        self.device = torch.device(cfg.device)
        self.scale  = compute_scale(cfg.rd_train, cfg.rd_test)
        self._viz_count = 0   # numbers the per-apply() enhanced-viz figures
        if cfg.enhanced_visualization and self.DIM != 2:
            import warnings
            warnings.warn('enhanced_visualization renders 2D panels only — '
                          'ignored for a 3D corrector.', UserWarning, stacklevel=2)
        self._load_model()

    def _load_model(self):
        model_cfg = load_model_config(self.cfg.model_config)
        module_path = model_cfg['model_file']   # required — no default architecture
        m = importlib.import_module(module_path.replace('/', '.'))
        self.model = m.CorrectorModel(
            model_cfg, input_dim=self.DIM, initialization='xavier_uniform'
        )
        state = torch.load(self.cfg.checkpoint, map_location='cpu')
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        # Box-aware models (model12) take a separate attention radius, distinct
        # from rd_train (constraint rd, which still drives scaling and
        # ghost-width). The ghost buffer already resolves periodicity into flat
        # local coordinates, so box=None is passed to the model at inference
        # regardless of what it trained with. Non-box models (the removed
        # model9 family, which needed the center+PCA invariant frame) are no
        # longer supported on this branch.
        if not getattr(self.model, 'uses_box', False):
            raise TypeError(
                f'{module_path} is not box-aware — this branch serves model12-style '
                f'models only (the model9 family and its invariant frame were removed; '
                f'they live on the main/simplify branches).')
        attention_rd = model_cfg.get('attention_rd', self.cfg.rd_train)
        self.rd_model_t = torch.tensor(attention_rd, dtype=torch.float32, device=self.device)

    def _infer_frame(self, points: np.ndarray):
        """Center on the centroid; domain = largest axis extent of the centered cloud."""
        centroid = points.mean(axis=0)
        centered = points - centroid
        domain = float((centered.max(axis=0) - centered.min(axis=0)).max())
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

        tiling  = TilingConfig(self.cfg.n_cells, self.cfg.ghost_factor, domain, dim=self.DIM)
        ghost_w = tiling.ghost_width(self.cfg.rd_test)   # warns once if too narrow

        pts = (points - centroid + domain / 2).astype(np.float32) % domain   # shift into [0, domain)
        pts_in = pts.copy() if (self.cfg.enhanced_visualization and self.DIM == 2) else None
        PAD = np.float32(1e3)   # sentinel: far outside the domain, no violation vs real points

        for _ in range(k):
            # ── build every tile's core + ghost particles ──
            tiles = [build_ghost_tile(pts, lo, hi, ghost_w, domain)
                     for lo, hi in iter_tiles(tiling)]
            n_tiles = len(tiles)
            max_N   = max(len(pts_ext) for pts_ext, _, _ in tiles)

            # ── pack into one padded batch, scaled to the model's training rd ──
            batch_pts_s    = np.full((n_tiles, max_N, self.DIM), PAD, dtype=np.float32)
            batch_is_core  = np.zeros((n_tiles, max_N), dtype=bool)
            batch_orig_idx = np.zeros((n_tiles, max_N), dtype=np.int32)
            for i, (pts_ext, is_core, orig_idx) in enumerate(tiles):
                m = len(pts_ext)
                batch_pts_s[i, :m]    = pts_ext * self.scale
                batch_is_core[i, :m]  = is_core
                batch_orig_idx[i, :m] = orig_idx

            # ── run the model once over all tiles (model12 is translation-
            #    invariant by construction — no frame transform needed) ──
            x_t = torch.tensor(batch_pts_s, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                disp = self.model(x_t, rd=self.rd_model_t, box=None).cpu().numpy()
            # add-then-subtract kept deliberately: bit-identical to the
            # historical (validated) arithmetic, unlike a bare disp / scale
            corrected_s = batch_pts_s + disp
            disp_orig   = (corrected_s - batch_pts_s) / self.scale

            # ── each particle keeps the displacement from its one core (non-ghost) tile ──
            core_t, core_s = np.where(batch_is_core)
            orig_indices    = batch_orig_idx[core_t, core_s]   # permutation of [0, N)
            displacement    = np.empty_like(pts)
            displacement[orig_indices] = disp_orig[core_t, core_s]
            pts = (pts + displacement) % domain

        if pts_in is not None:
            self._save_enhanced_viz(pts_in, pts, tiling, ghost_w, k)

        return pts - domain / 2 + centroid

    def _save_enhanced_viz(self, pts_before, pts_after, tiling, ghost_w, k):
        """Save the 3-panel per-apply() figure (input | tiling | displacements)."""
        from utils.visualizations.inference_visualizations.enhanced import plot_enhanced_correction
        path = Path(self.cfg.viz_dir) / f'apply_{self._viz_count:04d}.png'
        self._viz_count += 1
        plot_enhanced_correction(
            pts_before, pts_after, tiling, ghost_w,
            rd_test=self.cfg.rd_test, k=k, save_path=str(path),
        )
        print(f'[enhanced viz] saved -> {path}')


class GridCorrector2D(_GridCorrectorND):
    """2D grid corrector: n_cells^2 tiles, 9 periodic images per tile."""
    DIM = 2


class GridCorrector3D(_GridCorrectorND):
    """3D grid corrector: n_cells^3 tiles, 27 periodic images per tile."""
    DIM = 3

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
