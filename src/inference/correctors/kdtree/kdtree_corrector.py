"""
src/inference/correctors/kdtree/kdtree_corrector.py
----------------------------
KDTreeCorrector2D / KDTreeCorrector3D — violation-targeted correctors
sharing the dimension-generic body _KDTreeCorrectorND.

Unlike GridCorrector2D (which tiles the whole domain every pass), this
corrector runs the model only where violations exist. One sweep:

  1. Build a PBC cKDTree (boxsize), collect every point with a
     neighbour closer than rd_test, sorted worst-first.
  2. Greedy claiming: the worst unclaimed violator becomes a site; its
     neighbourhood is its `total_core` nearest points, its inner core the
     `inner_core` most central of those not already claimed this sweep.
     Claimed points receive a displacement exactly once per sweep, so
     inner cores are disjoint by construction.
  3. One rectangular (n_sites, total_core, 2) batched model call;
     displacements are applied to the inner cores only — the outer ring
     is frozen context, playing the role of the grid's ghost buffer.

apply(points, k) runs at most k sweeps and stops early once no violations
remain. Later sweeps only touch residual violations, so repetition gets
cheaper — the grid pays full price every pass.

Correctness note: the frozen ring must be at least rd_test thick around
the inner core, else a violating pair straddling the inner-core boundary
is seen but its partner never moved here. Count-based cores can't
guarantee that geometrically, so apply() warns if any site's ring
(r_total - r_inner) falls below rd_test.

Usage
-----
    from inference.correctors.kdtree.kdtree_corrector import (
        KDTreeCorrector2D, KDTreeCorrector2DConfig)

    cfg = KDTreeCorrector2DConfig.from_yaml('src/configs/experiments/sph_tv/grid_6x6.yaml')
    corrector = KDTreeCorrector2D(cfg)

    pts_corrected = corrector.apply(pts, k=10)   # at most 10 sweeps
"""
import sys
import importlib
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[3]))
from utils.config import load_model_config
from models.architectures.model9.invariance import compute_invariant, revert_invariant
from ..base import Corrector
from ..common.scaling import compute_scale


@dataclass
class KDTreeCorrector2DConfig:
    # model
    checkpoint:   str
    model_config: str
    rd_train:     float

    # data
    rd_test:      float
    device:       str = 'cpu'

    # neighbourhoods
    total_core:   int = 100   # points per model input, ~ N_train of the checkpoint
    inner_core:   int = 25    # most central points that receive displacements

    @classmethod
    def from_yaml(cls, path: str) -> 'KDTreeCorrector2DConfig':
        """Load a corrector config. 'kdtree' is an inline {total_core, inner_core} dict."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        m   = raw['model']
        d   = raw['data']
        exp = raw.get('experiment', {})
        kd  = raw.get('kdtree', {})

        return cls(
            checkpoint   = m['checkpoint'],
            model_config = m['config'],
            rd_train     = float(m['rd_train']),
            rd_test      = float(d['rd_test']),
            device       = exp.get('device', 'cpu'),
            total_core   = int(kd.get('total_core', cls.total_core)),
            inner_core   = int(kd.get('inner_core', cls.inner_core)),
        )


@dataclass
class KDTreeCorrector3DConfig(KDTreeCorrector2DConfig):
    # Defaults sized for the first 3D checkpoint (N_train=50); the 2D defaults
    # track the n100_p050 checkpoint instead.
    total_core:   int = 50
    inner_core:   int = 12


class _KDTreeCorrectorND(Corrector):
    """
    Applies the trained Poisson-disk corrector only around violations,
    via greedy KD-tree neighbourhoods with PBC (see module docstring).
    Dimension-generic body — concrete subclasses set DIM.

    The point cloud's frame is inferred fresh on every apply() call, the
    same way the grid correctors do it: centered on its own centroid,
    domain taken as the largest axis extent of the centered cloud.
    """

    DIM: int = None   # set by concrete subclasses

    def __init__(self, cfg: KDTreeCorrector2DConfig):
        if cfg.inner_core > cfg.total_core:
            raise ValueError(
                f'inner_core={cfg.inner_core} must be <= total_core={cfg.total_core}')
        self.cfg    = cfg
        self.device = torch.device(cfg.device)
        self.scale  = compute_scale(cfg.rd_train, cfg.rd_test)
        self.rd_t   = torch.tensor(cfg.rd_train, dtype=torch.float32, device=self.device)
        self._load_model()

    def _load_model(self):
        model_cfg = load_model_config(self.cfg.model_config)
        module_path = model_cfg.get('model_file', 'models/architectures/model9/model9')
        m = importlib.import_module(module_path.replace('/', '.'))
        self.model = m.CorrectorModel(
            model_cfg, input_dim=self.DIM, initialization='xavier_uniform'
        )
        state = torch.load(self.cfg.checkpoint, map_location='cpu')
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        # See grid/corrector.py's _load_model for the attention_rd / periodic_frame
        # rationale — same adapter, same reasoning, kept in sync deliberately.
        self.periodic_frame = getattr(self.model, 'uses_box', False)
        attention_rd = model_cfg.get('attention_rd', self.cfg.rd_train)
        self.rd_model_t = torch.tensor(attention_rd, dtype=torch.float32, device=self.device)

    def apply(self, points: np.ndarray, k: int = 1) -> np.ndarray:
        """
        Run at most k greedy sweeps; stop early once no violations remain.

        Parameters
        ----------
        points : (N, 2)  positions in any consistent coordinate frame
        k      : int     maximum number of sweeps

        Returns
        -------
        (N, 2)  corrected positions, in the same coordinate frame as the input
        """
        rd_test  = self.cfg.rd_test
        points   = points.astype(np.float32)
        centroid = points.mean(axis=0)
        centered = points - centroid
        domain   = float((centered.max(axis=0) - centered.min(axis=0)).max())
        # wrap into [0, domain); the clip only guards the exact-domain float32
        # edge, which cKDTree(boxsize) rejects — it must never move a point
        dom_edge = np.nextafter(np.float32(domain), np.float32(0.0))
        pts = ((centered + domain / 2) % domain).astype(np.float32)
        pts = np.clip(pts, 0.0, dom_edge)

        n_nb = min(self.cfg.total_core, len(pts))
        n_in = min(self.cfg.inner_core, n_nb)

        for _ in range(k):
            tree  = cKDTree(pts, boxsize=domain)
            pairs = tree.query_pairs(rd_test, output_type='ndarray')   # (M, 2), i < j
            if len(pairs) == 0:
                break

            # per-point severity: distance to the closest violating neighbour
            diff = pts[pairs[:, 1]] - pts[pairs[:, 0]]
            diff -= domain * np.round(diff / domain)
            dist = np.linalg.norm(diff, axis=1)
            closest = np.full(len(pts), np.inf, dtype=np.float64)
            np.minimum.at(closest, pairs[:, 0], dist)
            np.minimum.at(closest, pairs[:, 1], dist)
            violators = np.where(np.isfinite(closest))[0]
            order = violators[np.argsort(closest[violators])]   # worst first

            # ── greedy claiming: disjoint inner cores covering every violator ──
            claimed  = np.zeros(len(pts), dtype=bool)
            nb_idx, tgt_local, thin_rings = [], [], 0
            for p in order:
                if claimed[p]:
                    continue
                dists, idx = tree.query(pts[p], k=n_nb)
                inner = idx[:n_in]
                keep  = ~claimed[inner]
                claimed[inner[keep]] = True
                nb_idx.append(idx)
                tgt_local.append(np.where(keep)[0])
                if dists[-1] - dists[n_in - 1] < rd_test:
                    thin_rings += 1
            if thin_rings:
                warnings.warn(
                    f'{thin_rings} neighbourhood(s) have a frozen ring thinner than '
                    f'rd_test={rd_test}: violating pairs straddling the inner-core '
                    f'boundary may not fully separate. Raise total_core or lower '
                    f'inner_core.', UserWarning, stacklevel=2)

            # ── one rectangular batch: local PBC-unwrapped coords, scaled ──
            nb_idx  = np.asarray(nb_idx)                      # (S, n_nb)
            local   = pts[nb_idx] - pts[nb_idx[:, 0]][:, None]   # col 0 = the center itself
            local  -= domain * np.round(local / domain)       # unwrap around center
            batch   = (local * self.scale).astype(np.float32)

            if self.periodic_frame:
                x_inv = batch
            else:
                x_inv, mean_b, eig = compute_invariant(batch)
            x_t = torch.tensor(x_inv, dtype=torch.float32, device=self.device)
            model_kwargs = {'box': None} if self.periodic_frame else {}
            with torch.no_grad():
                disp = self.model(x_t, rd=self.rd_model_t, **model_kwargs).cpu().numpy()
            if self.periodic_frame:
                corrected = x_inv + disp
            else:
                corrected = revert_invariant(x_inv + disp, mean_b, eig)
            disp_orig = (corrected - batch) / self.scale

            # ── scatter to the disjoint inner cores ──
            tgt_global = np.concatenate([nb_idx[s, tl] for s, tl in enumerate(tgt_local)])
            moves      = np.concatenate([disp_orig[s, tl] for s, tl in enumerate(tgt_local)])
            pts[tgt_global] = (pts[tgt_global] + moves) % domain
            pts = np.clip(pts, 0.0, dom_edge)

        return pts - domain / 2 + centroid


class KDTreeCorrector2D(_KDTreeCorrectorND):
    """2D violation-targeted corrector."""
    DIM = 2


class KDTreeCorrector3D(_KDTreeCorrectorND):
    """3D violation-targeted corrector."""
    DIM = 3
