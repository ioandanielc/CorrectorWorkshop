"""
src/inference/correctors/wholecloud/wholecloud_corrector.py
-----------------------------------------------------------
Whole-cloud sparse corrector: one model call over the ENTIRE cloud per pass,
using forward_sparse (edge-list message passing) instead of tiling. No tiles,
no ghosts, no seams — the deployment path for box-aware models (model12):
~50x faster than tiled inference at N=2500 and equal-or-better KG at every
timestep (seam-free), validated by an actual SPH re-simulation.

Requires a model with forward_sparse (model12); model9 has no sparse path.

Usage
-----
    from inference.correctors.wholecloud.wholecloud_corrector import (
        WholeCloudCorrector2D, WholeCloudCorrector2DConfig)

    cfg = WholeCloudCorrector2DConfig.from_yaml('src/configs/experiments/sph_tv/model12_wholecloud.yaml')
    corrector = WholeCloudCorrector2D(cfg)

    pts_corrected = corrector.apply(pts, k=5)   # (N, 2) -> (N, 2)
"""
import sys
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parents[3]))
from utils.config import load_model_config
from ..base import Corrector
from ..common.scaling import compute_scale


@dataclass
class WholeCloudCorrector2DConfig:
    # model
    checkpoint:   str
    model_config: str
    rd_train:     float

    # data
    rd_test: float
    box:     Optional[float] = None   # true PBC box size; None = infer from extent
    device:  str             = 'cpu'

    @classmethod
    def from_yaml(cls, path: str) -> 'WholeCloudCorrector2DConfig':
        """Load a corrector config. 'box' lives in the data section (a property
        of the dataset, like rd_test)."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        m   = raw['model']
        d   = raw['data']
        exp = raw.get('experiment', {})
        box = d.get('box', None)
        return cls(
            checkpoint   = m['checkpoint'],
            model_config = m['config'],
            rd_train     = float(m['rd_train']),
            rd_test      = float(d['rd_test']),
            box          = None if box is None else float(box),
            device       = exp.get('device', 'cpu'),
        )


class WholeCloudCorrector3DConfig(WholeCloudCorrector2DConfig):
    """Same fields as the 2D config; separate type so 3D defaults can diverge later."""


class _WholeCloudCorrectorND(Corrector):
    """
    Whole-cloud sparse inference: per pass, scale the cloud to the model's
    training rd, build the PBC edge list (all ordered pairs within
    attention_rd, both directions), run forward_sparse once, unscale the
    displacements, wrap back into the box. Dimension-generic body — concrete
    subclasses set DIM.

    Unlike the tiled correctors, the periodic box can (and for known-PBC data
    should) be given explicitly via cfg.box: extent-based domain inference
    undershoots the true box on lattice-like clouds, pinching the wrap seam
    (see CLAUDE.md known rough edges). With cfg.box = None the frame is
    inferred exactly like the grid corrector's.
    """

    DIM: int = None   # set by concrete subclasses

    def __init__(self, cfg: WholeCloudCorrector2DConfig):
        self.cfg    = cfg
        self.device = torch.device(cfg.device)
        self.scale  = compute_scale(cfg.rd_train, cfg.rd_test)
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

        if not hasattr(self.model, 'forward_sparse'):
            raise TypeError(
                f'{module_path} has no forward_sparse — the whole-cloud corrector '
                f'needs the sparse edge-list path (model12); use the grid/kdtree '
                f'correctors for dense-only models (model9).')

        # The edge cutoff and the rd passed to forward_sparse are the model's
        # attention radius (its calling convention), NOT the constraint rd —
        # rd_train only drives the coordinate scaling above.
        self.attention_rd = float(model_cfg.get('attention_rd', self.cfg.rd_train))
        self.rd_t = torch.tensor(self.attention_rd, dtype=torch.float32, device=self.device)

    def apply(self, points: np.ndarray, k: int = 1) -> np.ndarray:
        """
        Apply k whole-cloud sparse passes.

        Parameters
        ----------
        points : (N, DIM)  positions; if cfg.box is set they must live on that
                 torus (any representative coordinates in [0, box) after
                 wrapping), otherwise the frame is inferred from the extent
        k      : int       number of correction passes

        Returns
        -------
        (N, DIM)  corrected positions, wrapped into [0, box) when cfg.box is
        set, else in the input's coordinate frame
        """
        assert points.shape[1] == self.DIM, f'expected (N, {self.DIM}), got {points.shape}'

        if self.cfg.box is not None:
            box, centroid = float(self.cfg.box), None
            p = points.astype(np.float32) % np.float32(box)
        else:
            # grid-corrector frame inference: center on centroid, domain = max extent
            centroid = points.mean(axis=0)
            centered = points - centroid
            box = float((centered.max(axis=0) - centered.min(axis=0)).max())
            p = (centered + box / 2).astype(np.float32) % np.float32(box)

        # scaled box; clip guards cKDTree's strict [0, boxsize) requirement
        box_s = np.float32(self.scale * box)
        edge  = box_s - np.float32(1e-4)

        for _ in range(k):
            ps = np.clip((p * self.scale) % box_s, 0.0, edge)
            pairs = cKDTree(ps, boxsize=float(box_s)).query_pairs(
                self.attention_rd, output_type='ndarray')
            if len(pairs) == 0:
                break   # no pair within the attention radius — nothing to correct
            ei = torch.tensor(np.stack([np.concatenate([pairs[:, 0], pairs[:, 1]]),
                                        np.concatenate([pairs[:, 1], pairs[:, 0]])]),
                              dtype=torch.long, device=self.device)
            with torch.no_grad():
                disp = self.model.forward_sparse(
                    torch.tensor(ps, device=self.device), ei,
                    rd=self.rd_t, box=float(box_s)).cpu().numpy()
            p = (p + disp / self.scale) % box

        if centroid is None:
            return p
        return p - box / 2 + centroid


class WholeCloudCorrector2D(_WholeCloudCorrectorND):
    """2D whole-cloud sparse corrector."""
    DIM = 2


class WholeCloudCorrector3D(_WholeCloudCorrectorND):
    """3D whole-cloud sparse corrector."""
    DIM = 3
