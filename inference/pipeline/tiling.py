"""
pipeline/tiling.py
------------------
Spatial tiling of a 2D periodic domain for blocked model inference.

The domain [0, L)^2 is split into a grid_size x grid_size regular grid.
Each tile is extended by ghost_width = ghost_factor * rd_test on all sides.

Grid configs live in inference/configs/grids/*.yaml.
Use TilingConfig.from_yaml() to load one.

Recommended grids (N=2500 SPH, rd_test=0.02):
  grid_6x6.yaml   ->  ~107 pts/tile (69 core + 38 ghost)  [N=100 model]
  grid_10x10.yaml ->  ~49 pts/tile  (25 core + 24 ghost)  [N=50 model]

Ghost buffer correctness:
  ghost_factor >= 1.0 guarantees every PBC-violating pair is visible in at
  least one tile's ghost-augmented neighbourhood. Do not set below 1.0.
"""
from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np
import yaml


@dataclass
class TilingConfig:
    grid_size:    int    # number of tiles per dimension (total: grid_size^2)
    ghost_factor: float  # ghost_width = ghost_factor * rd_test  (must be >= 1.0)
    domain:       float = 1.0
    name:         str   = ""

    @classmethod
    def from_yaml(cls, path: str) -> 'TilingConfig':
        """Load a grid config from inference/configs/grids/*.yaml."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            grid_size    = int(raw['grid_size']),
            ghost_factor = float(raw.get('ghost_factor', 1.0)),
            name         = raw.get('name', ''),
        )

    @property
    def cell_size(self) -> float:
        return self.domain / self.grid_size

    @property
    def n_tiles(self) -> int:
        return self.grid_size ** 2

    def ghost_width(self, rd_test: float) -> float:
        if self.ghost_factor < 1.0:
            import warnings
            warnings.warn(
                f"ghost_factor={self.ghost_factor} < 1.0 violates the correctness "
                "guarantee (ghost_width < rd_test). Some cross-boundary violations "
                "may be invisible to the model. Use only for ablation experiments.",
                UserWarning, stacklevel=2,
            )
        return self.ghost_factor * rd_test


def iter_tiles(cfg: TilingConfig) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (tile_lo, tile_hi) for every tile in row-major order."""
    c = cfg.cell_size
    for i in range(cfg.grid_size):
        for j in range(cfg.grid_size):
            yield (np.array([i * c, j * c]),
                   np.array([(i+1)*c, (j+1)*c]))
