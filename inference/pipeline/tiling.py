"""
pipeline/tiling.py
------------------
Spatial tiling of a 2D periodic domain for blocked model inference.

The domain [0, L)^2 is split into a grid_size x grid_size regular grid.
Each tile is extended by ghost_width = ghost_factor * rd_test on all sides.

Typical configurations (N=2500 SPH data, rd_test=0.02):
  grid=6,  ghost_factor=1.0  ->  ~107 pts/tile (69 core + 38 ghost)  [N=100 model]
  grid=10, ghost_factor=1.0  ->  ~49 pts/tile  (25 core + 24 ghost)  [N=50 model]
"""
from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np


@dataclass
class TilingConfig:
    grid_size: int       # number of tiles per dimension (total: grid_size^2)
    ghost_factor: float  # ghost_width = ghost_factor * rd_test
    domain: float = 1.0

    @property
    def cell_size(self) -> float:
        return self.domain / self.grid_size

    def ghost_width(self, rd_test: float) -> float:
        return self.ghost_factor * rd_test


def iter_tiles(cfg: TilingConfig) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (tile_lo, tile_hi) for every tile in the grid."""
    c = cfg.cell_size
    for i in range(cfg.grid_size):
        for j in range(cfg.grid_size):
            yield (np.array([i * c, j * c]),
                   np.array([(i+1)*c, (j+1)*c]))
