"""
pipeline/tiling.py
------------------
Spatial tiling of a 2D periodic domain for blocked model inference.

The domain [0, L)^2 is split into an n_cells x n_cells regular grid.
Each tile is extended by ghost_width = ghost_factor * cell_size on all sides.

Recommended grids (N=2500 SPH, rd_test=0.02, domain~1.0):
  n_cells=6  ->  ~107 pts/tile (69 core + 38 ghost)  [N=100 model]
  n_cells=10 ->  ~49 pts/tile  (25 core + 24 ghost)  [N=50 model]

Ghost buffer correctness:
  ghost_width = ghost_factor * cell_size must be >= rd_test, or some
  cross-boundary violations become invisible to the model. ghost_width()
  below warns if that's violated.
"""
from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np


@dataclass
class TilingConfig:
    n_cells:      int    # number of tiles per dimension (total: n_cells^2)
    ghost_factor: float  # ghost_width = ghost_factor * cell_size (fraction of a tile)
    domain:       float = 1.0

    @property
    def cell_size(self) -> float:
        return self.domain / self.n_cells

    @property
    def n_tiles(self) -> int:
        return self.n_cells ** 2

    def ghost_width(self, rd_test: float) -> float:
        width = self.ghost_factor * self.cell_size
        if width < rd_test:
            import warnings
            warnings.warn(
                f"ghost_width={width:.4g} (ghost_factor={self.ghost_factor} of "
                f"cell_size={self.cell_size:.4g}) is below rd_test={rd_test}: some "
                "cross-boundary violations may be invisible to the model.",
                UserWarning, stacklevel=2,
            )
        return width


def iter_tiles(cfg: TilingConfig) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (tile_lo, tile_hi) for every tile in row-major order."""
    c = cfg.cell_size
    for i in range(cfg.n_cells):
        for j in range(cfg.n_cells):
            yield (np.array([i * c, j * c]),
                   np.array([(i+1)*c, (j+1)*c]))
