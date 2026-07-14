"""
obstruction_demo.py
-------------------
Visualize the three obstruction types: ellipse, polygon (hexagon), gear.

Two rows x three columns:
  Row 1 — full fill  (ghosts start at the boundary → real particles pushed rd
           away from the contour, leaving a visible gap)
  Row 2 — eroded fill  (erode_by=rd: ghosts start rd inside the boundary →
           real particles can sit directly on the contour)

The red line marks the true boundary in every panel for reference.

Run:
    .venv\\Scripts\\python.exe src/inference/experiments/obstruction/obstruction_demo.py
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, Polygon
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))

from inference.experiments.obstruction.obstruction import (
    fill_obstruction, ellipse_mask, polygon_mask, gear_mask,
)

RD   = 0.01      # fine grid for visual clarity (actual SPH use: rd_test = 0.02)
DOM  = 1.0
CX, CY = 0.5, 0.5

# ── full-domain background grid ────────────────────────────────────────────────
xs = np.arange(0, DOM + 0.5 * RD, RD)
ys = np.arange(0, DOM + 0.5 * RD, RD)
gx, gy = np.meshgrid(xs, ys)
grid   = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

# ── obstruction definitions ────────────────────────────────────────────────────
_A, _B = 0.22, 0.14
_R_HEX = 0.18
_R_GEAR, _N_TEETH, _K = 0.18, 8, 2

hex_angles = np.linspace(0, 2 * np.pi, 7)[:-1]
hex_verts  = np.column_stack([CX + _R_HEX * np.cos(hex_angles),
                               CY + _R_HEX * np.sin(hex_angles)])

obstructions = [
    ('Ellipse',           f'a={_A},  b={_B}',                 ellipse_mask(CX, CY, _A, _B)),
    ('Polygon (hexagon)', f'r={_R_HEX},  6 vertices',          polygon_mask(hex_verts)),
    ('Gear',              f'r={_R_GEAR},  8 teeth,  k={_K}',   gear_mask(CX, CY, _R_GEAR, _N_TEETH, _K)),
]

# ── boundary drawing helpers ───────────────────────────────────────────────────
def _draw_boundary(ax, idx):
    """Overlay the true boundary as a thin red line."""
    if idx == 0:   # ellipse
        ax.add_patch(Ellipse((CX, CY), 2*_A, 2*_B,
                             fill=False, edgecolor='red', lw=0.8, zorder=5))
    elif idx == 1:  # hexagon
        ax.add_patch(Polygon(hex_verts, closed=True,
                             fill=False, edgecolor='red', lw=0.8, zorder=5))
    else:           # gear — draw dense parametric outline
        gear = gear_mask(CX, CY, _R_GEAR, _N_TEETH, _K)
        # sample the outermost layer: thin ring just inside/outside boundary
        theta = np.linspace(0, 2*np.pi, 2000)
        for r_probe in np.linspace(0.01, _R_GEAR + _R_GEAR/_K, 200):
            pass  # skip — draw contour differently below

        # contour via matplotlib: plot boundary by sampling a dense grid
        _res  = 400
        _t    = np.linspace(0, DOM, _res)
        _gx2, _gy2 = np.meshgrid(_t, _t)
        _pts2 = np.stack([_gx2.ravel(), _gy2.ravel()], axis=1).astype(np.float32)
        _m    = gear(_pts2).reshape(_res, _res).astype(float)
        ax.contour(_t, _t, _m, levels=[0.5], colors='red', linewidths=0.8, zorder=5)


# ── colors ─────────────────────────────────────────────────────────────────────
C_REAL  = '#cccccc'
C_GHOST = '#111111'
S_REAL  = 4
S_GHOST = 6

# ── plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(
    f'Obstruction ghost fill  (rd = {RD})  —  '
    'gray = real,  black = ghost,  red line = true boundary',
    fontsize=12, fontweight='bold', y=1.01,
)

rows = [
    (None, 'Full fill\n'
           'ghosts touch boundary → real particles\n'
           'pushed rd away from contour (gap visible)'),
    (RD,   f'Eroded fill  (erode_by = rd = {RD})\n'
            'ghosts start rd inside boundary → real particles\n'
            'can sit directly on the contour (no gap)'),
]

for row_idx, (erode_by, row_label) in enumerate(rows):
    for col_idx, (name, params, mask) in enumerate(obstructions):
        ax = axes[row_idx, col_idx]

        inside = mask(grid)
        real   = grid[~inside]
        ghost  = fill_obstruction(mask, rd=RD, erode_by=erode_by)

        ax.scatter(real[:, 0],  real[:, 1],  c=C_REAL,  s=S_REAL,  lw=0, alpha=0.7)
        ax.scatter(ghost[:, 0], ghost[:, 1], c=C_GHOST, s=S_GHOST, lw=0, alpha=1.0)

        _draw_boundary(ax, col_idx)

        title_top = f'{name}  —  {params}' if row_idx == 0 else ''
        ax.set_title(
            f'{title_top}\n{len(ghost)} ghost,  {len(real)} real'.strip('\n'),
            fontsize=9, pad=4,
        )
        ax.set_aspect('equal')
        ax.set_xlim(0.0, DOM)
        ax.set_ylim(0.0, DOM)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        if col_idx == 0:
            ax.set_ylabel(row_label, fontsize=8.5, labelpad=8)

legend_handles = [
    mpatches.Patch(color=C_REAL,  label='real particles  (outside obstruction)'),
    mpatches.Patch(color=C_GHOST, label='ghost particles (inside obstruction)'),
    mpatches.Patch(color='red',   label='true boundary'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=3,
           fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.03))

plt.tight_layout(rect=[0, 0.04, 1, 1])

out = Path('artifacts/inference/experiments/obstruction/runs') / 'obstruction_demo.png'
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=160, bbox_inches='tight')
print(f'Saved -> {out}')
plt.show()
