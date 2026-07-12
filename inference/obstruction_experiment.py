"""
obstruction_experiment.py
-------------------------
Two initialization strategies for SPH particles around an obstacle.

Method 1 — grid + ghost particles
  - Real particles: regular grid excluding the obstacle interior
  - Ghost particles: eroded fill (erode_by=rd) so real particles can sit on contour
  - Add small noise to real particles to break the grid symmetry
  - Apply corrector cumulatively (K=1 → 3 → 5), ghosts held fixed each step

Method 2 — uniform + wireframe MC
  - Real particles: uniform random draw outside the obstacle
  - No ghost particles; the obstacle mask is queried directly
  - Apply corrector, then MC-resample any particle that entered the obstacle
  - Repeat cumulatively for K=1 → 3 → 5 steps

Visualization: 2 rows (methods) × 4 columns (initial, K=1, K=3, K=5)
  - Particles colored by min-NN distance (global color scale)
  - Ghost particles shown as faint gray in Method 1 panels
  - True obstacle boundary shown as red contour in all panels

Run:
    .venv\\Scripts\\python.exe inference/obstruction_experiment.py
"""
import sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import cKDTree
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.pipeline.corrector import Corrector, CorrectorConfig
from inference.pipeline.obstruction import fill_obstruction, gear_mask, ellipse_mask

# ── config ────────────────────────────────────────────────────────────────────
RD          = 0.012
DOMAIN      = 1.0
CX, CY      = 0.5, 0.5
NOISE_SCALE = 0.3       # noise_std = NOISE_SCALE * RD
K_VALUES    = [1, 3, 5]
SEED        = 42
DEVICE      = 'cuda'    # 'cpu' if no GPU

# ── obstacle ──────────────────────────────────────────────────────────────────
mask   = gear_mask(CX, CY, r=0.18, n_teeth=8, k=2)
ghosts = fill_obstruction(mask, RD, erode_by=RD)
print(f'Ghost particles: {len(ghosts)}')

# ── corrector ─────────────────────────────────────────────────────────────────
cfg           = CorrectorConfig.from_yaml('inference/configs/grid_6x6.yaml')
cfg.rd_test   = RD     # override to match experiment rd (scale = rd_train/rd_test)
cfg.device    = DEVICE
corrector     = Corrector(cfg)

# ── helpers ───────────────────────────────────────────────────────────────────
def grid_outside(rd, domain, exclude):
    xs = np.arange(0, domain + 0.5 * rd, rd)
    ys = np.arange(0, domain + 0.5 * rd, rd)
    gx, gy = np.meshgrid(xs, ys)
    pts    = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    return pts[~exclude(pts)]


def uniform_outside(N, domain, exclude, rng):
    buf = []
    n   = 0
    while n < N:
        batch = rng.uniform(0, domain, (max(N * 4, 512), 2)).astype(np.float32)
        valid = batch[~exclude(batch)]
        buf.append(valid)
        n += len(valid)
    return np.concatenate(buf)[:N]


def min_nn(pts):
    return cKDTree(pts).query(pts, k=2)[0][:, 1].astype(np.float32)


def draw_boundary(ax, resolution=400):
    t        = np.linspace(0, DOMAIN, resolution)
    gx, gy   = np.meshgrid(t, t)
    pts2     = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    m        = mask(pts2).reshape(resolution, resolution).astype(float)
    ax.contour(t, t, m, levels=[0.5], colors='#e74c3c', linewidths=1.0, zorder=5)


# ── Method 1: grid + ghosts + noise ──────────────────────────────────────────
print('\n--- Method 1 ---')
rng1       = np.random.default_rng(SEED)
real_grid  = grid_outside(RD, DOMAIN, mask)
n_real     = len(real_grid)
noise      = rng1.normal(0, NOISE_SCALE * RD, real_grid.shape).astype(np.float32)
noisy_real = np.clip(real_grid + noise, 0.0, DOMAIN)

m1 = {'initial': noisy_real.copy()}
current_m1 = noisy_real.copy()
k_prev = 0
for k in K_VALUES:
    for _ in range(k - k_prev):
        all_pts   = np.concatenate([current_m1, ghosts])   # ghosts fixed each step
        corrected = corrector.apply(all_pts, k=1)
        current_m1 = corrected[:n_real]
    m1[k] = current_m1.copy()
    print(f'  K={k}: {n_real} real pts,  mean_nn={min_nn(current_m1).mean():.4f}')
    k_prev = k

# ── Method 2: uniform + wireframe MC ─────────────────────────────────────────
print('\n--- Method 2 ---')
rng2       = np.random.default_rng(SEED + 1)
init_unif  = uniform_outside(n_real, DOMAIN, mask, rng2)

m2 = {'initial': init_unif.copy()}
current_m2 = init_unif.copy()
k_prev = 0
for k in K_VALUES:
    for _ in range(k - k_prev):
        corrected = corrector.apply(current_m2, k=1)
        inside    = mask(corrected)
        if inside.any():
            replacement          = uniform_outside(int(inside.sum()), DOMAIN, mask, rng2)
            corrected[inside]    = replacement
        current_m2 = corrected
    m2[k] = current_m2.copy()
    print(f'  K={k}: {n_real} real pts,  mean_nn={min_nn(current_m2).mean():.4f}')
    k_prev = k

# ── visualization ─────────────────────────────────────────────────────────────
print('\nRendering...')
cols       = ['initial'] + K_VALUES
col_titles = ['Initial'] + [f'K = {k}' for k in K_VALUES]

# global color range across all real-particle sets
all_nn = np.concatenate([min_nn(v) for results in [m1, m2] for v in results.values()])
vmin   = np.percentile(all_nn, 2)
vmax   = np.percentile(all_nn, 98)

fig, axes = plt.subplots(4, len(cols), figsize=(5.5 * len(cols), 22))
fig.suptitle(
    f'Obstacle initialization  (gear: r=0.18, 8 teeth, k=2  |  rd={RD})',
    fontsize=12, fontweight='bold', y=1.005,
)

# ── rows 0-1: detailed view (ghosts + color + boundary) ──────────────────────
row_info = [
    (m1, 'Method 1\ngrid + ghost particles + noise\n(gray = ghost, colored = real)'),
    (m2, 'Method 2\nuniform random + wireframe MC replacement'),
]

sc = None
for row_idx, (results, row_label) in enumerate(row_info):
    for col_idx, key in enumerate(cols):
        ax  = axes[row_idx, col_idx]
        pts = results[key]
        nn  = min_nn(pts)

        if row_idx == 0:
            ax.scatter(ghosts[:, 0], ghosts[:, 1],
                       c='#aaaaaa', s=4, lw=0, alpha=0.45, zorder=2)

        sc = ax.scatter(pts[:, 0], pts[:, 1],
                        c=nn, cmap='viridis', s=5, lw=0,
                        vmin=vmin, vmax=vmax, alpha=0.9, zorder=3)

        draw_boundary(ax)

        ax.set_title(col_titles[col_idx], fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlim(0.0, DOMAIN); ax.set_ylim(0.0, DOMAIN)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)

        if col_idx == 0:
            ax.set_ylabel(row_label, fontsize=9, labelpad=8)

        ax.text(0.02, 0.02, f'mean nn={nn.mean():.4f}',
                transform=ax.transAxes, fontsize=7.5, color='white',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='#333', alpha=0.6))

# ── separator line between sections ──────────────────────────────────────────
fig.add_artist(plt.Line2D(
    [0.02, 0.98], [0.505, 0.505],
    transform=fig.transFigure, color='#bbbbbb', lw=1.2, linestyle='--',
))

# ── rows 2-3: clean view (black dots, no obstacle, no color) ─────────────────
clean_row_info = [
    (m1, 'Method 1  (clean)'),
    (m2, 'Method 2  (clean)'),
]

for row_idx, (results, row_label) in enumerate(clean_row_info):
    ax_row = row_idx + 2
    for col_idx, key in enumerate(cols):
        ax  = axes[ax_row, col_idx]
        pts = results[key]

        ax.scatter(pts[:, 0], pts[:, 1],
                   c='black', s=5, lw=0, alpha=0.85, zorder=3)

        ax.set_title(col_titles[col_idx] if ax_row == 2 else '', fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlim(0.0, DOMAIN); ax.set_ylim(0.0, DOMAIN)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor('white')
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)

        if col_idx == 0:
            ax.set_ylabel(row_label, fontsize=9, labelpad=8)

# ── colorbar for top section ──────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0, 0.93, 1])
cbar_ax = fig.add_axes([0.94, 0.515, 0.015, 0.465])
cb = fig.colorbar(sc, cax=cbar_ax)
cb.set_label('min-NN distance', fontsize=9)
cb.ax.axhline(RD, color='red', lw=1.0, linestyle='--')
cb.ax.text(1.6, RD, f'rd={RD}', fontsize=7.5, color='red',
           va='center', transform=cb.ax.get_yaxis_transform())

out = Path(__file__).parent / 'obstruction_experiment.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved -> {out}')
plt.show()
