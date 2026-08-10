"""
datagen_stages.py  (vibecoding / visualizations)
------------------------------------------------
The SPH data-*generation* process, exactly as datagen.py does it, broken into
its four internal steps. The pre-torus steps are drawn flat; once the cloud is
placed on the periodic box it is drawn on the actual 3D torus (the box IS a
donut once x=0..1 and y=0..1 are glued):

    1. Lattice          the 7x7 grid at cell centres            [flat]
    2. + jitter         per-point jitter (capped so rd holds)   [flat]
    3. Torus placement  + random torus shift, wrapped           [on the donut]
                        ==  CLEAN output of _periodic_cloud()
    4. + noise          + Gaussian noise, wrapped               [on the donut]
                        ==  NOISY output of noise_sample()

Panels 1-2 are intermediates inside _periodic_cloud(); panels 3-4 are the two
states the dataset actually yields per example: (clean, noisy). No correction
step — going back to a lattice is the model's job (inference), not the dataset's.

Points are coloured by nearest-neighbour distance on a shared scale, so the
violations (nn < rd) the noise introduces show up in panel 4.

Prototype visualization — not part of src/, not tested.

Run (from the repo root):
    .venv\\Scripts\\python.exe vibecoding/visualizations/datagen_stages.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from training.datagen import PoissonDiskDataset
from utils.metrics import nn_dists

# ── knobs ─────────────────────────────────────────────────────────────────────
N            = 49            # 7x7 lattice — the production SPH training cardinality
RD           = 0.14          # min-distance constraint
NOISE_SIGMA  = 0.40 * RD     # displacement noise (fixed, for a clear "noisy")
SEED         = 3
BOX          = 1.0           # unit torus
R_MAJOR      = 1.0           # donut major radius; R_MINOR the tube radius
R_MINOR      = 0.55
Z_EXAGG      = 1.8           # vertical stretch so the (geometrically flat) donut reads as 3D
ELEV, AZIM   = 18, 45        # torus camera: low, angled, isometric-ish
OUT          = Path(__file__).parent / 'outputs' / 'datagen_stages.png'

# ── reproduce datagen's steps (lattice params read off the real dataset) ──────
rng = np.random.default_rng(SEED)
ds = PoissonDiskDataset(dim=2, cardinality=N, rd=RD, seed=SEED,
                        noise_scale_min=0.0, noise_scale_max=NOISE_SIGMA, periodic=True)
if not getattr(ds, '_use_lattice', False):
    raise SystemExit(f'lattice fast-path inactive for N={N}, rd={RD} — this demo needs it')
n_side, jitter = ds._lattice_side, ds._lattice_jitter

# _periodic_cloud(): lattice -> +jitter -> +torus shift -> wrap  == clean output
g        = (np.arange(n_side) + 0.5) / n_side
lattice  = np.stack(np.meshgrid(g, g, indexing='ij'), -1).reshape(-1, 2).astype(np.float32)
jittered = lattice + rng.uniform(-jitter, jitter, size=lattice.shape)
clean    = np.mod(jittered + rng.uniform(size=(1, 2)), 1.0).astype(np.float32)

# noise_sample(): clean + Gaussian noise -> wrap  == noisy output
noisy    = np.mod(clean + rng.normal(0.0, NOISE_SIGMA, size=clean.shape), 1.0).astype(np.float32)

#          title                          points     projection  flat-ghosts
stages = [('1. Lattice',                  lattice,  '2d', False),
          ('2. + jitter',                 jittered, '2d', False),
          ('3. Torus placement  (clean)', clean,    '3d', None),
          ('4. + noise  (noisy)',         noisy,    '3d', None),
          ('5. Unwrapped to 2D  (noisy)', noisy,    '2d', True)]

# ── shared colour scale (nn distance, periodic) ───────────────────────────────
nn_all = np.concatenate([nn_dists(p, box=BOX) for _, p, *_ in stages])
vmin, vmax = np.percentile(nn_all, 2), np.percentile(nn_all, 98)

NORM = plt.Normalize(vmin, vmax)
SCATTER = dict(cmap='viridis', vmin=vmin, vmax=vmax, s=40, lw=0.4, edgecolor='white')


def _eye_dir(elev, azim):
    """Unit vector from the origin toward the camera (for depth ordering)."""
    e, a = np.radians(elev), np.radians(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def to_torus(uv, lift=1.0):
    """Flat unit square (u, v) in [0,1)^2 -> point on the 3D torus surface."""
    u  = 2 * np.pi * uv[:, 0]
    v  = 2 * np.pi * uv[:, 1]
    rr = R_MINOR * lift
    x  = (R_MAJOR + rr * np.cos(v)) * np.cos(u)
    y  = (R_MAJOR + rr * np.cos(v)) * np.sin(u)
    z  = rr * np.sin(v)
    return x, y, z


# translucent donut mesh for the 3d panels
uu, vv = np.meshgrid(np.linspace(0, 2 * np.pi, 90), np.linspace(0, 2 * np.pi, 45))
Xs = (R_MAJOR + R_MINOR * np.cos(vv)) * np.cos(uu)
Ys = (R_MAJOR + R_MINOR * np.cos(vv)) * np.sin(uu)
Zs = R_MINOR * np.sin(vv)


def draw_flat(ax, pts, title, show_ghosts=False):
    # faint periodic replicas convey the wrap for a cloud that lives on the torus
    if show_ghosts:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                ax.scatter(pts[:, 0] + dx, pts[:, 1] + dy, c='#cfcfcf',
                           s=14, lw=0, alpha=0.30, zorder=1)
    nn = nn_dists(pts, box=BOX)
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=nn, zorder=3, **SCATTER)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec='#333333', lw=1.1, zorder=2))
    m = 0.14 if show_ghosts else 0.05
    ax.set_xlim(-m, 1 + m); ax.set_ylim(-m, 1 + m)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    _annotate(ax, nn)
    return sc


def draw_graticule(ax, n_lines):
    """The n_lines x n_lines reference grid drawn as thin lines on the torus."""
    tfine = np.linspace(0, 2 * np.pi, 160)
    for a in np.linspace(0, 2 * np.pi, n_lines, endpoint=False):
        # parallel: circle around the big loop at tube-angle v = a
        ax.plot((R_MAJOR + R_MINOR * np.cos(a)) * np.cos(tfine),
                (R_MAJOR + R_MINOR * np.cos(a)) * np.sin(tfine),
                R_MINOR * np.sin(a) * np.ones_like(tfine),
                color='#9a9a9a', lw=0.4, alpha=0.55, zorder=2)
        # meridian: circle around the tube at loop-angle u = a
        ax.plot((R_MAJOR + R_MINOR * np.cos(tfine)) * np.cos(a),
                (R_MAJOR + R_MINOR * np.cos(tfine)) * np.sin(a),
                R_MINOR * np.sin(tfine),
                color='#9a9a9a', lw=0.4, alpha=0.55, zorder=2)


def draw_torus(ax, pts, title):
    ax.plot_surface(Xs, Ys, Zs, color='#e2e2e2', alpha=0.10,
                    linewidth=0, antialiased=True, shade=False, zorder=1)
    draw_graticule(ax, n_side)
    x, y, z = to_torus(pts, lift=1.05)         # float points just above the tube
    nn = nn_dists(pts, box=BOX)
    # hue = nn distance; depth toward the camera drives size + opacity so the
    # near face reads as near (matplotlib won't occlude scatter behind a surface)
    depth = np.column_stack([x, y, z]) @ _eye_dir(ELEV, AZIM)
    d01   = (depth - depth.min()) / (np.ptp(depth) + 1e-9)   # 0 = far, 1 = near
    rgba  = plt.cm.viridis(NORM(nn))
    rgba[:, 3] = 0.35 + 0.65 * d01                            # far points fade
    sc = ax.scatter(x, y, z, c=rgba, s=16 + 46 * d01, lw=0.4, edgecolor='white',
                    depthshade=False, zorder=4)
    # isometric orthographic view; z exaggerated so the flat donut reads as 3D
    lim_xy, lim_z = R_MAJOR + R_MINOR + 0.05, R_MINOR + 0.05
    ax.set_xlim(-lim_xy, lim_xy); ax.set_ylim(-lim_xy, lim_xy); ax.set_zlim(-lim_z, lim_z)
    ax.set_box_aspect((2 * lim_xy, 2 * lim_xy, 2 * lim_z * Z_EXAGG))
    ax.set_proj_type('ortho')
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11)
    _annotate(ax, nn)
    return sc


def _annotate(ax, nn):
    label = f'mean nn = {nn.mean():.3f}   illegal = {100*(nn < RD).mean():.0f}%'
    # 3d axes annotate with text2D (figure-relative), 2d axes with text
    place = ax.text2D if hasattr(ax, 'text2D') else ax.text
    place(0.5, -0.02, label, transform=ax.transAxes,
          ha='center', va='top', fontsize=8.5, color='#333333')


n_panels = len(stages)
fig = plt.figure(figsize=(4.1 * n_panels, 4.8))
fig.suptitle(f'SPH data generation (datagen.py)   (N={N}, rd={RD}, noise sigma={NOISE_SIGMA:.3f})',
             fontsize=12, y=1.0)
for i, (title, pts, proj, gh) in enumerate(stages):
    if proj == '3d':
        ax = fig.add_subplot(1, n_panels, i + 1, projection='3d')
        draw_torus(ax, pts, title)
    else:
        ax = fig.add_subplot(1, n_panels, i + 1)
        draw_flat(ax, pts, title, show_ghosts=bool(gh))

fig.subplots_adjust(left=0.01, right=0.93, bottom=0.06, top=0.90, wspace=0.05)
cbar_ax = fig.add_axes([0.945, 0.18, 0.008, 0.62])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap='viridis'), cax=cbar_ax)
cb.set_label('nearest-neighbour distance', fontsize=9)
cb.ax.axhline(RD, color='#e74c3c', lw=1.2)   # rd reference: below the line = violation

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved -> {OUT}')
for title, pts, *_ in stages:
    nn = nn_dists(pts, box=BOX)
    print(f'  {title:30s}  mean_nn={nn.mean():.4f}  illegal={100*(nn<RD).mean():.1f}%')
