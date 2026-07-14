"""
src/inference/experiments/obstruction/obstruction.py
-----------------------
Utilities for handling domain obstructions (gears, ellipses, etc.)
in SPH initialization with the ML corrector.

The key idea: fill the obstruction interior with ghost particles at
spacing rd so the corrector sees a uniform-looking environment and
naturally pushes real particles away from the boundary.

Typical workflow
----------------
    mask      = ellipse_mask(cx=0.5, cy=0.5, a=0.15, b=0.10)
    ghosts    = fill_obstruction(mask, rd=0.02)
    real_pts  = grid_init(rd=0.02, domain=1.0, exclude=mask)
    all_pts   = np.concatenate([real_pts, ghosts])
    corrected = corrector.apply(all_pts, k=3)
    result    = corrected[: len(real_pts)]   # drop ghost rows
"""
import numpy as np
from typing import Callable, Optional, Tuple


# ── fill ──────────────────────────────────────────────────────────────────────

def fill_obstruction(
    is_inside: Callable[[np.ndarray], np.ndarray],
    rd:        float,
    bounds:    Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    erode_by:  Optional[float] = None,
) -> np.ndarray:
    """
    Fill the interior of an obstruction with a square-grid of ghost particles
    at spacing rd.

    Grid spacing equal to rd guarantees every ghost-to-ghost minimum distance
    is exactly rd (orthogonal neighbours), matching the corrector's training
    distribution and preventing spurious violation edges between ghosts.

    Parameters
    ----------
    is_inside : callable (N, 2) -> (N,) bool
                Returns True for points that lie inside the obstruction.
    rd        : grid spacing — should match rd_test used in the corrector.
    bounds    : (xmin, xmax, ymin, ymax) search bounding box.
                Defaults to the unit domain; pass the tight bounding box of
                the obstruction for efficiency.
    erode_by  : if given, discard ghost particles closer than this distance to
                the exterior before filling.  Use erode_by=rd to shift the
                outermost ghost layer rd inward from the true boundary.
                Effect: a real particle sitting exactly on the boundary is at
                distance rd from the nearest ghost — satisfying the minimum-
                distance constraint without triggering repulsion.  Real
                particles therefore naturally conform to the shape contour.
                None (default) fills the full interior.

    Returns
    -------
    (M, 2) float32 array of ghost particle positions.
    """
    from scipy.spatial import cKDTree

    xmin, xmax, ymin, ymax = bounds
    xs   = np.arange(xmin, xmax + 0.5 * rd, rd)
    ys   = np.arange(ymin, ymax + 0.5 * rd, rd)
    gx, gy = np.meshgrid(xs, ys)
    pts    = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

    mask      = is_inside(pts)
    ghost_pts = pts[mask]

    if erode_by is None or len(ghost_pts) == 0:
        return ghost_pts

    # Discard ghosts closer than erode_by to any exterior grid point.
    # This shifts the outermost ghost layer erode_by inward from the boundary.
    exterior_pts = pts[~mask]
    dists = cKDTree(exterior_pts).query(ghost_pts, workers=-1)[0]
    return ghost_pts[dists >= erode_by]


# ── convenience masks ─────────────────────────────────────────────────────────

def ellipse_mask(
    cx: float, cy: float,
    a:  float, b:  float,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    is_inside predicate for an axis-aligned ellipse.

    Parameters
    ----------
    cx, cy : centre coordinates
    a, b   : semi-axes (x and y directions)
    """
    def _inside(pts: np.ndarray) -> np.ndarray:
        x = pts[:, 0] - cx
        y = pts[:, 1] - cy
        return (x / a) ** 2 + (y / b) ** 2 <= 1.0
    return _inside


def circle_mask(
    cx: float, cy: float,
    r:  float,
) -> Callable[[np.ndarray], np.ndarray]:
    """is_inside predicate for a circle (special case of ellipse_mask)."""
    return ellipse_mask(cx, cy, r, r)


def gear_mask(
    cx: float, cy: float,
    r:  float,
    n_teeth: int   = 8,
    k:       float = 4.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    is_inside predicate for a gear shape.

    Gear = circle of radius r  +  n_teeth square teeth on the circumference.
    Each tooth is a square of side r/k centred on the circumference point,
    with two sides perpendicular to the local radius (tangential) and two
    sides parallel to it (radial).  The tooth therefore protrudes r/(2k)
    beyond the circle and r/(2k) into it.

    Parameters
    ----------
    cx, cy  : centre coordinates
    r       : radius of the base circle
    n_teeth : number of equally-spaced teeth (default 8)
    k       : tooth-size divisor — side length = r/k (default 4)
    """
    s_half  = r / (2.0 * k)
    angles  = np.linspace(0.0, 2.0 * np.pi, n_teeth, endpoint=False)
    centers = np.column_stack([cx + r * np.cos(angles),
                               cy + r * np.sin(angles)])   # (T, 2)
    e_rad   = np.column_stack([np.cos(angles), np.sin(angles)])    # (T, 2) radial
    e_tan   = np.column_stack([-np.sin(angles), np.cos(angles)])   # (T, 2) tangential

    def _inside(pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        dx, dy    = pts[:, 0] - cx, pts[:, 1] - cy
        in_circle = dx ** 2 + dy ** 2 <= r ** 2

        # dp[n, t] = pts[n] - centers[t],  shape (N, T, 2)
        dp     = pts[:, None, :] - centers[None, :, :]
        proj_r = np.abs(np.einsum('ntd,td->nt', dp, e_rad))   # (N, T)
        proj_t = np.abs(np.einsum('ntd,td->nt', dp, e_tan))   # (N, T)
        in_tooth = np.any((proj_r <= s_half) & (proj_t <= s_half), axis=1)

        return in_circle | in_tooth

    return _inside


def polygon_mask(
    vertices: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    is_inside predicate for an arbitrary closed polygon (ray-casting).

    Parameters
    ----------
    vertices : (V, 2) array of polygon vertices in order (open or closed).
    """
    verts = np.asarray(vertices, dtype=np.float64)
    if not np.allclose(verts[0], verts[-1]):
        verts = np.concatenate([verts, verts[:1]], axis=0)

    def _inside(pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        x, y = pts[:, 0], pts[:, 1]
        n    = len(verts) - 1
        inside = np.zeros(len(pts), dtype=bool)
        xv, yv = verts[:n, 0], verts[:n, 1]
        xw, yw = verts[1:,  0], verts[1:,  1]
        for i in range(n):
            xi, yi = xv[i], yv[i]
            xj, yj = xw[i], yw[i]
            cond = ((yi > y) != (yj > y)) & \
                   (x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi)
            inside ^= cond
        return inside
    return _inside
