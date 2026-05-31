"""
sph_visualize.py
----------------
Two functions in one script:

  extract_positions_from_hSpart(path, drop_axes=None)
      Reads a .hSpart HDF5 file and returns a dict {timestep_label: (N, D) array}.
      D is 3 by default; pass drop_axes to collapse to 2D or 1D.

  animate_positions(data, drop_axes=None, ...)
      Animates the per-timestep arrays. Renders a 3D scatter, 2D scatter, or
      1D line depending on how many axes remain after dropping.

CLI (extract + animate in one shot):
    python sph_visualize.py data.hSpart
    python sph_visualize.py data.hSpart --drop z
    python sph_visualize.py data.hSpart --drop y z
    python sph_visualize.py data.hSpart --drop z --color x --save out.gif
    python sph_visualize.py data.hSpart --inspect

    # animate an already-extracted .npz
    python sph_visualize.py positions.npz --drop z

Programmatic:
    from sph_visualize import extract_positions_from_hSpart, animate_positions

    arrays = extract_positions_from_hSpart("data.hSpart", drop_axes=["z"])
    animate_positions(arrays, color_field="x", interval=80)
"""

import argparse
import re
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D projection)


# ══════════════════════════════════════════════════════════════════════════════
#  AXIS LABELS & DROP LOGIC
# ══════════════════════════════════════════════════════════════════════════════

_ALL_AXES  = ["x", "y", "z"]          # canonical order
_AXIS_IDX  = {"x": 0, "y": 1, "z": 2}


def _active_axes(drop_axes: list[str] | None) -> list[str]:
    """Return axis labels that remain after dropping."""
    drop = [a.lower() for a in (drop_axes or [])]
    invalid = [a for a in drop if a not in _AXIS_IDX]
    if invalid:
        raise ValueError(f"Unknown axis name(s): {invalid}. Choose from x, y, z.")
    return [a for a in _ALL_AXES if a not in drop]


def _apply_drop(pts: np.ndarray, active: list[str]) -> np.ndarray:
    """Select columns corresponding to active axes from an (N, 3) array."""
    cols = [_AXIS_IDX[a] for a in active]
    return pts[:, cols]


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def inspect_structure(h5file: h5py.File) -> None:
    """Print the HDF5 tree to stdout (useful for debugging unknown layouts)."""
    print("=== HDF5 structure ===")
    def _print(name, obj):
        indent = "  " * name.count("/")
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}{name}  shape={obj.shape}  dtype={obj.dtype}")
        else:
            print(f"{indent}{name}/")
    h5file.visititems(_print)
    print("======================")


def _sorted_keys(mapping) -> list[str]:
    keys = list(mapping.files if hasattr(mapping, "files") else mapping.keys())
    def _num(k):
        try:
            return int(k)
        except ValueError:
            m = re.search(r"\d+", k)
            return int(m.group()) if m else 0
    try:
        keys.sort(key=_num)
    except Exception:
        keys.sort()
    return keys


def _find_timestep_groups(h5file: h5py.File):
    root_keys = list(h5file.keys())

    # Layout B: VTKHDF
    if "VTKHDF" in root_keys:
        return _parse_vtkhdf(h5file)

    # Layout C: flat single snapshot
    if "coords_0" in root_keys:
        return [("t0", h5file)]

    # Layout A: numeric/named groups each containing coords_0/1/2
    groups = []
    for key in root_keys:
        item = h5file[key]
        if isinstance(item, h5py.Group) and "coords_0" in item:
            groups.append((key, item))

    if groups:
        try:
            groups.sort(key=lambda x: int(x[0]))
        except ValueError:
            groups.sort(key=lambda x: x[0])
        return groups

    raise ValueError(
        "Could not locate coords_0/coords_1/coords_2 datasets.\n"
        "Run with --inspect to view the file structure."
    )


def _parse_vtkhdf(h5file: h5py.File):
    vtk = h5file["VTKHDF"]
    steps_group = vtk.get("Steps")

    if steps_group is None:
        pts = vtk["Points"][:]
        return [("t0", {"coords_0": pts[:, 0],
                        "coords_1": pts[:, 1],
                        "coords_2": pts[:, 2]})]

    n_steps    = int(steps_group["NSteps"][()])
    pt_offsets = steps_group["PointOffsets"][:]
    n_total    = vtk["Points"].shape[0]
    ends       = np.append(pt_offsets[1:], n_total)
    counts     = ends - pt_offsets

    times = steps_group.get("Values")
    labels = (
        [f"t{i}_time{times[i]:.6g}" for i in range(n_steps)]
        if times is not None
        else [f"t{i}" for i in range(n_steps)]
    )

    all_pts = vtk["Points"][:]
    result  = []
    for i, label in enumerate(labels):
        s, e = int(pt_offsets[i]), int(pt_offsets[i] + counts[i])
        chunk = all_pts[s:e]
        result.append((label, {"coords_0": chunk[:, 0],
                                "coords_1": chunk[:, 1],
                                "coords_2": chunk[:, 2]}))
    return result


def _read_coords(source) -> np.ndarray:
    """Return (N, 3) [x, y, z] from an h5py.Group or dict."""
    if isinstance(source, dict):
        x, y, z = (np.asarray(source[f"coords_{i}"]) for i in range(3))
    else:
        x, y, z = (source[f"coords_{i}"][:] for i in range(3))
    return np.stack([x, y, z], axis=1)


def extract_positions_from_hSpart(
    path: str,
    drop_axes: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """
    Read a .hSpart HDF5 file and return a dict of per-timestep position arrays.

    Parameters
    ----------
    path : str
        Path to the .hSpart file.
    drop_axes : list of {"x","y","z"} | None
        Axes to remove from the output arrays.
        E.g. drop_axes=["z"] → each array is (N, 2) with columns [x, y].
        Default (None) keeps all three → (N, 3).

    Returns
    -------
    dict[str, np.ndarray]
        Keys are timestep labels (sorted); values are (N, D) float arrays.
    """
    active = _active_axes(drop_axes)

    with h5py.File(path, "r") as f:
        timesteps = _find_timestep_groups(f)
        arrays = {}
        for label, source in timesteps:
            pts = _read_coords(source)          # (N, 3)
            arrays[label] = _apply_drop(pts, active)
    return arrays


# ══════════════════════════════════════════════════════════════════════════════
#  ANIMATION
# ══════════════════════════════════════════════════════════════════════════════

def _global_bounds(arrays: list[np.ndarray]):
    all_pts = np.vstack(arrays)
    lo = all_pts.min(axis=0)
    hi = all_pts.max(axis=0)
    pad = (hi - lo) * 0.05
    pad[pad == 0] = 1e-6
    return lo - pad, hi + pad


def _get_colors(pts: np.ndarray, color_field: str | None,
                active: list[str], vmin, vmax):
    if color_field in active:
        col = active.index(color_field)
        return pts[:, col]
    if color_field == "r":
        return np.linalg.norm(pts - pts.mean(axis=0), axis=1)
    return None


def animate_positions(
    data,
    *,
    drop_axes: list[str] | None = None,
    interval: int = 100,
    s: float = 2,
    alpha: float = 0.6,
    color_field: str | None = None,
    cmap: str = "plasma",
    save: str | None = None,
    dpi: int = 120,
    title: str = "SPH particle animation",
    figsize: tuple = (8, 7),
) -> animation.FuncAnimation:
    """
    Animate SPH particle positions across timesteps.

    Parameters
    ----------
    data : NpzFile | dict[str, ndarray] | list[ndarray]
        Per-timestep position arrays.  Each array can be (N,3), (N,2), or (N,1)
        — the dimensionality controls whether a 3D scatter, 2D scatter, or 1D
        line is drawn.

    drop_axes : list of {"x","y","z"} | None
        Axes to drop *before* plotting.  Applied on top of any dropping already
        done by extract_positions_from_hSpart.  If the input arrays are already
        (N,2) you can still drop one more to get (N,1).
        Example: drop_axes=["z"] collapses a 3D dataset to an X-Y scatter.

    interval : int
        Milliseconds between frames.
    s : float
        Scatter point size.
    alpha : float
        Point opacity.
    color_field : {"x","y","z","r",None}
        Colour particles by the named axis or by radial distance from centroid.
        Must be one of the *active* (non-dropped) axes, or "r".
    cmap : str
        Matplotlib colormap.
    save : str | None
        Path to save animation (.gif or .mp4).  mp4 requires ffmpeg.
    dpi : int
        DPI for saved output.
    title : str
        Figure suptitle.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    FuncAnimation
    """

    # ── normalise input ───────────────────────────────────────────────────────
    if isinstance(data, list):
        keys   = [f"t{i}" for i in range(len(data))]
        arrays = [np.asarray(a, dtype=float) for a in data]
    else:
        keys   = _sorted_keys(data)
        arrays = [np.asarray(data[k], dtype=float) for k in keys]

    if not arrays:
        raise ValueError("No timesteps found in data.")

    # ── apply additional drop (on top of what was done at extraction) ─────────
    # Infer which axes are already present by column count
    in_dim = arrays[0].shape[1]
    if in_dim == 3:
        current_axes = ["x", "y", "z"]
    elif in_dim == 2:
        # Can't know which two were kept; assume x,y unless user says otherwise
        current_axes = ["x", "y"]
    else:
        current_axes = ["x"]

    if drop_axes:
        extra_active = _active_axes(drop_axes)
        # Intersect with whatever is currently present
        keep = [a for a in current_axes if a in extra_active]
        if not keep:
            raise ValueError("drop_axes removes all remaining dimensions.")
        arrays = [_apply_drop(a, keep) for a in arrays]
        current_axes = keep

    active = current_axes
    ndim   = len(active)

    if ndim not in (1, 2, 3):
        raise ValueError(f"Unexpected dimensionality {ndim} after dropping axes.")

    n_frames = len(arrays)
    lo, hi   = _global_bounds(arrays)

    # ── colour limits ─────────────────────────────────────────────────────────
    if color_field in active:
        col  = active.index(color_field)
        vmin = min(a[:, col].min() for a in arrays)
        vmax = max(a[:, col].max() for a in arrays)
    elif color_field == "r":
        vmin = 0.0
        vmax = max(np.linalg.norm(a - a.mean(axis=0), axis=1).max() for a in arrays)
    else:
        color_field = None
        vmin = vmax = None

    # ── figure & axes ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=12)

    if ndim == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_xlabel(active[0].upper())
        ax.set_ylabel(active[1].upper())
        ax.set_zlabel(active[2].upper())
    else:
        ax = fig.add_subplot(111)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1] if ndim == 2 else -1, hi[1] if ndim == 2 else 1)
        ax.set_xlabel(active[0].upper())
        if ndim == 2:
            ax.set_ylabel(active[1].upper())
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.4, alpha=0.5)

    # ── initial draw ──────────────────────────────────────────────────────────
    pts0 = arrays[0]
    c0   = _get_colors(pts0, color_field, active, vmin, vmax)

    scatter_kw = dict(c=c0, cmap=cmap, vmin=vmin, vmax=vmax,
                      s=s, alpha=alpha, linewidths=0)

    if ndim == 3:
        scat = ax.scatter(pts0[:, 0], pts0[:, 1], pts0[:, 2], **scatter_kw)
    elif ndim == 2:
        scat = ax.scatter(pts0[:, 0], pts0[:, 1], **scatter_kw)
    else:  # 1D — draw as a rug / strip
        y1d  = np.zeros(len(pts0))
        scat = ax.scatter(pts0[:, 0], y1d, **scatter_kw)
        ax.set_yticks([])

    if color_field is not None:
        fig.colorbar(scat, ax=ax, label=color_field,
                     pad=0.1 if ndim == 3 else 0.02, shrink=0.7)

    # Timestep label — text2D for 3-D axes, annotate for 2-D
    txt_kw = dict(fontsize=9, verticalalignment="top",
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    if ndim == 3:
        frame_text = ax.text2D(0.02, 0.97, "", transform=ax.transAxes, **txt_kw)
    else:
        frame_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, **txt_kw)

    # ── update ────────────────────────────────────────────────────────────────
    def _update(frame: int):
        pts = arrays[frame]
        c   = _get_colors(pts, color_field, active, vmin, vmax)

        if ndim == 3:
            scat._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        elif ndim == 2:
            scat.set_offsets(pts[:, :2])
        else:
            scat.set_offsets(np.column_stack([pts[:, 0], np.zeros(len(pts))]))

        if c is not None:
            scat.set_array(c)
        frame_text.set_text(f"step {keys[frame]}  |  {len(pts):,} pts")
        return scat, frame_text

    anim = animation.FuncAnimation(
        fig, _update,
        frames=n_frames,
        interval=interval,
        blit=False,
        repeat=True,
    )

    # ── save or show ──────────────────────────────────────────────────────────
    if save:
        ext = save.rsplit(".", 1)[-1].lower()
        if ext == "gif":
            writer = animation.PillowWriter(fps=max(1, 1000 // interval))
        elif ext == "mp4":
            writer = animation.FFMpegWriter(fps=max(1, 1000 // interval), bitrate=1800)
        else:
            raise ValueError(f"Unsupported format .{ext} — use .gif or .mp4")
        anim.save(save, writer=writer, dpi=dpi)
        print(f"Saved: {save}")
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()

    return anim


# ══════════════════════════════════════════════════════════════════════════════
#  CLI — handles both .hSpart and .npz inputs
# ══════════════════════════════════════════════════════════════════════════════

def save_as_npy(
    arrays: dict[str, np.ndarray],
    path: str,
) -> np.ndarray:
    """
    Stack per-timestep arrays into a single (T, N, D) array and save as .npy.

    Parameters
    ----------
    arrays : dict[str, ndarray]
        Output of extract_positions_from_hSpart — keys are timestep labels,
        values are (N, D) float arrays.  N must be the same across all steps.
    path : str
        Output path, e.g. "positions.npy".

    Returns
    -------
    np.ndarray  shape (T, N, D)
    """
    keys   = _sorted_keys(arrays)
    shapes = [arrays[k].shape for k in keys]
    if len(set(s[0] for s in shapes)) > 1:
        raise ValueError(
            "Particle count differs across timesteps — cannot stack into (T, N, D).\n"
            f"Counts: { {k: arrays[k].shape[0] for k in keys} }"
        )
    stacked = np.stack([arrays[k] for k in keys], axis=0)   # (T, N, D)
    np.save(path, stacked)
    T, N, D = stacked.shape
    print(f"Saved {path}  shape=({T}, {N}, {D})  →  T timesteps × N particles × {D} coords")
    return stacked


def main():
    parser = argparse.ArgumentParser(
        description="Extract and animate SPH particle positions from .hSpart or .npz files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sph_visualize.py data.hSpart                  # full 3-D animation
  python sph_visualize.py data.hSpart --drop z         # 2-D x/y scatter
  python sph_visualize.py data.hSpart --drop y z       # 1-D strip along x
  python sph_visualize.py data.hSpart --drop z --color x --save out.gif
  python sph_visualize.py positions.npz --drop z       # animate existing .npz
  python sph_visualize.py data.hSpart --inspect        # print HDF5 tree
  python sph_visualize.py data.hSpart --drop z --npy positions.npy   # save (T,N,2)
""",
    )
    parser.add_argument("input", help="Path to .hSpart or .npz file")
    parser.add_argument(
        "--drop", nargs="+", metavar="AXIS",
        choices=["x", "y", "z"], default=None,
        help="Axes to drop: x  y  z  (space-separated, e.g. --drop y z)",
    )
    parser.add_argument("--interval", type=int, default=100,
                        help="ms between frames (default: 100)")
    parser.add_argument("--s", type=float, default=2,
                        help="Point size (default: 2)")
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="Point opacity (default: 0.6)")
    parser.add_argument("--color", default=None,
                        choices=["x", "y", "z", "r"],
                        help="Colour by axis or radial distance 'r'")
    parser.add_argument("--cmap", default="plasma",
                        help="Matplotlib colormap (default: plasma)")
    parser.add_argument("--save", default=None,
                        help="Save animation: out.gif or out.mp4")
    parser.add_argument("--dpi", type=int, default=120,
                        help="DPI for saved output (default: 120)")
    parser.add_argument("--npz", default=None, metavar="PATH",
                        help="Also save extracted arrays to this .npz path")
    parser.add_argument("--npy", default=None, metavar="PATH",
                        help="Save stacked (T, N, D) array to this .npy path")
    parser.add_argument("--inspect", action="store_true",
                        help="Print HDF5 structure and exit (.hSpart only)")
    args = parser.parse_args()

    ext = args.input.rsplit(".", 1)[-1].lower()

    # ── inspect mode ─────────────────────────────────────────────────────────
    if args.inspect:
        if ext not in ("hspart", "h5", "hdf5"):
            parser.error("--inspect only works on HDF5 / .hSpart files")
        with h5py.File(args.input, "r") as f:
            inspect_structure(f)
        sys.exit(0)

    # ── load data ─────────────────────────────────────────────────────────────
    if ext == "npz":
        raw = np.load(args.input)
        keys   = _sorted_keys(raw)
        arrays = {k: np.asarray(raw[k], dtype=float) for k in keys}
        print(f"Loaded {len(arrays)} timestep(s) from {args.input}")
        # For .npz we apply drop inside animate_positions
        anim_data = arrays
        drop_for_anim = args.drop
    else:
        print(f"Extracting from {args.input} …")
        arrays = extract_positions_from_hSpart(args.input, drop_axes=args.drop)
        print(f"Found {len(arrays)} timestep(s).")
        for label, a in arrays.items():
            print(f"  {label:>14s}  →  {a.shape[0]:>8,} particles  dim={a.shape[1]}")
        if args.npz:
            np.savez_compressed(args.npz, **arrays)
            print(f"Saved arrays → {args.npz}")
        if args.npy:
            save_as_npy(arrays, args.npy)
        anim_data    = arrays
        drop_for_anim = None   # already applied during extraction

    # ── animate ───────────────────────────────────────────────────────────────
    animate_positions(
        anim_data,
        drop_axes=drop_for_anim,
        interval=args.interval,
        s=args.s,
        alpha=args.alpha,
        color_field=args.color,
        cmap=args.cmap,
        save=args.save,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()