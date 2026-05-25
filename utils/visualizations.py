import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import cKDTree


def plot_poisson_disk(cloud, rd):
    tree = cKDTree(cloud)
    dists, _ = tree.query(cloud, k=2)
    nearest = dists[:, 1]
    colors = ['red' if d < rd else 'green' for d in nearest]

    dim = cloud.shape[1]

    if dim == 2:
        fig, ax = plt.subplots()
        for point, color in zip(cloud, colors):
            circle = patches.Circle(point, radius=rd / 2, fill=False,
                                    linestyle='--', linewidth=0.5, color=color, alpha=0.4)
            ax.add_patch(circle)
        ax.scatter(cloud[:, 0], cloud[:, 1], c=colors, s=10, zorder=3)
        ax.set_aspect('equal')
        ax.autoscale()
        ax.axis('on')
    elif dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=colors, s=10)
        ax.axis('on')

    plt.show()


def plot_comparison(noisy_clouds, corrected_clouds, rd, save_path):
    """2×n grid: top row = noisy input, bottom row = corrected output."""
    n = len(noisy_clouds)
    dim = noisy_clouds[0].shape[1]

    # Compute shared bounding box across all clouds so comparisons are fair
    all_points = np.concatenate(list(noisy_clouds) + list(corrected_clouds), axis=0)
    margin = rd
    if dim == 2:
        xlim = (all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
        ylim = (all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
        bbox = (xlim, ylim, None)
    else:
        xlim = (all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
        ylim = (all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
        zlim = (all_points[:, 2].min() - margin, all_points[:, 2].max() + margin)
        bbox = (xlim, ylim, zlim)

    if dim == 3:
        fig, axes = plt.subplots(2, n, figsize=(3 * n, 6),
                                 subplot_kw={'projection': '3d'}, squeeze=False)
    else:
        fig, axes = plt.subplots(2, n, figsize=(3 * n, 6), squeeze=False)

    for row, (clouds, label) in enumerate([(noisy_clouds, 'Noisy input'), (corrected_clouds, 'Corrected')]):
        axes[row, 0].set_ylabel(label, fontsize=10, fontweight='bold')
        for col, cloud in enumerate(clouds):
            _draw_cloud(axes[row, col], cloud, rd, dim, bbox)

    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def _draw_cloud(ax, cloud, rd, dim, bbox):
    tree = cKDTree(cloud)
    dists, _ = tree.query(cloud, k=2)
    colors = ['#e74c3c' if d < rd else '#2ecc71' for d in dists[:, 1]]

    xlim, ylim, zlim = bbox

    if dim == 2:
        for point, color in zip(cloud, colors):
            ax.add_patch(patches.Circle(point, radius=rd / 2, fill=False,
                                        linestyle='--', linewidth=0.6, color=color, alpha=0.35))
        ax.scatter(cloud[:, 0], cloud[:, 1], c=colors, s=15, zorder=3, edgecolors='none')
        ax.set_aspect('equal')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('#f8f9fa')
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#cccccc')
    else:
        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=colors, s=15, edgecolors='none')
        ax.set_xlim3d(xlim)
        ax.set_ylim3d(ylim)
        ax.set_zlim3d(zlim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
