"""
Two argument figures for the paper.

  k_curve.png          |KG| and illegal% against correction passes on the real
                       trajectory. Carries three findings at once: k=1 is worse than
                       no correction, |KG| never floors, and illegal% turns around at
                       k=8 — which is where the violation<->symmetry trade-off actually
                       lives (it was previously mis-reported as a "KG floor").

  benchmark_vs_deployment.png
                       Every architecture's rank on the N=49 synthetic benchmark against
                       its rank on the N=2500 deployment task. Lines crossing = the
                       benchmark misranks. Two do.

Numbers are hard-coded from paper/results.csv rather than recomputed: these are
published values and the figure must not silently drift from the table.

    .venv\\Scripts\\python.exe vibecoding/visualizations/paper_curves.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / 'outputs'

# k -> (|KG|, illegal%) on the real trajectory, production model12, t >= 300
K_SWEEP = {0: (0.3331, 99.4), 1: (0.4661, 97.1), 2: (0.2710, 92.0), 3: (0.1860, 87.9),
           5: (0.1269, 82.0), 8: (0.0984, 79.2), 12: (0.0846, 80.4), 16: (0.0796, 82.3),
           20: (0.0769, 83.6), 30: (0.0715, 86.6), 40: (0.0675, 88.4)}

# architecture -> (N=49 |KG|, N=2500 trajectory |KG|)
ARCH = {
    'model12':   (0.0216, 0.1269),
    'GNS':       (0.0205, 0.1319),
    'maxagg':    (0.0106, 0.1495),
    'nonorm':    (0.0196, 0.1204),
    'PointNet':  (0.2258, 0.3338),
}
INK, ACCENT, WARN, MUTED = '#1b1f24', '#2d6a8e', '#b3524a', '#8a938c'


def k_curve():
    ks = sorted(K_SWEEP)
    kg = [K_SWEEP[k][0] for k in ks]
    ill = [K_SWEEP[k][1] for k in ks]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(ks, kg, 'o-', color=ACCENT, lw=2, ms=5, label='|KG|  (SPH consistency)')
    ax.axhline(K_SWEEP[0][0], color=MUTED, ls=':', lw=1.4)
    ax.text(1.6, K_SWEEP[0][0] + 0.008, 'raw (no correction)', va='bottom',
            color=MUTED, fontsize=8.5)
    ax.annotate('k=1 is WORSE\nthan no correction', xy=(1, K_SWEEP[1][0]),
                xytext=(2.2, 0.435), color=WARN, fontsize=9,
                arrowprops=dict(arrowstyle='->', color=WARN, lw=1.2))
    ax.annotate('shipped k=5', xy=(5, K_SWEEP[5][0]), xytext=(7.5, 0.20),
                fontsize=9, color=INK,
                arrowprops=dict(arrowstyle='->', color=INK, lw=1.2))
    ax.set_xlabel('correction passes  k')
    ax.set_ylabel('mean |KG|', color=ACCENT)
    ax.tick_params(axis='y', labelcolor=ACCENT)
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xticks(ks); ax.set_xticklabels(ks)
    ax.set_ylim(0, 0.50)

    ax2 = ax.twinx()
    ax2.plot(ks, ill, 's--', color=WARN, lw=1.6, ms=4, alpha=0.85,
             label='illegal pairs %')
    ax2.set_ylim(74, 103)          # headroom so the annotation clears the curves
    ax2.scatter([8], [K_SWEEP[8][1]], s=150, facecolors='none',
                edgecolors=WARN, lw=2, zorder=5)
    ax2.annotate('illegal% minimum at k=8 — past here,\n'
                 'symmetry improves at legality’s expense',
                 xy=(8, K_SWEEP[8][1]), xytext=(2.1, 76.2), fontsize=8.5, color=WARN,
                 arrowprops=dict(arrowstyle='->', color=WARN, lw=1.2))
    ax2.set_ylabel('illegal pairs %', color=WARN)
    ax2.tick_params(axis='y', labelcolor=WARN)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right', frameon=False, fontsize=9)
    ax.set_title('|KG| never floors; the trade-off lives in k\n'
                 'real SPH trajectory, N=2500, t >= 300', fontsize=11)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / 'k_curve.png', dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f'-> {OUT / "k_curve.png"}')


def benchmark_vs_deployment():
    """Slope chart: rank on the benchmark vs rank on the real task."""
    bench = sorted(ARCH, key=lambda a: ARCH[a][0])
    deploy = sorted(ARCH, key=lambda a: ARCH[a][1])

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for a in ARCH:
        y0, y1 = bench.index(a), deploy.index(a)
        crossed = y0 != y1
        is_ours = a == 'model12'
        col = INK if is_ours else (WARN if crossed else MUTED)
        ax.plot([0, 1], [y0, y1], '-o', color=col, lw=2.6 if is_ours else 1.8,
                ms=7 if is_ours else 5, zorder=3 if is_ours else 2)
        ax.text(-0.04, y0, f'{a}  {ARCH[a][0]:.4f}', ha='right', va='center',
                fontsize=9.5, color=col, fontweight='bold' if is_ours else 'normal')
        ax.text(1.04, y1, f'{ARCH[a][1]:.4f}  {a}', ha='left', va='center',
                fontsize=9.5, color=col, fontweight='bold' if is_ours else 'normal')

    ax.set_xlim(-0.55, 1.55); ax.set_ylim(len(ARCH) - 0.5, -0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['N=49 synthetic benchmark\n(|KG|, lower is better)',
                        'N=2500 real trajectory\n(|KG|, lower is better)'], fontsize=10)
    ax.set_yticks([])
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.set_title('The small-N benchmark misranks architectures against deployment\n'
                 'model12 places 4th of 5 on the benchmark and 2nd on the real task',
                 fontsize=11)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / 'benchmark_vs_deployment.png', dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f'-> {OUT / "benchmark_vs_deployment.png"}')


if __name__ == '__main__':
    k_curve()
    benchmark_vs_deployment()
