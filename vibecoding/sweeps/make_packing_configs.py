"""
Generate the packing-sweep configs: N=49, rd/spacing from 0.80 up to the packing
limit 1.00, where the only feasible configuration is the exact lattice.

Every length is pinned to rd so the rungs differ ONLY in how rigid the constraint is:
    rd              = frac * spacing            (spacing = 1/7)
    noise_scale_max = 0.6 * rd                  (constant RELATIVE disorder)
    lambda1         = 1 / rd
    lambda2         = 0.1 * lambda1 / (N-1)
lambda3 is set to hold the symmetry:illegality ratio at its rd/s=0.98 value (measured,
see the table below). max_displacement is deliberately NOT scaled -- the output bound
is not part of the difficulty axis, so the model configs are reused unchanged and the
architectures are byte-identical across rungs.

    .venv\\Scripts\\python.exe vibecoding/sweeps/make_packing_configs.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'src/configs/training/ablations/packing'
N = 49
SPACING = 1.0 / 7

# frac -> (lambda3, measured clean-lattice jitter, measured input illegal%)
RUNGS = {
    0.80: (0.1795, 0.01429, 1.8),
    0.90: (0.2274, 0.00714, 2.7),
    0.95: (0.2532, 0.00357, 3.4),
    1.00: (0.2814, 0.00000, 4.6),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for frac, (lam3, jitter, ill) in RUNGS.items():
        rd = round(frac * SPACING, 6)
        tag = f'{frac:.2f}'.replace('.', 'p')
        lam1 = round(1.0 / rd, 4)
        lam2 = round(0.1 * lam1 / (N - 1), 6)

        (OUT / f'dataset_config_pack{tag}.yaml').write_text(
            f'# Packing sweep, rd/spacing = {frac:.2f} (spacing = 1/7 = 0.142857).\n'
            f'# Clean-lattice jitter {jitter:.5f}; {ill:.1f}% of input pairs illegal.\n'
            + ('# PACKING LIMIT: jitter is exactly 0, so the only feasible configuration\n'
               '# is the exact lattice, whose KG is exactly 0. Maximum coordination demand.\n'
               if frac >= 1.0 else '')
            + f'points_per_cloud: {N}\n'
              f'dim: 2\n'
              f'rd: {rd}\n'
              f'seed: 42\n'
              f'periodic: true\n'
              f'noise_scale_min: 0.0\n'
              f'noise_scale_max: {round(0.6 * rd, 6)}   # 0.6 * rd, constant relative disorder\n')

        (OUT / f'loss_config_pack{tag}.yaml').write_text(
            f'name: rdsph_loss\n\n'
            f'# Packing sweep, rd/spacing = {frac:.2f}. lambda3 holds the\n'
            f'# symmetry:illegality ratio at its rd/s=0.98 value so the rungs stay\n'
            f'# comparable; it varies only mildly (0.18 - 0.28) across the sweep.\n'
            f'params:\n'
            f'  lambda1: {lam1}\n'
            f'  lambda1_quad: 0\n'
            f'  lambda2: {lam2}\n'
            f'  lambda3: {lam3}\n\n'
            f'  h_factor: 2.0\n'
            f'  box:      1.0\n')
        print(f'rd/s {frac:.2f}  rd={rd}  lam1={lam1}  lam2={lam2}  lam3={lam3}')
    print(f'\n-> {OUT}')


if __name__ == '__main__':
    main()
