"""
Generate the cutoff_rd (kernel width) sweep configs.

`cutoff_rd` is the support of the fixed proximity kernel, fixed at 2*dx = h all
session because that is the SPH smoothing length. It has never been ablated — and
now that the fixed kernel is the candidate mechanism for size transfer, its width
is the obvious knob.

Everything else is held at the production recipe. Reach is L*cutoff_rd, so the
receptive field moves with it, which is the confound to state: this sweep varies
kernel width AND reach together. Separating them would need L to compensate
(L*cutoff_rd constant), which is a second sweep, not run.

    .venv\\Scripts\\python.exe vibecoding/sweeps/make_cutoff_configs.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'src/configs/training/ablations/cutoff'
SPACING = 1.0 / 7          # N=49 lattice spacing
L = 4

# multiples of the lattice spacing; 2.0 is production (= h, the SPH smoothing length)
WIDTHS = {'1p5': 1.5, '2p0': 2.0, '3p0': 3.0}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, mult in WIDTHS.items():
        cutoff = round(mult * SPACING, 6)
        reach = L * cutoff
        note = ('production setting (= h, the SPH smoothing length)' if mult == 2.0
                else f'{mult}x the lattice spacing')
        (OUT / f'model_config_cutoff_{tag}.yaml').write_text(
            f'# cutoff_rd sweep: {mult} * spacing — {note}.\n'
            f'# Kernel support {cutoff}; reach = {L} * cutoff = {reach:.4f} '
            f'({reach:.2f} of the unit box).\n'
            f'# CONFOUND: reach moves with the width, since reach = L * cutoff_rd.\n'
            f'# Separating the two needs L adjusted to hold reach constant — not run.\n'
            f'architecture: iterative_message_passing_corrector\n'
            f'model_file: models/architectures/model12/model12\n\n'
            f'hidden_dim: 128\n'
            f'num_layers: {L}\n'
            f'norm: layer\n'
            f'activation: GELU\n'
            f'max_displacement: 0.168\n\n'
            f'cutoff_rd: {cutoff}\n')
        print(f'{tag}: cutoff_rd={cutoff}  ({mult}x spacing)  reach={reach:.4f}')
    print(f'\n-> {OUT}')


if __name__ == '__main__':
    main()
