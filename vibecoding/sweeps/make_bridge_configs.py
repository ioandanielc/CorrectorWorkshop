"""
Generate the architecture-bridge configs: model12 -> DGCNN, one mechanism at a time.

Every rung runs through `models/architectures/model12_ablate/model12_ablate.py` so the
code path is identical and only the flags differ. The baseline rung is verified
bit-identical to production `model12.py` (max|diff| = 0.0), and every rung is
parameter-matched to 350,594 within 0.3%.

    .venv\\Scripts\\python.exe vibecoding/sweeps/make_bridge_configs.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'src/configs/training/ablations/bridge'

RUNGS = {
    'baseline': (dict(graph='radius', weight='kernel', aggregation='wsum', periodic=True),
                 'Production model12, reached through the ablation code path. Verified\n'
                 '# bit-identical to models/architectures/model12/model12.py.'),
    'nokernel': (dict(graph='radius', weight='learned', aggregation='wsum', periodic=True),
                 'Fixed (1-q^2)^2 SPH kernel -> a learned linear gate on the edge\n'
                 '# features. Isolates what writing the kernel into the wiring is worth.\n'
                 '# The richly-learned version of this axis is the GNS baseline.'),
    'nonorm':   (dict(graph='radius', weight='kernel', aggregation='sum', periodic=True),
                 'Kernel-weighted MEAN -> unnormalised SUM. Isolates per-particle\n'
                 '# normalisation, which is what makes node states independent of\n'
                 '# neighbour count and is suspected to drive size transfer.'),
    'maxagg':   (dict(graph='radius', weight='kernel', aggregation='max', periodic=True),
                 'Weighted sum -> MAX. The sharpest predicted failure: KG symmetry is a\n'
                 '# statement that neighbour directions cancel, and a max cannot express\n'
                 '# cancellation. Expect KG to degrade far more than violations.'),
    'knngraph': (dict(graph='knn', weight='kernel', aggregation='wsum', periodic=True),
                 'Radius ball -> k nearest, rebuilt in feature space after round 0.\n'
                 '# Isolates "physical cutoff" from "fixed neighbour count".'),
    'noperiod': (dict(graph='radius', weight='kernel', aggregation='wsum', periodic=False),
                 'Minimum image disabled. Isolates the value of wrap geometry; the model\n'
                 '# now sees the torus as a bounded box and misses every seam pair.'),
    'dgcnnmech': (dict(graph='knn', weight='learned', aggregation='max', periodic=True),
                  'All three mechanisms swapped at once = DGCNN\'s mechanism set inside\n'
                  '# model12\'s code path. Bridges to the real DGCNN baseline and shows\n'
                  '# whether the rungs compose.'),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (flags, blurb) in RUNGS.items():
        (OUT / f'model_config_bridge_{name}.yaml').write_text(
            f'# Architecture bridge rung: {name}.\n'
            f'# {blurb}\n'
            f'#\n'
            f'# Parameter-matched to model12 (350,594) within 0.3%; run with the\n'
            f'# production dataset/loss/trainer configs so only the mechanism differs.\n'
            f'architecture: model12_ablate\n'
            f'model_file: models/architectures/model12_ablate/model12_ablate\n\n'
            f'hidden_dim: 128\n'
            f'num_layers: 4\n'
            f'norm: layer\n'
            f'activation: GELU\n'
            f'max_displacement: 0.168\n'
            f'cutoff_rd: 0.286\n\n'
            f'graph: {flags["graph"]}\n'
            f'weight: {flags["weight"]}\n'
            f'aggregation: {flags["aggregation"]}\n'
            f'periodic: {str(flags["periodic"]).lower()}\n'
            + ('k: 12                     # matched to the measured radius-graph degree 11.6\n'
               if flags['graph'] == 'knn' else ''))
        print(f'{name:11s} graph={flags["graph"]:6s} weight={flags["weight"]:7s} '
              f'agg={flags["aggregation"]:4s} periodic={flags["periodic"]}')
    print(f'\n-> {OUT}')


if __name__ == '__main__':
    main()
