"""
tests/test_sparse_paths.py
--------------------------
Every architecture deployed on a whole cloud reaches N=2500 through a
`forward_sparse` that must agree with its own dense `forward`. Nothing else
checks that: `test_wholecloud.py` guards model12's *output* against a stored
artifact, but gns and model12_ablate gained sparse paths later and had no guard
at all — their equivalence was only ever verified in throwaway scripts.

Agreement is to float32 round-off, not bit-exact: the sparse path sums messages
in a different order.

    .venv\\Scripts\\python.exe tests/test_sparse_paths.py
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

BOX, CUTOFF, TOL = 1.0, 0.286, 1e-5
COMMON = dict(hidden_dim=128, num_layers=4, norm='layer', activation='GELU',
              max_displacement=0.168)

CASES = [
    ('model12', 'models.architectures.model12.model12', dict(COMMON)),
    ('gns', 'models.architectures.gns.gns', dict(COMMON, hidden_dim=107)),
    # every radius-graph rung of the ablation model; knn has no fixed edge list
    ('ablate/baseline', 'models.architectures.model12_ablate.model12_ablate',
     dict(COMMON, graph='radius', weight='kernel', aggregation='wsum', periodic=True)),
    ('ablate/maxagg', 'models.architectures.model12_ablate.model12_ablate',
     dict(COMMON, graph='radius', weight='kernel', aggregation='max', periodic=True)),
    ('ablate/nonorm', 'models.architectures.model12_ablate.model12_ablate',
     dict(COMMON, graph='radius', weight='kernel', aggregation='sum', periodic=True)),
    ('ablate/nokernel', 'models.architectures.model12_ablate.model12_ablate',
     dict(COMMON, graph='radius', weight='learned', aggregation='wsum', periodic=True)),
]


def edge_index(pts, box, cutoff):
    """All ordered pairs within the cutoff, both directions, no self-loops."""
    pairs = cKDTree(pts, boxsize=box).query_pairs(cutoff, output_type='ndarray')
    return torch.tensor(np.stack([np.concatenate([pairs[:, 0], pairs[:, 1]]),
                                  np.concatenate([pairs[:, 1], pairs[:, 0]])]),
                        dtype=torch.long)


def main():
    failures = []
    for name, module_path, cfg in CASES:
        mod = importlib.import_module(module_path)
        torch.manual_seed(0)
        net = mod.CorrectorModel(cfg, input_dim=2, initialization='xavier_uniform').eval()

        for n in (60, 250, 600):
            pts = np.random.default_rng(n).random((n, 2)).astype(np.float32) * BOX
            x, ei = torch.tensor(pts), edge_index(pts, BOX, CUTOFF)
            with torch.no_grad():
                dense = net(x.unsqueeze(0), rd=torch.tensor(CUTOFF), box=BOX)[0]
                sparse = net.forward_sparse(x, ei, rd=torch.tensor(CUTOFF), box=BOX)
            diff = (dense - sparse).abs().max().item()
            status = 'ok' if diff < TOL else 'FAIL'
            if diff >= TOL:
                failures.append(f'{name} N={n}: {diff:.2e}')
            print(f'{name:18s} N={n:4d}  E={ei.shape[1]:6d}  max|dense-sparse| {diff:.2e}  {status}')

    print()
    if failures:
        print('FAILED\n  ' + '\n  '.join(failures))
        sys.exit(1)
    print(f'PASSED — every sparse path agrees with its dense forward (tol {TOL:g})')


if __name__ == '__main__':
    main()
