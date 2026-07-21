"""
tests/test_wholecloud.py
------------------------
Regression: WholeCloudCorrector2D must reproduce the sim-validated whole-cloud
artifact (positions_model12_corrected.npy, produced 2026-07-15 by the
scratchpad correct_all.py and validated by an actual SPH re-simulation)
BIT-EXACTLY. If this fails, the corrector is no longer the validated procedure.

Needs the gitignored artifacts (SPH trajectory + reference output) on disk.

Run:
    .venv\\Scripts\\python.exe tests/test_wholecloud.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from inference.correctors import WholeCloudCorrector2D, WholeCloudCorrector2DConfig

CONFIG   = ROOT / 'src/configs/experiments/sph_tv/model12_wholecloud.yaml'
RAW      = ROOT / 'artifacts/inference/experiments/sph_tv/data/positions_without.npy'
REFERENCE = ROOT / 'artifacts/inference/experiments/sph_tv/for_sim/positions_model12_corrected.npy'
TIMESTEPS = (0, 300, 600, 1000)   # ordered frame + the three sim-validated starts
K = 5


def test_wholecloud_bit_exact():
    wc  = WholeCloudCorrector2D(WholeCloudCorrector2DConfig.from_yaml(str(CONFIG)))
    raw = np.load(RAW, mmap_mode='r')
    ref = np.load(REFERENCE, mmap_mode='r')
    for t in TIMESTEPS:
        out = wc.apply(np.asarray(raw[t]).astype(np.float32), k=K)
        assert np.array_equal(out, np.asarray(ref[t])), (
            f't={t}: output differs from the sim-validated artifact '
            f'(max|diff|={np.abs(out - np.asarray(ref[t])).max():.2e})')
        print(f't={t:4d}: bit-exact')


if __name__ == '__main__':
    if not (RAW.exists() and REFERENCE.exists()):
        sys.exit('artifacts missing — this test needs the SPH trajectory on disk')
    test_wholecloud_bit_exact()
    print('\nPASSED — WholeCloudCorrector2D reproduces the validated artifact')
