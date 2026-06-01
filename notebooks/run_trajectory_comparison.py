"""
Run shifted-grid comparison for every 100th timestep.
Runs two experiments:
  1. Normal ghost buffer    (ghost_factor=1.0) -> docs/images/trajectory/
  2. Tight ghost buffer     (ghost_factor=0.5) -> docs/images/trajectory_tight/
"""
import matplotlib; matplotlib.use('Agg')
import sys, numpy as np, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from inference.pipeline import Corrector, CorrectorConfig
from inference.visualization import plot_shifted_grid_comparison

ROOT = Path(__file__).parent.parent

pos_without = np.load(ROOT / 'inference/sph_data/positions_without.npy')
pos_with    = np.load(ROOT / 'inference/sph_data/positions.npy')
T           = pos_without.shape[0]
timesteps   = range(0, T, 100)

EXPERIMENTS = [
    ('inference/configs/grid_6x6.yaml',       'docs/images/trajectory',       'Normal ghost (factor=1.0)'),
    ('inference/configs/grid_6x6_tight.yaml', 'docs/images/trajectory_tight', 'Tight ghost  (factor=0.5)'),
]

for cfg_path, out_rel, label in EXPERIMENTS:
    cfg       = CorrectorConfig.from_yaml(str(ROOT / cfg_path))
    corrector = Corrector(cfg)
    out       = ROOT / out_rel
    out.mkdir(parents=True, exist_ok=True)

    print(f'\n=== {label} -> {out} ===')
    for t in timesteps:
        save = out / f'comparison_t{t:04d}.png'
        print(f'  t={t}', end='  ', flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)   # suppress ghost-factor warning
            plot_shifted_grid_comparison(
                pos_without[t].astype('float32'),
                pos_with[t].astype('float32'),
                corrector,
                timestep=t,
                save_path=str(save),
                show=False,
            )

    print(f'Done: {len(list(out.glob("*.png")))} figures in {out}')
