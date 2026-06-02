"""
Run shifted-grid comparison for every 100th timestep.
Measures timing on both CPU and GPU, plus peak GPU VRAM.
Runs two experiments:
  1. Normal ghost buffer    (ghost_factor=1.0) -> docs/images/trajectory/
  2. Tight ghost buffer     (ghost_factor=0.5) -> docs/images/trajectory_tight/
"""
import matplotlib; matplotlib.use('Agg')
import sys, time, numpy as np, warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from inference.pipeline import Corrector, CorrectorConfig
from inference.visualization import plot_shifted_grid_comparison

ROOT      = Path(__file__).parent.parent
N_TIMING  = 3     # reps for stable timing estimate

pos_without = np.load(ROOT / 'inference/sph_data/positions_without.npy')
pos_with    = np.load(ROOT / 'inference/sph_data/positions.npy')
T           = pos_without.shape[0]
timesteps   = range(0, T, 100)

EXPERIMENTS = [
    ('inference/configs/grid_6x6.yaml',       'docs/images/trajectory',        'Normal ghost (factor=1.0)'),
    ('inference/configs/grid_6x6_tight.yaml', 'docs/images/trajectory_tight',  'Tight ghost  (factor=0.5)'),
]

STRATEGIES = [
    ('k1',      lambda c, p: c.apply(p, k=1)),
    ('k2',      lambda c, p: c.apply(p, k=2)),
    ('shifted', lambda c, p: c.apply_shifted_grid(p, shift_fraction=0.5)),
    ('k5',      lambda c, p: c.apply(p, k=5)),
]


def measure_perf(corrector_cpu, corrector_gpu, pts):
    """Return perf_data dict with cpu_ms, gpu_ms, vram_mb per strategy."""
    perf = {}
    for key, fn in STRATEGIES:
        # CPU warmup + timing
        fn(corrector_cpu, pts)
        t0 = time.perf_counter()
        for _ in range(N_TIMING):
            fn(corrector_cpu, pts)
        cpu_ms = (time.perf_counter() - t0) / N_TIMING * 1000

        # GPU warmup, then timing + VRAM
        fn(corrector_gpu, pts)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(N_TIMING):
            fn(corrector_gpu, pts)
        torch.cuda.synchronize()
        gpu_ms  = (time.perf_counter() - t0) / N_TIMING * 1000
        vram_mb = torch.cuda.max_memory_allocated() / 1024 ** 2

        perf[key] = dict(cpu_ms=cpu_ms, gpu_ms=gpu_ms, vram_mb=vram_mb)
    return perf


for cfg_path, out_rel, label in EXPERIMENTS:
    cfg_gpu        = CorrectorConfig.from_yaml(str(ROOT / cfg_path))
    cfg_gpu.device = 'cuda'
    cfg_cpu        = CorrectorConfig.from_yaml(str(ROOT / cfg_path))
    cfg_cpu.device = 'cpu'

    corrector_gpu = Corrector(cfg_gpu)
    corrector_cpu = Corrector(cfg_cpu)
    out = ROOT / out_rel
    out.mkdir(parents=True, exist_ok=True)

    print(f'\n=== {label} -> {out} ===')
    for t in timesteps:
        save = out / f'comparison_t{t:04d}.png'
        pts  = pos_without[t].astype('float32')
        pts_tv = pos_with[t].astype('float32')
        print(f'  t={t}', end='  ', flush=True)

        print('timing...', end=' ', flush=True)
        perf_data = measure_perf(corrector_cpu, corrector_gpu, pts)
        k5_gpu = perf_data['k5']['gpu_ms']
        k5_cpu = perf_data['k5']['cpu_ms']
        print(f'K=5 GPU={k5_gpu:.1f}ms CPU={k5_cpu:.0f}ms', end='  ', flush=True)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            plot_shifted_grid_comparison(
                pts, pts_tv,
                corrector_gpu,
                timestep=t,
                save_path=str(save),
                show=False,
                perf_data=perf_data,
            )

    print(f'Done: {len(list(out.glob("*.png")))} figures in {out}')
