"""End-to-end timing for corrector.apply(k=5) on CPU and CUDA."""
import time, numpy as np, torch, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from inference.pipeline.corrector import Corrector, CorrectorConfig

pos = np.load('inference/sph_data/positions_without.npy')[1000].astype('float32')
print(f'N={len(pos)}  (positions_without t=1000)')
print()

configs = [
    ('inference/configs/grid_6x6.yaml',       '6x6  factor=1.0'),
    ('inference/configs/grid_6x6_tight.yaml',  '6x6  factor=0.5'),
]
devices = ['cpu', 'cuda']
N_REPS  = 20

for cfg_path, label in configs:
    for dev in devices:
        cfg = CorrectorConfig.from_yaml(cfg_path)
        cfg.device = dev
        c = Corrector(cfg)
        for _ in range(3):
            c.apply(pos, k=5)
        if dev == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_REPS):
            c.apply(pos, k=5)
        if dev == 'cuda':
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / N_REPS * 1000
        print(f'{label}  {dev:4s}  K=5: {ms:7.1f} ms')
    print()
