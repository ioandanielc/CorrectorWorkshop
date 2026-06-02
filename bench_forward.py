"""Benchmark: sparse vs dense model forward, CPU vs GPU."""
import time, numpy as np, torch, sys, torch.nn as nn
sys.path.insert(0, '.')
from inference.pipeline.corrector import Corrector, CorrectorConfig
from inference.pipeline.pbc import build_ghost_tile
from inference.pipeline.tiling import iter_tiles


def build_batch(corrector, pos):
    ghost_w = corrector.tiling.ghost_width(corrector.cfg.rd_test)
    PAD = np.float32(1e3)
    tiles = []
    for tlo, thi in iter_tiles(corrector.tiling):
        ext, ic, oi = build_ghost_tile(pos, tlo, thi, ghost_w, corrector.cfg.domain)
        if ic.sum() == 0:
            continue
        pts_s = ext * corrector.scale
        x_inv, _, rev = corrector.processor.make_invariant(pts_s[None])
        tiles.append((x_inv, pts_s, ic, oi, rev))
    max_N = max(t[0].shape[1] for t in tiles)
    pad_offsets = np.arange(max_N, dtype=np.float32) * 0.2
    batch_x = np.empty((len(tiles), max_N, 2), dtype=np.float32)
    batch_x[:, :, 0] = PAD + pad_offsets
    batch_x[:, :, 1] = PAD
    for i, (x_inv, *_) in enumerate(tiles):
        n = x_inv.shape[1]
        batch_x[i, :n] = x_inv[0]
    return tiles, torch.tensor(batch_x)


class DenseModel(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, rd=None):
        B, N, D = x.shape
        xi = x.unsqueeze(2); xj = x.unsqueeze(1)
        rel_pos  = xi - xj
        dist     = rel_pos.norm(dim=-1, keepdim=True)
        violation = torch.relu(rd - dist)
        edge_feat = torch.cat([rel_pos, dist, violation], dim=-1)
        edge_emb  = self.m.edge_mlp(edge_feat)
        eye       = torch.eye(N, dtype=torch.bool, device=x.device)
        edge_emb  = edge_emb  * (~eye).unsqueeze(0).unsqueeze(-1)
        violation = violation * (~eye).unsqueeze(0).unsqueeze(-1)
        viol_sum  = violation.sum(dim=2, keepdim=True)
        weights   = violation / (viol_sum + 1e-8)
        agg       = (edge_emb * weights).sum(dim=2)
        return torch.tanh(self.m.output_mlp(agg)) * self.m.max_disp


def bench(model, x, rd, device, n=30):
    x  = x.to(device)
    rd = rd.to(device)
    model = model.to(device)
    with torch.no_grad():
        for _ in range(5):
            model(x, rd=rd)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            model(x, rd=rd)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n


cfg = CorrectorConfig.from_yaml('inference/configs/grid_6x6.yaml')
cfg.device = 'cpu'
corrector = Corrector(cfg)
pos = np.load('inference/sph_data/positions_without.npy')[300].astype(np.float32)
_, x = build_batch(corrector, pos)
rd   = corrector.rd_t

sparse_model = corrector.model
dense_model  = DenseModel(corrector.model)

cpu_sp = bench(sparse_model, x, rd, torch.device('cpu'))
gpu_sp = bench(sparse_model, x, rd, torch.device('cuda'))
cpu_dn = bench(dense_model,  x, rd, torch.device('cpu'))
gpu_dn = bench(dense_model,  x, rd, torch.device('cuda'))

print(f'Batch: {tuple(x.shape)}')
print()
print(f'{"":20s}  {"ms":>8s}')
print(f'{"CPU sparse":20s}  {cpu_sp*1000:>8.2f}')
print(f'{"GPU sparse":20s}  {gpu_sp*1000:>8.2f}')
print(f'{"CPU dense":20s}  {cpu_dn*1000:>8.2f}')
print(f'{"GPU dense":20s}  {gpu_dn*1000:>8.2f}')
print()
print(f'CPU sparse vs CPU dense : {cpu_dn/cpu_sp:5.1f}x  (sparse wins on CPU)')
print(f'GPU dense  vs GPU sparse: {gpu_sp/gpu_dn:5.1f}x  (dense wins on GPU?)')
print(f'GPU dense  vs CPU sparse: {cpu_sp/gpu_dn:5.1f}x  (best GPU vs best CPU)')
