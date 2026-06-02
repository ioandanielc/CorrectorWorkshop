"""
Profile corrector.apply(k=5) phase-by-phase across configs and devices.
Generates docs/profile_report.html.
"""
import time, warnings, sys
from pathlib import Path

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from inference.pipeline.corrector import Corrector, CorrectorConfig

# ── Config ────────────────────────────────────────────────────────────────────
CONFIGS = [
    ('inference/configs/grid_6x6.yaml',       '6×6 factor=1.0'),
    ('inference/configs/grid_6x6_tight.yaml',  '6×6 factor=0.5'),
]
DEVICES  = ['cpu', 'cuda']
K        = 5
N_WARMUP = 5
N_REPS   = 40
DATA_T   = 1000

PHASES = ['1a  tile membership (vec)', '1b  slot + make_invariant', '2  H2D+GPU fwd+D2H', '3  unrotate + scatter']
COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

# ── Data ──────────────────────────────────────────────────────────────────────
pos_all = np.load('inference/sph_data/positions_without.npy')
pos = pos_all[DATA_T].astype('float32')


def profile_one(cfg_path: str, device: str) -> dict:
    """Returns per-phase timings (ms) averaged over N_REPS, for K passes total."""
    cfg = CorrectorConfig.from_yaml(cfg_path)
    cfg.device = device
    c = Corrector(cfg)

    n_tiles = c._tile_lo.shape[0]
    PAD = np.float32(1e3)
    accum = np.zeros(4)   # phase totals
    max_N = 0; n_pairs = 0

    for rep in range(N_WARMUP + N_REPS):
        pts = pos.copy()
        rep_times = np.zeros(4)

        for _k in range(K):
            N = len(pts)

            # ── Phase 1a: vectorised tile membership ─────────────────────────
            t0 = time.perf_counter()
            t_idx, p_idx, pos_pairs, core_pairs = c._phase1a(pts)
            rep_times[0] += time.perf_counter() - t0

            # ── Phase 1b: slot assignment + batch build + make_invariant ─────
            t0 = time.perf_counter()
            sort_ord   = np.argsort(t_idx, kind='stable')
            t_sorted   = t_idx[sort_ord]
            tile_start = np.searchsorted(t_sorted, np.arange(n_tiles))
            slot_sorted = np.arange(sort_ord.size) - tile_start[t_sorted]
            slot_idx    = np.empty(sort_ord.size, dtype=np.int32)
            slot_idx[sort_ord] = slot_sorted
            cur_max_N  = int(np.bincount(t_idx, minlength=n_tiles).max())
            pad_x = PAD + np.arange(cur_max_N, dtype=np.float32) * 0.2
            batch_pts_s    = np.empty((n_tiles, cur_max_N, 2), dtype=np.float32)
            batch_pts_s[:, :, 0] = pad_x; batch_pts_s[:, :, 1] = PAD
            batch_is_core  = np.zeros((n_tiles, cur_max_N), dtype=bool)
            batch_orig_idx = np.zeros((n_tiles, cur_max_N), dtype=np.int32)
            real_mask_np   = np.zeros((n_tiles, cur_max_N), dtype=np.float32)
            pts_s = pos_pairs * c.scale
            batch_pts_s   [t_idx, slot_idx] = pts_s
            batch_is_core [t_idx, slot_idx] = core_pairs
            batch_orig_idx[t_idx, slot_idx] = p_idx
            real_mask_np  [t_idx, slot_idx] = 1.0
            N_real_safe = np.maximum(real_mask_np.sum(axis=1), 1.0)
            mean_batch  = ((batch_pts_s * real_mask_np[:, :, None]).sum(axis=1)
                           / N_real_safe[:, None])
            centered    = batch_pts_s - mean_batch[:, None]
            centered_r  = centered * real_mask_np[:, :, None]
            cov         = (np.matmul(centered_r.transpose(0, 2, 1), centered_r)
                           / N_real_safe[:, None, None])
            _, eigvecs  = np.linalg.eigh(cov)
            eigvecs     = np.ascontiguousarray(eigvecs[:, :, ::-1])
            x_inv       = np.matmul(centered, eigvecs)
            rep_times[1] += time.perf_counter() - t0

            # ── Phase 2: H2D + GPU forward + D2H ────────────────────────────
            t0 = time.perf_counter()
            x_t = torch.tensor(x_inv, dtype=torch.float32, device=c.device)
            with torch.no_grad():
                disp = c.model(x_t, rd=c.rd_t).cpu().numpy()
            if device == 'cuda':
                torch.cuda.synchronize()
            rep_times[2] += time.perf_counter() - t0

            # ── Phase 3: un-rotate (matmul) + direct scatter ─────────────────
            t0 = time.perf_counter()
            disp_orig = np.matmul(disp, eigvecs.transpose(0, 2, 1)) / c.scale
            core_t, core_s = np.where(batch_is_core)
            orig_indices   = batch_orig_idx[core_t, core_s]
            disps_core     = disp_orig[core_t, core_s]
            displacement   = np.empty_like(pts)
            displacement[orig_indices] = disps_core
            pts = (pts + displacement) % c.cfg.domain
            rep_times[3] += time.perf_counter() - t0

            if rep >= N_WARMUP:
                max_N  = max(max_N, cur_max_N)
                n_pairs = len(t_idx)

        if rep >= N_WARMUP:
            accum += rep_times

    ms = accum / N_REPS * 1000
    return {
        'phases_ms': ms.tolist(),
        'total_ms':  float(ms.sum()),
        'per_k_ms':  float(ms.sum() / K),
        'max_N':     max_N,
        'n_pairs':   n_pairs,
        'n_tiles':   n_tiles,
    }


# ── Run all profiles ──────────────────────────────────────────────────────────
results = {}
for cfg_path, label in CONFIGS:
    for device in DEVICES:
        key = f'{label} / {device}'
        print(f'Profiling: {key} ...', flush=True)
        results[key] = profile_one(cfg_path, device)
        r = results[key]
        print(f'  Total: {r["total_ms"]:.1f} ms  |  per-step: {r["per_k_ms"]:.2f} ms')

# ── Build HTML ────────────────────────────────────────────────────────────────
labels   = list(results.keys())
n_bars   = len(labels)

fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.65, 0.35],
    subplot_titles=[
        f'Phase breakdown — K={K} passes, N=2500, t={DATA_T}',
        'Total time (ms)'
    ],
)

# Stacked bar — one trace per phase
for ph_i, (ph_name, color) in enumerate(zip(PHASES, COLORS)):
    vals = [results[lbl]['phases_ms'][ph_i] for lbl in labels]
    fig.add_trace(go.Bar(
        name=ph_name,
        x=labels,
        y=vals,
        marker_color=color,
        text=[f'{v:.2f}' for v in vals],
        textposition='inside',
        insidetextanchor='middle',
    ), row=1, col=1)

# Total bar (single color, annotated with per-step time)
totals   = [results[lbl]['total_ms']  for lbl in labels]
per_step = [results[lbl]['per_k_ms']  for lbl in labels]
fig.add_trace(go.Bar(
    name='Total',
    x=labels,
    y=totals,
    marker_color='#8172B3',
    text=[f'{t:.1f} ms<br>({s:.2f}/step)' for t, s in zip(totals, per_step)],
    textposition='outside',
    showlegend=False,
), row=1, col=2)

fig.update_layout(
    barmode='stack',
    height=520,
    width=1100,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    margin=dict(t=80, b=120),
    font=dict(size=12),
    title=dict(
        text=f'Corrector Phase Profiler — K={K}, N={len(pos)}, t={DATA_T}',
        font=dict(size=16),
    ),
)
fig.update_yaxes(title_text='Time (ms)', row=1, col=1)
fig.update_yaxes(title_text='Time (ms)', row=1, col=2)
fig.update_xaxes(tickangle=-25, row=1, col=1)
fig.update_xaxes(tickangle=-25, row=1, col=2)

# ── Summary table ─────────────────────────────────────────────────────────────
rows_html = ''
for lbl in labels:
    r = results[lbl]
    ph = r['phases_ms']
    total = r['total_ms']
    pct   = [p/total*100 for p in ph]
    rows_html += f'''
    <tr>
      <td>{lbl}</td>
      {"".join(f"<td>{p:.2f} <span class='pct'>({q:.0f}%)</span></td>" for p,q in zip(ph,pct))}
      <td><b>{total:.1f}</b></td>
      <td>{r["per_k_ms"]:.2f}</td>
      <td>{r["max_N"]}</td>
      <td>{r["n_pairs"]}</td>
    </tr>'''

table_html = f'''
<table>
  <thead>
    <tr>
      <th>Config / Device</th>
      {"".join(f"<th>{p}</th>" for p in PHASES)}
      <th>Total (ms)</th>
      <th>Per step (ms)</th>
      <th>max_N/tile</th>
      <th>Pairs</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>'''

chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Corrector Phase Profiler</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #f8f9fa; margin: 0; padding: 24px; color: #212529; }}
  h1   {{ font-size: 1.4em; margin-bottom: 4px; }}
  p.meta {{ color: #6c757d; font-size: 0.9em; margin-bottom: 20px; }}
  .card {{ background: white; border-radius: 10px; padding: 20px;
           box-shadow: 0 1px 4px rgba(0,0,0,.1); margin-bottom: 24px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.88em; }}
  th,td {{ padding: 8px 12px; text-align: right; border-bottom: 1px solid #dee2e6; }}
  th    {{ background: #343a40; color: white; font-weight: 600; }}
  td:first-child, th:first-child {{ text-align: left; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  .pct  {{ color: #6c757d; font-size: 0.85em; }}
  .note {{ font-size: 0.82em; color: #6c757d; margin-top: 10px; }}
</style>
</head>
<body>
<h1>Corrector Phase Profiler</h1>
<p class="meta">K={K} passes · N=2500 · positions_without t={DATA_T} · {N_REPS} reps · warmup {N_WARMUP}</p>

<div class="card">{chart_html}</div>

<div class="card">
  <h2 style="font-size:1.1em;margin-top:0">Numeric breakdown</h2>
  {table_html}
  <p class="note">
    Phase 1a: arithmetic tile membership (CPU numpy, 9-image loop)<br>
    Phase 1b: batch make_invariant — padded sort + masked mean + batched PCA (CPU numpy)<br>
    Phase 2: GPU forward pass (H2D transfer → model → D2H transfer + sync)<br>
    Phase 3: batch un-rotate + scatter accumulate + PBC wrap (CPU numpy)
  </p>
</div>
</body>
</html>'''

out = Path('docs/profile_report.html')
out.write_text(html, encoding='utf-8')
print(f'\nReport written to {out}')
