"""
compare_capacity_grid.py — K=1/2/3 comparison across the p=0.90 capacity grid.

Models compared (all trained at packedness=0.90, N~416, rd=0.05):
  - model9  hd128 d3  (51K params)  — baseline
  - model9  hd256 d3  (200K params) — 2x width
  - model9  hd256 d4  (265K params) — depth
  - model10 hd256 d4  (267K params) — depth + self-feature

Outputs (written to analysis/):
  capacity_grid_report.md
  capacity_grid_k123.png
"""

import importlib, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_generator import PackedPoissonDiskDataset
from data.data_processor  import DataProcessor
from utils.config         import load_model_config

# ── Config ───────────────────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RD          = 0.05
PACKEDNESS  = 0.9
N_CLOUDS    = 200
SEED        = 1337          # held-out, different from training seed 42
PASSES      = [1, 2, 3]
ROOT        = Path(__file__).parent.parent

RUNS = {
    'model9\nhd128 d3\n(51K)':  {
        'run_dir'   : ROOT / 'training_artifacts/train_run_2026-05-27_14-17-02',
        'model_file': 'models.fixed_rd.model9',
        'color'     : '#4e79a7',
    },
    'model9\nhd256 d3\n(200K)': {
        'run_dir'   : ROOT / 'training_artifacts/train_run_2026-05-27_15-53-28',
        'model_file': 'models.fixed_rd.model9',
        'color'     : '#f28e2b',
    },
    'model9\nhd256 d4\n(265K)': {
        'run_dir'   : ROOT / 'training_artifacts/train_run_2026-05-28_10-54-48',
        'model_file': 'models.fixed_rd.model9',
        'color'     : '#59a14f',
    },
    'model10\nhd256 d4\n(267K)\n+self-feat': {
        'run_dir'   : ROOT / 'training_artifacts/train_run_2026-05-28_12-51-14',
        'model_file': 'models.fixed_rd.model10',
        'color'     : '#e15759',
    },
}

# ── Shared test set ───────────────────────────────────────────────────────────
print(f"Generating test set: packedness={PACKEDNESS}, rd={RD}, N_clouds={N_CLOUDS}, seed={SEED}")
# Generate one cloud at a time to avoid the batch-min truncation cutting N~416 → ~239
# when an unlucky seed produces a short cloud in a large batch.
dataset   = PackedPoissonDiskDataset(dim=2, rd=RD, packedness=PACKEDNESS,
                                     seed=SEED, noise_scale_min=0.0, noise_scale_max=0.02)
processor = DataProcessor()

singles = []
for i in range(N_CLOUDS):
    ds_i = PackedPoissonDiskDataset(dim=2, rd=RD, packedness=PACKEDNESS,
                                    seed=SEED + i, noise_scale_min=0.0, noise_scale_max=0.02)
    singles.append(ds_i.generate_sample(1)[0])   # (N_i, 2)

# Truncate to shared minimum N (should all be ~N_target now, tiny variance)
n_min    = min(c.shape[0] for c in singles)
clean_np = np.stack([c[:n_min] for c in singles])  # (N_CLOUDS, n_min, 2)
noisy_np           = dataset.noise_sample(clean_np)
proc_np, _, revert = processor.make_invariant(noisy_np)

x_noisy = torch.tensor(proc_np, dtype=torch.float32, device=DEVICE)
rd_t    = torch.tensor(RD, dtype=torch.float32, device=DEVICE)
N_PTS   = x_noisy.shape[1]
print(f"  N_pts per cloud: {N_PTS}")

# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(cloud_t, origin_t):
    B, N, _ = cloud_t.shape
    eye      = torch.eye(N, dtype=torch.bool, device=cloud_t.device)
    pw       = torch.cdist(cloud_t, cloud_t)
    pd       = pw[:, ~eye]
    viol     = torch.relu(rd_t - pd)

    illegal_pairs   = (pd < rd_t).float().sum(dim=-1).mean().item() / 2
    illegal_pct     = (pd < rd_t).float().mean().item() * 100
    mean_viol       = viol.mean().item()
    mean_nn         = pw.masked_fill(eye.unsqueeze(0), float('inf')).min(-1).values.mean().item()
    mean_disp       = (cloud_t - origin_t).norm(dim=-1).mean().item()

    return dict(
        illegal_pairs = illegal_pairs,
        illegal_pct   = illegal_pct,
        mean_viol     = mean_viol,
        mean_nn       = mean_nn,
        mean_disp     = mean_disp,
    )

with torch.no_grad():
    baseline = compute_metrics(x_noisy, x_noisy)
    baseline['mean_disp'] = 0.0

print(f"  Noisy baseline: {baseline['illegal_pairs']:.1f} viol/cloud, "
      f"{baseline['illegal_pct']:.2f}% illegal pairs")

# ── Evaluate all models ───────────────────────────────────────────────────────
results = {}   # name -> {k: metrics_dict}

for name, cfg in RUNS.items():
    run_dir    = cfg['run_dir']
    model_file = cfg['model_file']

    yaml_files = list((run_dir / 'configs').glob('model_config*.yaml'))
    assert len(yaml_files) == 1, f"Expected 1 model config in {run_dir}, found {yaml_files}"
    model_cfg = load_model_config(str(yaml_files[0]))

    mod   = importlib.import_module(model_file)
    model = mod.CorrectorModel(model_config=model_cfg, input_dim=2,
                               initialization='xavier_uniform').to(DEVICE)
    model.load_state_dict(torch.load(run_dir / 'model_final.pt', map_location=DEVICE))
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    tag = name.replace('\n', ' ')
    print(f"\n{tag}  ({n_params:,} params)")

    run_results = {}
    with torch.no_grad():
        for k in PASSES:
            xk = x_noisy.clone()
            for _ in range(k):
                xk = xk + model(xk, rd=rd_t)
            m = compute_metrics(xk, x_noisy)
            viol_red = (baseline['mean_viol'] - m['mean_viol']) / (baseline['mean_viol'] + 1e-9) * 100
            m['viol_reduction'] = viol_red
            eff_ceil = baseline['mean_viol'] / (baseline['mean_nn'] - rd_t.item() + 1e-9)
            m['efficiency_pct'] = (m['mean_viol'] / (m['mean_disp'] + 1e-9)) / (eff_ceil + 1e-9) * 100
            m['efficiency_pct'] = (baseline['mean_viol'] - m['mean_viol']) / (m['mean_disp'] + 1e-9) / (baseline['mean_viol'] / (baseline['mean_nn'] - rd_t.item() + 1e-9) + 1e-9) * 100
            run_results[k] = m
            print(f"  K={k}: viol_red={viol_red:.1f}%  "
                  f"viol/cloud={m['illegal_pairs']:.1f}  "
                  f"ill%={m['illegal_pct']:.2f}%  "
                  f"disp={m['mean_disp']/RD:.3f}xrd")
    results[name] = run_results

# ── Markdown report ───────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / 'outputs'
OUT_DIR.mkdir(exist_ok=True)

SHORT = {
    'model9\nhd128 d3\n(51K)'                  : 'model9 hd128 d3',
    'model9\nhd256 d3\n(200K)'                 : 'model9 hd256 d3',
    'model9\nhd256 d4\n(265K)'                 : 'model9 hd256 d4',
    'model10\nhd256 d4\n(267K)\n+self-feat'    : 'model10 hd256 d4 +self',
}

lines = []
lines.append("# Capacity Grid — K=1/2/3 Comparison\n")
lines.append(
    f"**Test set:** {N_CLOUDS} clouds · ~{N_PTS} pts · packedness={PACKEDNESS} · "
    f"rd={RD} · noise ∈ [0, 0.02] · seed={SEED} (held-out)\n"
)
lines.append(f"**Device:** {DEVICE}\n")
lines.append(
    f"**Noisy baseline:** {baseline['illegal_pairs']:.1f} viol/cloud · "
    f"{baseline['illegal_pct']:.2f}% illegal pairs\n"
)

lines.append("\n## Violation Reduction %\n")
lines.append("| Model | K=1 | K=2 | K=3 |")
lines.append("|---|---:|---:|---:|")
for name in RUNS:
    r  = results[name]
    lines.append(
        f"| {SHORT[name]} "
        f"| {r[1]['viol_reduction']:.1f}% "
        f"| {r[2]['viol_reduction']:.1f}% "
        f"| {r[3]['viol_reduction']:.1f}% |"
    )

lines.append("\n## Violations / Cloud\n")
lines.append("| Model | pre | K=1 | K=2 | K=3 |")
lines.append("|---|---:|---:|---:|---:|")
for name in RUNS:
    r = results[name]
    lines.append(
        f"| {SHORT[name]} "
        f"| {baseline['illegal_pairs']:.1f} "
        f"| {r[1]['illegal_pairs']:.1f} "
        f"| {r[2]['illegal_pairs']:.1f} "
        f"| {r[3]['illegal_pairs']:.1f} |"
    )

lines.append("\n## Illegal Pair %\n")
lines.append("| Model | pre | K=1 | K=2 | K=3 |")
lines.append("|---|---:|---:|---:|---:|")
for name in RUNS:
    r = results[name]
    lines.append(
        f"| {SHORT[name]} "
        f"| {baseline['illegal_pct']:.2f}% "
        f"| {r[1]['illegal_pct']:.2f}% "
        f"| {r[2]['illegal_pct']:.2f}% "
        f"| {r[3]['illegal_pct']:.2f}% |"
    )

lines.append("\n## Mean Displacement / rd\n")
lines.append("| Model | K=1 | K=2 | K=3 |")
lines.append("|---|---:|---:|---:|")
for name in RUNS:
    r = results[name]
    lines.append(
        f"| {SHORT[name]} "
        f"| {r[1]['mean_disp']/RD:.3f} "
        f"| {r[2]['mean_disp']/RD:.3f} "
        f"| {r[3]['mean_disp']/RD:.3f} |"
    )

lines.append("\n![K=1/2/3 comparison](capacity_grid_k123.png)\n")

report_path = OUT_DIR / 'capacity_grid_report.md'
report_path.write_text('\n'.join(lines), encoding='utf-8')
print(f"\nReport written to {report_path}")

# ── Figure — line plot per metric ─────────────────────────────────────────────
METRICS_PLOT = [
    ('viol_reduction', 'Violation reduction %',   True),
    ('illegal_pairs',  'Violations / cloud',       False),
    ('illegal_pct',    'Illegal pair %',           False),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('#0e1117')

for ax, (metric, title, higher_better) in zip(axes, METRICS_PLOT):
    ax.set_facecolor('#1a1d23')
    ax.spines[:].set_visible(False)
    ax.tick_params(colors='#aaaaaa')
    ax.set_title(title, color='white', fontsize=11, pad=8)
    ax.set_xlabel('Inference passes (K)', color='#aaaaaa', fontsize=9)
    ax.set_xticks(PASSES)
    ax.xaxis.label.set_color('#aaaaaa')

    if metric == 'viol_reduction':
        # draw noisy baseline at 0%
        ax.axhline(0, color='#ff4444', linestyle='--', linewidth=1, alpha=0.6, label='Noisy (0%)')
    else:
        ax.axhline(baseline[metric], color='#ff4444', linestyle='--',
                   linewidth=1, alpha=0.6, label='Noisy baseline')

    for name, cfg in RUNS.items():
        r      = results[name]
        vals   = [r[k][metric] for k in PASSES]
        label  = SHORT[name]
        color  = cfg['color']
        ax.plot(PASSES, vals, color=color, linewidth=2.2, marker='o',
                markersize=7, label=label)
        for k, v in zip(PASSES, vals):
            va = 'bottom' if higher_better else 'top'
            offset = max(vals) * 0.015 if higher_better else -max(baseline[metric], max(vals)) * 0.015
            ax.text(k, v + offset, f'{v:.1f}', ha='center', va=va,
                    fontsize=7.5, color=color)

    ax.legend(fontsize=7.5, labelcolor='white', framealpha=0.15,
              loc='lower right' if higher_better else 'upper right')
    ax.yaxis.tick_right()
    ax.yaxis.set_tick_params(labelcolor='#aaaaaa')

fig.suptitle(
    f'Capacity grid — p=0.90  ({N_CLOUDS} test clouds · ~{N_PTS} pts · rd={RD})',
    color='white', fontsize=13, y=1.02,
)
plt.tight_layout()
chart_path = OUT_DIR / 'capacity_grid_k123.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Chart saved to {chart_path}")
print("\nDone.")
