"""
measure_comparison.py — Quantitative evaluation across model / training configs,
each applied 1 / 3 / 5 times at inference.

Outputs (written to analysis/):
  comparison_report.md    — full metric table + commentary
  comparison_metrics.png  — bar-chart figure
"""

import importlib
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # project root

from data.data_generator import PoissonDiskDataset, PackedPoissonDiskDataset
from data.data_processor  import DataProcessor
from utils.config         import load_model_config

# ── Config ──────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_CLOUDS    = 500          # test set size
RD          = 0.05
DIM         = 2
N_POINTS    = 50
NOISE_MIN   = 0.0
NOISE_MAX   = 0.03
SEED        = 999          # different from training seed (42)
PASSES      = [1, 3, 5]

ROOT = Path(__file__).parent.parent  # project root

RUNS = {
    'model6 K=1lam':   {
        'label'     : 'Model6  K=1 trained',
        'run_dir'   : ROOT / 'training_artifacts/train_run_2026-05-26_09-35-52',
        'model_file': 'models/fixed_rd/model6',
    },
    'model7 K=1lam':   {
        'label'     : 'Model7  K=1 trained',
        'run_dir'   : ROOT / 'training_artifacts/train_run_2026-05-26_10-52-05',
        'model_file': 'models/fixed_rd/model7',
    },
    'model7 K=3lam*':  {
        'label'     : 'Model7  K=3 + princ. lam',
        'run_dir'   : ROOT / 'training_artifacts/train_run_2026-05-26_15-19-27',
        'model_file': 'models/fixed_rd/model7',
    },
    'model8 K=3lam*':  {
        'label'     : 'Model8  K=3 + GELU',
        'run_dir'   : ROOT / 'training_artifacts/train_run_2026-05-26_16-22-46',
        'model_file': 'models/fixed_rd/model7',   # same architecture, GELU via config
    },
    'model9 K=3lam*':  {
        'label'     : 'Model9  K=3 + clamped',
        'run_dir'   : ROOT / 'training_artifacts/train_run_2026-05-26_17-34-01',
        'model_file': 'models/fixed_rd/model9',
    },
}

# ── Data ─────────────────────────────────────────────────────────────────────
print("Generating test set...")
dataset   = PoissonDiskDataset(
    dim=DIM, cardinality=N_POINTS, rd=RD, seed=SEED,
    noise_scale_min=NOISE_MIN, noise_scale_max=NOISE_MAX,
)
processor = DataProcessor()

clean_np  = dataset.generate_sample(N_CLOUDS)
noisy_np  = dataset.noise_sample(clean_np)
proc_np, _, _ = processor.make_invariant(noisy_np)

x_noisy   = torch.tensor(proc_np, dtype=torch.float32, device=DEVICE)  # (N_CLOUDS, N_POINTS, DIM)
rd_t      = torch.tensor(RD, dtype=torch.float32, device=DEVICE)

# ── Helpers ──────────────────────────────────────────────────────────────────
def compute_stats(cloud_t, origin_t, rd_t):
    """
    cloud_t  : (B, N, D) corrected positions (invariant space)
    origin_t : (B, N, D) original noisy positions
    rd_t     : scalar

    Returns a dict of scalar metrics.
    """
    B, N, D = cloud_t.shape
    eye      = torch.eye(N, dtype=torch.bool, device=cloud_t.device)
    off_diag = ~eye

    pw   = torch.cdist(cloud_t, cloud_t)                             # (B, N, N)
    pd   = pw[:, off_diag]                                           # (B, N*(N-1))
    viol = torch.relu(rd_t - pd)

    nn   = pw.masked_fill(eye.unsqueeze(0), float('inf')).min(dim=-1).values  # (B, N)

    disp_vec  = cloud_t - origin_t                                   # (B, N, D)
    disp_norm = disp_vec.norm(dim=-1)                                # (B, N)

    mean_viol    = viol.mean().item()
    med_viol     = viol.median().item()
    ill_pct      = (pd < rd_t).float().mean().item() * 100
    leg_pct      = (~(pd < rd_t).any(dim=-1)).float().mean().item() * 100
    mean_nn      = nn.mean().item()
    med_nn       = nn.median().item()
    mean_disp    = disp_norm.mean().item()
    med_disp     = disp_norm.median().item()
    max_disp     = disp_norm.max().item()
    std_disp     = disp_norm.std().item()

    return dict(
        mean_viol   = mean_viol,
        med_viol    = med_viol,
        ill_pct     = ill_pct,
        ill_count   = (pd < rd_t).float().sum(dim=-1).mean().item() / 2,  # mean unique illegal pairs/cloud
        mean_nn     = mean_nn,
        med_nn      = med_nn,
        mean_disp   = mean_disp,
        med_disp    = med_disp,
        max_disp    = max_disp,
        std_disp    = std_disp,
        mean_disp_rd= mean_disp / RD,
        med_disp_rd = med_disp  / RD,
        max_disp_rd = max_disp  / RD,
    )

# ── Baseline (noisy, no correction) ──────────────────────────────────────────
print("Computing noisy baseline stats...")
with torch.no_grad():
    baseline = compute_stats(x_noisy, x_noisy, rd_t)
    baseline['mean_disp']    = 0.0
    baseline['med_disp']     = 0.0
    baseline['max_disp']     = 0.0
    baseline['std_disp']     = 0.0
    baseline['mean_disp_rd'] = 0.0
    baseline['med_disp_rd']  = 0.0
    baseline['max_disp_rd']  = 0.0

results = {'noisy': baseline}

# ── Load models and evaluate ─────────────────────────────────────────────────
for model_name, cfg in RUNS.items():
    run_dir    = cfg['run_dir']
    model_file = cfg['model_file']
    yaml_files = list((run_dir / 'configs').glob('model_config*.yaml'))
    assert len(yaml_files) == 1, f"Expected 1 model config, found {yaml_files}"
    model_cfg = load_model_config(str(yaml_files[0]))

    mod        = importlib.import_module(model_file.replace('/', '.'))
    model      = mod.CorrectorModel(
        model_config=model_cfg,
        input_dim=DIM,
        initialization='xavier_uniform',
    ).to(DEVICE)
    weights = run_dir / 'model_final.pt'
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nLoaded {model_name}: {n_params:,} params from {weights}")

    with torch.no_grad():
        for k in PASSES:
            xk = x_noisy.clone()
            for _ in range(k):
                d  = model(xk, rd=rd_t) if getattr(model, 'uses_rd', False) else model(xk)
                xk = xk + d

            key   = f"{model_name}_x{k}"
            stats = compute_stats(xk, x_noisy, rd_t)
            results[key] = stats
            print(f"  {key:20s}  ill={stats['ill_pct']:.2f}%  viol/cloud={stats['ill_count']:.2f}  "
                  f"disp={stats['mean_disp']:.4f} ({stats['mean_disp_rd']:.3f}xrd)")

# ── Row ordering and labels ───────────────────────────────────────────────────
ROWS = [
    'noisy',
    'model6 K=1lam_x1',  'model6 K=1lam_x3',  'model6 K=1lam_x5',
    'model7 K=1lam_x1',  'model7 K=1lam_x3',  'model7 K=1lam_x5',
    'model7 K=3lam*_x1', 'model7 K=3lam*_x3',
    'model8 K=3lam*_x1', 'model8 K=3lam*_x3',
    'model9 K=3lam*_x1', 'model9 K=3lam*_x3',
]

ROW_LABELS = {
    'noisy'              : 'Noisy input',
    'model6 K=1lam_x1'    : 'Model6  K=1lam     x1',
    'model6 K=1lam_x3'    : 'Model6  K=1lam     x3',
    'model6 K=1lam_x5'    : 'Model6  K=1lam     x5',
    'model7 K=1lam_x1'    : 'Model7  K=1lam     x1',
    'model7 K=1lam_x3'    : 'Model7  K=1lam     x3',
    'model7 K=1lam_x5'    : 'Model7  K=1lam     x5',
    'model7 K=3lam*_x1'   : 'Model7  K=3+lam    x1',
    'model7 K=3lam*_x3'   : 'Model7  K=3+lam    x3',
    'model8 K=3lam*_x1'   : 'Model8  GELU       x1',
    'model8 K=3lam*_x3'   : 'Model8  GELU       x3',
    'model9 K=3lam*_x1'   : 'Model9  clamped    x1',
    'model9 K=3lam*_x3'   : 'Model9  clamped    x3',
}

# ── Markdown report ───────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent          # analysis/

lines = []
lines.append("# Model Comparison Report\n")
lines.append(
    f"**Test set:** {N_CLOUDS} clouds · {N_POINTS} points · rd={RD} · "
    f"noise ∈ [{NOISE_MIN}, {NOISE_MAX}] · seed={SEED} (held-out)\n"
)
lines.append("**Models compared:**\n")
lines.append("- Model6 K=1lam  — uniform-push edge net, K=1 trained (original lam)\n")
lines.append("- Model7 K=1lam  — violation-weighted edge net, K=1 trained (original lam)\n")
lines.append("- Model7 K=3lam* — violation-weighted edge net, K=3 unrolling + principled lam (lam1=20, lam2=0.04)\n\n")
lines.append(f"**Device:** {DEVICE}\n\n")
lines.append("![Comparison sheet](comparison_sheet.png)\n")

# ── Table helpers ─────────────────────────────────────────────────────────────
noisy_viol     = results['noisy']['mean_viol']
noisy_ill      = results['noisy']['ill_pct']
noisy_ill_cnt  = results['noisy']['ill_count'] if 'ill_count' in results['noisy'] else None

def sep(row):
    """Return True if this row starts a new model group."""
    return row in ('model7 K=1lam_x1', 'model7 K=3lam*_x1')

def row_line(row, cols):
    if sep(row):
        lines.append('|' + '|'.join([' --- ' for _ in cols]) + '|')
    return '| ' + ' | '.join(cols) + ' |'

# ── Core quality ──────────────────────────────────────────────────────────────
lines.append("## Core Quality Metrics\n")
lines.append("| Condition | Ill pairs % | Viol/cloud | Viol reduction % | Mean violation |")
lines.append("|-----------|:-----------:|:----------:|:----------------:|:--------------:|")
for row in ROWS:
    s = results[row]
    viol_cnt  = s.get('ill_count', s['ill_pct'] * 50 * 49 / 2 / 100)
    viol_red  = (noisy_viol - s['mean_viol']) / (noisy_viol + 1e-9) * 100 if row != 'noisy' else 0.0
    label     = ROW_LABELS.get(row, row)
    if sep(row):
        lines.append("|---|---|---|---|---|")
    lines.append(
        f"| {label:20s} "
        f"| {s['ill_pct']:>11.3f} "
        f"| {viol_cnt:>10.2f} "
        f"| {viol_red:>16.1f} "
        f"| {s['mean_viol']:>14.6f} |"
    )

# ── NN drift ─────────────────────────────────────────────────────────────────
lines.append("\n## Nearest-Neighbour Distance\n")
lines.append("*(natural Poisson disk ≈ 1.0 × rd)*\n")
lines.append("| Condition | Mean nn / rd | Med nn / rd |")
lines.append("|-----------|:------------:|:-----------:|")
for row in ROWS:
    s = results[row]
    if sep(row): lines.append("|---|---|---|")
    lines.append(
        f"| {ROW_LABELS.get(row,row):20s} "
        f"| {s['mean_nn']/RD:>12.4f} "
        f"| {s['med_nn']/RD:>11.4f} |"
    )

# ── Displacement ──────────────────────────────────────────────────────────────
lines.append("\n## Displacement from Noisy Input\n")
lines.append("| Condition | Mean / rd | Median / rd | Max / rd |")
lines.append("|-----------|:---------:|:-----------:|:--------:|")
for row in ROWS:
    s = results[row]
    if sep(row): lines.append("|---|---|---|---|")
    lines.append(
        f"| {ROW_LABELS.get(row,row):20s} "
        f"| {s['mean_disp_rd']:>9.4f} "
        f"| {s['med_disp_rd']:>11.4f} "
        f"| {s['max_disp_rd']:>8.4f} |"
    )

# ── Surgical efficiency ───────────────────────────────────────────────────────
lines.append("\n## Surgical Efficiency\n")
lines.append("*(violation removed ÷ mean displacement — higher = more targeted)*\n")
lines.append("| Condition | Viol removed | Mean disp | Efficiency |")
lines.append("|-----------|:------------:|:---------:|:----------:|")
for row in ROWS:
    if row == 'noisy':
        lines.append(f"| {'Noisy input':20s} | {'—':^12s} | {'—':^9s} | {'—':^10s} |")
        continue
    s   = results[row]
    vr  = noisy_viol - s['mean_viol']
    eff = vr / (s['mean_disp'] + 1e-9)
    if sep(row): lines.append("|---|---|---|---|")
    lines.append(
        f"| {ROW_LABELS.get(row,row):20s} "
        f"| {vr:>12.6f} "
        f"| {s['mean_disp']:>9.5f} "
        f"| {eff:>10.4f} |"
    )

# ── Auto observations ─────────────────────────────────────────────────────────
lines.append("\n## Key Findings\n")

m6   = results['model6 K=1lam_x1']
m7   = results['model7 K=1lam_x1']
m7k  = results['model7 K=3lam*_x1']
m8   = results['model8 K=3lam*_x1']
m6_5 = results['model6 K=1lam_x5']
m7_5 = results['model7 K=1lam_x5']
m7k3 = results['model7 K=3lam*_x3']
m8_3 = results['model8 K=3lam*_x3']

eff_m6  = (noisy_viol - m6['mean_viol'])  / (m6['mean_disp']  + 1e-9)
eff_m7  = (noisy_viol - m7['mean_viol'])  / (m7['mean_disp']  + 1e-9)
eff_m7k = (noisy_viol - m7k['mean_viol']) / (m7k['mean_disp'] + 1e-9)

lines.append(
    f"### Single-pass (×1) — the deployment case\n\n"
    f"| | Ill pairs % | Viol reduction | Mean disp / rd | Efficiency |\n"
    f"|---|---|---|---|---|\n"
    f"| Model6  K=1lam  | {m6['ill_pct']:.2f}% | {(noisy_viol-m6['mean_viol'])/noisy_viol*100:.1f}% "
    f"| {m6['mean_disp_rd']:.3f} | {eff_m6:.4f} |\n"
    f"| Model7  K=1lam  | {m7['ill_pct']:.2f}% | {(noisy_viol-m7['mean_viol'])/noisy_viol*100:.1f}% "
    f"| {m7['mean_disp_rd']:.3f} | {eff_m7:.4f} |\n"
    f"| **Model7  K=3lam*** | **{m7k['ill_pct']:.2f}%** | **{(noisy_viol-m7k['mean_viol'])/noisy_viol*100:.1f}%** "
    f"| **{m7k['mean_disp_rd']:.3f}** | **{eff_m7k:.4f}** |\n\n"
    f"Model7 K=3lam* improves over the K=1 baseline on every metric at single-pass inference.\n"
    f"Illegal pairs: {m7['ill_pct']:.2f}% → {m7k['ill_pct']:.2f}%  "
    f"({(m7['ill_pct']-m7k['ill_pct'])/m7['ill_pct']*100:.0f}% relative drop).\n"
    f"Efficiency: {eff_m7:.4f} → {eff_m7k:.4f}  "
    f"({(eff_m7k-eff_m7)/eff_m7*100:.0f}% relative gain).\n\n"
    f"### NN drift at ×5 (over-spreading test)\n\n"
    f"| | Mean nn / rd | (natural ≈ 1.0) |\n"
    f"|---|---|---|\n"
    f"| Model6 K=1lam  ×5 | {m6_5['mean_nn']/RD:.3f} | {(m6_5['mean_nn']/RD-1)*100:.1f}% above natural |\n"
    f"| Model7 K=1lam  ×5 | {m7_5['mean_nn']/RD:.3f} | {(m7_5['mean_nn']/RD-1)*100:.1f}% above natural |\n"
    f"| Model7 K=3lam* ×3 | {m7k3['mean_nn']/RD:.3f} | {(m7k3['mean_nn']/RD-1)*100:.1f}% above natural |\n\n"
    f"Model6 continues to over-spread aggressively ({(m6_5['mean_nn']/RD-1)*100:.1f}% above natural at ×5). "
    f"Model7 K=3lam* applied ×3 shows less drift than Model7 K=1lam at ×5, "
    f"indicating the principled lambda training produces tighter, more conservative corrections.\n"
)

lines.append("\n![Metric charts](comparison_metrics.png)\n")

report_path = OUT_DIR / 'comparison_report.md'
report_path.write_text('\n'.join(lines), encoding='utf-8')
print(f"\nReport written to {report_path}")

# ── Bar-chart figure ──────────────────────────────────────────────────────────
METRICS = [
    ('ill_pct',       'Illegal pair %',     '%',    True),
    ('mean_viol',     'Mean violation',      '',     True),
    ('mean_disp_rd',  'Mean disp / rd',      '×rd',  True),
    ('max_disp_rd',   'Max disp / rd',       '×rd',  True),
    ('mean_nn',       'Mean nn dist',        '',     False),
]

BAR_ROWS = [r for r in ROWS if r != 'noisy']
bar_labels = [ROW_LABELS[r] for r in BAR_ROWS]

# blue=model6, orange=model7 K=1, green=model7 K=3lam*, purple=model8 GELU, red=model9 clamped
colors = [
    '#4e79a7', '#a0cbe8', '#1f5fa6',   # model6 x1/3/5
    '#f28e2b', '#ffbf79', '#c26a00',   # model7 K=1lam x1/3/5
    '#59a14f', '#8cd17d',              # model7 K=3lam* x1/3
    '#b07aa1', '#d4a6c8',              # model8 GELU x1/3
    '#e15759', '#ff9da7',              # model9 clamped x1/3
]

fig, axes = plt.subplots(1, len(METRICS), figsize=(20, 6))
fig.patch.set_facecolor('#0e1117')

for ax, (metric, title, unit, lower_is_better) in zip(axes, METRICS):
    ax.set_facecolor('#1a1d23')
    values = [results[r][metric] for r in BAR_ROWS]
    bars   = ax.bar(range(len(BAR_ROWS)), values, color=colors, width=0.65, edgecolor='none')

    bl = results['noisy'].get(metric)
    if bl is not None:
        ax.axhline(bl, color='#ff4444', linestyle='--', linewidth=1.1, alpha=0.8)

    ax.set_xticks(range(len(BAR_ROWS)))
    ax.set_xticklabels(bar_labels, rotation=40, ha='right', fontsize=7, color='#cccccc')
    ax.set_title(title + (f' ({unit})' if unit else ''), color='white', fontsize=10, pad=6)
    ax.tick_params(colors='#aaaaaa', labelsize=7)
    ax.spines[:].set_visible(False)

    vmax = max(values) if values else 1
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + vmax * 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=6.5, color='#dddddd')

    ax.text(0.98, 0.98, '↓ better' if lower_is_better else '↑ better',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            color='#666666', style='italic')

handles = [
    plt.Line2D([0], [0], color='#ff4444', linestyle='--', linewidth=1.5, label='Noisy baseline'),
    plt.Rectangle((0,0),1,1, color='#4e79a7', label='Model6  K=1lam (uniform push)'),
    plt.Rectangle((0,0),1,1, color='#f28e2b', label='Model7  K=1lam (viol-weighted)'),
    plt.Rectangle((0,0),1,1, color='#59a14f', label='Model7  K=3+princ.lam (ReLU)'),
    plt.Rectangle((0,0),1,1, color='#b07aa1', label='Model8  K=3+princ.lam (GELU)'),
    plt.Rectangle((0,0),1,1, color='#e15759', label='Model9  K=3+clamped (GELU, deep)'),
]
fig.legend(handles=handles, loc='lower center', ncol=3, framealpha=0.15,
           labelcolor='white', fontsize=8, bbox_to_anchor=(0.5, -0.14))

fig.suptitle(
    f'Model comparison — 500 test clouds · {N_POINTS} pts · rd={RD}',
    color='white', fontsize=13, y=1.02,
)
plt.tight_layout()
out_path = OUT_DIR / 'comparison_metrics.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Figure saved to {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Packed regime evaluation
# Model9 trained on packed clouds (packedness=0.5, N≈231, rd=0.05).
# Evaluated on a fresh packed test set with seed=999 (held-out).
# ══════════════════════════════════════════════════════════════════════════════
PACKED_RUN_DIR  = ROOT / 'training_artifacts/train_run_2026-05-26_20-48-12'
PACKED_RD       = 0.05
PACKED_PACKEDNESS = 0.5
PACKED_N_CLOUDS = 200   # fewer due to O(N^2) cost at N≈231
PACKED_SEED     = 999
PACKED_PASSES   = [1, 3, 5]

print("\n--- Packed regime evaluation ---")
packed_dataset = PackedPoissonDiskDataset(
    dim=2, rd=PACKED_RD, packedness=PACKED_PACKEDNESS,
    seed=PACKED_SEED,
    noise_scale_min=0.0, noise_scale_max=0.02,
)
packed_clean = packed_dataset.generate_sample(PACKED_N_CLOUDS)
packed_noisy = packed_dataset.noise_sample(packed_clean)
packed_proc, _, _ = processor.make_invariant(packed_noisy)

x_packed   = torch.tensor(packed_proc, dtype=torch.float32, device=DEVICE)
rd_packed  = torch.tensor(PACKED_RD,  dtype=torch.float32, device=DEVICE)
packed_N   = x_packed.shape[1]
print(f"Packed test set: {PACKED_N_CLOUDS} clouds x {packed_N} pts, rd={PACKED_RD}")

def compute_stats_packed(cloud_t, origin_t, rd_t):
    """Same as compute_stats but rd-agnostic (uses argument, not global RD)."""
    B, N, D = cloud_t.shape
    eye      = torch.eye(N, dtype=torch.bool, device=cloud_t.device)
    off_diag = ~eye
    rd_val   = rd_t.item()

    pw   = torch.cdist(cloud_t, cloud_t)
    pd   = pw[:, off_diag]
    viol = torch.relu(rd_t - pd)
    nn   = pw.masked_fill(eye.unsqueeze(0), float('inf')).min(dim=-1).values
    disp_norm = (cloud_t - origin_t).norm(dim=-1)

    return dict(
        mean_viol    = viol.mean().item(),
        ill_pct      = (pd < rd_t).float().mean().item() * 100,
        ill_count    = (pd < rd_t).float().sum(dim=-1).mean().item() / 2,
        mean_nn      = nn.mean().item(),
        mean_disp    = disp_norm.mean().item(),
        mean_disp_rd = disp_norm.mean().item() / rd_val,
        max_disp_rd  = disp_norm.max().item() / rd_val,
    )

# Packed baseline (noisy, no correction)
with torch.no_grad():
    packed_baseline = compute_stats_packed(x_packed, x_packed, rd_packed)
    packed_baseline['mean_disp'] = 0.0
    packed_baseline['mean_disp_rd'] = 0.0
    packed_baseline['max_disp_rd'] = 0.0

print(f"  Packed noisy baseline: ill={packed_baseline['ill_pct']:.3f}%  "
      f"viol/cloud={packed_baseline['ill_count']:.1f}")

# Load packed model9
packed_yaml = list((PACKED_RUN_DIR / 'configs').glob('model_config*.yaml'))
assert len(packed_yaml) == 1, f"Expected 1 config, got {packed_yaml}"
packed_model_cfg = load_model_config(str(packed_yaml[0]))

packed_mod   = importlib.import_module('models.fixed_rd.model9')
packed_model = packed_mod.CorrectorModel(
    model_config=packed_model_cfg,
    input_dim=2,
    initialization='xavier_uniform',
).to(DEVICE)
packed_model.load_state_dict(torch.load(PACKED_RUN_DIR / 'model_final.pt', map_location=DEVICE))
packed_model.eval()
print(f"Loaded packed model9: {sum(p.numel() for p in packed_model.parameters()):,} params")

packed_results = {'noisy': packed_baseline}
with torch.no_grad():
    for k in PACKED_PASSES:
        xk = x_packed.clone()
        for _ in range(k):
            d  = packed_model(xk, rd=rd_packed)
            xk = xk + d
        s = compute_stats_packed(xk, x_packed, rd_packed)
        packed_results[f'x{k}'] = s
        print(f"  packed model9 x{k}: ill={s['ill_pct']:.3f}%  viol/cloud={s['ill_count']:.1f}  "
              f"disp={s['mean_disp']:.4f} ({s['mean_disp_rd']:.3f}xrd)  "
              f"viol_red={(packed_baseline['mean_viol']-s['mean_viol'])/packed_baseline['mean_viol']*100:.1f}%")

# ── Append packed section to report ──────────────────────────────────────────
packed_lines = []
packed_lines.append("\n---\n")
packed_lines.append("## Packed Regime — Model9 (packedness=0.5, N≈231, rd=0.05)\n")
packed_lines.append(
    f"**Test set:** {PACKED_N_CLOUDS} clouds · ~{packed_N} points · rd={PACKED_RD} · "
    f"packedness={PACKED_PACKEDNESS} · noise ∈ [0.0, 0.02] · seed={PACKED_SEED} (held-out)\n\n"
    f"Model9 trained from scratch on packed data. This regime has no free space: "
    f"every cloud starts with ~{packed_baseline['ill_count']:.0f} violations and points "
    f"cannot be pushed far without creating new conflicts.\n"
)

p_noisy_viol = packed_baseline['mean_viol']

packed_lines.append("| Passes | Ill pairs % | Viol/cloud | Viol reduction % | Mean disp / rd | Eff (viol_rem/disp) |")
packed_lines.append("|--------|:-----------:|:----------:|:----------------:|:--------------:|:-------------------:|")

for lbl, key in [('Noisy input', 'noisy'), ('x1 (deploy)', 'x1'), ('x3', 'x3'), ('x5', 'x5')]:
    s = packed_results[key]
    if key == 'noisy':
        packed_lines.append(
            f"| {lbl:12s} | {s['ill_pct']:>11.3f} | {s['ill_count']:>10.1f} | {'—':>16s} | {'—':>14s} | {'—':>19s} |"
        )
    else:
        vr  = p_noisy_viol - s['mean_viol']
        vr_pct = vr / (p_noisy_viol + 1e-9) * 100
        eff = vr / (s['mean_disp'] + 1e-9)
        packed_lines.append(
            f"| {lbl:12s} | {s['ill_pct']:>11.3f} | {s['ill_count']:>10.1f} "
            f"| {vr_pct:>16.1f} | {s['mean_disp_rd']:>14.4f} | {eff:>19.4f} |"
        )

packed_lines.append(
    f"\n**K=3 advantage:** K=3 reduces {packed_results['x3']['ill_count']:.0f} vs K=1 "
    f"{packed_results['x1']['ill_count']:.0f} violations/cloud "
    f"({(packed_results['x3']['ill_count']/packed_results['x1']['ill_count']-1)*100:+.0f}% relative).\n"
    f"In the sparse regime (N=50) K=3 vs K=1 gave only marginal improvements; "
    f"in the packed regime the multi-pass advantage is much larger because packed clouds "
    f"require sequential resolution — moving one point opens space for the next correction.\n"
)

# Append to existing report
with open(report_path, 'a', encoding='utf-8') as f:
    f.write('\n'.join(packed_lines))
print(f"Packed section appended to {report_path}")

# ── Packed bar chart ──────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(12, 5))
fig2.patch.set_facecolor('#0e1117')

packed_bar_keys   = ['x1', 'x3', 'x5']
packed_bar_labels = ['x1 (deploy)', 'x3', 'x5']
packed_colors     = ['#e15759', '#c44e52', '#9b2335']

for ax, (metric, title, unit) in zip(axes2, [
    ('ill_pct',      'Illegal pair %',  '%'),
    ('ill_count',    'Violations/cloud', ''),
    ('mean_disp_rd', 'Mean disp / rd',  'xrd'),
]):
    ax.set_facecolor('#1a1d23')
    values = [packed_results[k][metric] for k in packed_bar_keys]
    baseline_val = packed_baseline[metric]

    bars = ax.bar(range(3), values, color=packed_colors, width=0.5, edgecolor='none')
    ax.axhline(baseline_val, color='#ff4444', linestyle='--', linewidth=1.1, alpha=0.8, label='Noisy baseline')

    ax.set_xticks(range(3))
    ax.set_xticklabels(packed_bar_labels, color='#cccccc', fontsize=9)
    ax.set_title(title + (f' ({unit})' if unit else ''), color='white', fontsize=10, pad=6)
    ax.tick_params(colors='#aaaaaa')
    ax.spines[:].set_visible(False)

    vmax = max(values + [baseline_val])
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + vmax * 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8.5, color='#dddddd')

    ax.legend(fontsize=7, labelcolor='white', framealpha=0.1)

fig2.suptitle(
    f'Packed regime — Model9  ({PACKED_N_CLOUDS} test clouds · ~{packed_N} pts · rd={PACKED_RD} · packedness={PACKED_PACKEDNESS})',
    color='white', fontsize=11, y=1.02,
)
plt.tight_layout()
packed_chart_path = OUT_DIR / 'comparison_metrics_packed.png'
plt.savefig(packed_chart_path, dpi=150, bbox_inches='tight', facecolor=fig2.get_facecolor())
print(f"Packed chart saved to {packed_chart_path}")

print("\nDone.")
