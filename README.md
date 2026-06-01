# Poisson Disk Corrector

A neural network that enforces the minimum-distance constraint on 2D particle clouds, deployed as a post-processing corrector for SPH simulations.

---

## Start here

| Resource | What it is |
|---|---|
| **[`docs/guide.html`](docs/guide.html)** | Complete project guide — open in a browser |
| **[`notebooks/pipeline_smoke_test.ipynb`](notebooks/pipeline_smoke_test.ipynb)** | Interactive walkthrough — runs every component end-to-end |

---

## What this does

SPH simulations require particles to maintain a minimum nearest-neighbour distance $r_d$. Over time particles drift closer together, introducing violations. The standard fix — the Transport Velocity (TV) algorithm — applies a global velocity correction that interferes with the simulation physics.

This project trains a neural network (model9) to apply purely geometric corrections: given a violating cloud, predict the smallest per-particle displacement that restores $d_\text{PBC}(p_i, p_j) \geq r_d$ for all pairs.

**Key result:** K=5 correction passes on non-TV SPH data consistently outperform TV from t≈300 onwards, with a +11% improvement in mean nearest-neighbour distance at t=700.

---

## The notebook

`notebooks/pipeline_smoke_test.ipynb` is the fastest way to understand the project. It runs every pipeline component in order and produces inline figures for each step.

```bash
.venv\Scripts\pip install jupyter
.venv\Scripts\jupyter notebook notebooks/pipeline_smoke_test.ipynb
```

What it covers, in order:

| Cell | What you see |
|---|---|
| Data generation | Clean vs noisy Poisson disk cloud, coloured by violation status |
| Preprocessing | `make_invariant` round-trip error ≈ 1e-7 |
| Training artifacts | Loss curve (K=1 vs K=3) + last training sample image |
| Model forward pass | Displacement shape, max bounded by `max_displacement` |
| SPH data | Violation stats at t=0, 300, 700 |
| **Tiling decomposition** | Full 6×6 grid coloured by tile ID + zoom showing core / adjacent ghost / PBC ghost |
| **Ghost buffer proof** | Closest violating pair — both endpoints captured in one tile's ghost buffer |
| Coordinate scaling | Manual tile demo: scale → invariant → model → unscale |
| Full corrector | K=1, 3, 5 on t=300; grid swap (10×10) with same model |
| Timeseries | K=5 vs TV over 5 timesteps |
| Comparison figure | Standard 4-row × 5-col figure |

---

## The guide

[`docs/guide.html`](docs/guide.html) — open in a browser. Covers:

1. The Poisson disk constraint and SPH context
2. Model architecture (model9 — violation-weighted all-pairs message passing)
3. Training: configs, commands, artifact structure
4. Production checkpoints and when to use each
5. Inference pipeline: tiling → ghost buffer → coordinate scaling → model
6. Running experiments with the config-based CLI
7. Reading the comparison figure
8. Key results table (K=5 vs TV across t=0–1000)
9. End-to-end walkthrough with code
10. Non-obvious design decisions

---

## Config-based modular design

Every component is configured by a YAML file. Components are **independently swappable** — changing one does not require touching the others.

```
configs/
├── dataset_configs/     # N, rd, packedness, noise range
├── model_configs/       # architecture, hidden_dim, max_displacement
├── loss_configs/        # lambda1, lambda2 (hybrid_loss)
├── trainer_configs/     # K unroll steps, batch size, iterations
├── grid_configs/        # grid_size, ghost_factor — TILING ONLY
└── smoke_test/          # fast CPU configs for quick checks

inference/configs/       # experiment configs: link model + data + grid
├── grid_6x6.yaml        # n100 model, 6×6 tiling, scale=3.8
└── grid_10x10.yaml      # n50 model, 10×10 tiling, scale=2.5
```

**Swap the tiling without touching the model:**
```yaml
# inference/configs/my_experiment.yaml
model:
  checkpoint: training_artifacts/train_run_.../model_final.pt
  rd_train:   0.076
data:
  rd_test:    0.02
tiling: configs/grid_configs/grid_6x6.yaml   # ← change this one line
experiment:
  k_values: [1, 3, 5]
```

**Run:**
```bash
.venv\Scripts\python.exe inference/run_experiment.py inference/configs/grid_6x6.yaml
```

**Output** (mirrors training artifact structure):
```
inference/experiments/exp_YYYY-MM-DD_HH-MM-SS/
├── config.yaml       # copy of the config used
├── run.log           # per-timestep metrics
├── timeseries.png    # mean nn-distance + CV over all timesteps
└── frames/
    ├── t0000.png     # 4-row × 5-col comparison figure
    └── ...
```

---

## Training a model

```bash
# Smoke test (~30 s, CPU)
.venv\Scripts\python.exe -m training.trainer `
  --train-config   configs/smoke_test/train_config.yaml `
  --dataset-config configs/smoke_test/dataset_config.yaml `
  --loss-config    configs/smoke_test/loss_config.yaml `
  --model-config   configs/smoke_test/model_config_9.yaml

# Full packed run (GPU recommended)
.venv\Scripts\python.exe -m training.trainer `
  --train-config   configs/trainer_configs/train_config_packed.yaml `
  --dataset-config configs/dataset_configs/dataset_config_packed.yaml `
  --loss-config    configs/loss_configs/loss_config_packed.yaml `
  --model-config   configs/model_configs/model_config_9.yaml
```

---

## Programmatic use

```python
from inference.pipeline import Corrector, CorrectorConfig

cfg       = CorrectorConfig.from_yaml('inference/configs/grid_6x6.yaml')
corrector = Corrector(cfg)

import numpy as np
pts = np.load('inference/sph_data/positions_without.npy')[300]  # (2500, 2)
corrected = corrector.apply(pts, k=5)                            # (2500, 2)
```

---

## Project structure

```
├── data/               # online dataset generators (no stored files)
├── models/fixed_rd/    # model9.py — current architecture
├── training/           # trainer.py, loss.py (hybrid_loss only)
├── inference/
│   ├── pipeline/       # pbc.py, tiling.py, scaling.py, corrector.py
│   ├── visualization/  # comparison.py, tiling.py, training.py, timeseries.py
│   ├── configs/        # experiment configs (reference grid_configs/)
│   ├── experiments/    # auto-created per run
│   └── sph_data/       # positions.npy, positions_without.npy
├── configs/            # all YAML configs (dataset, model, loss, trainer, grid)
├── notebooks/          # pipeline_smoke_test.ipynb
├── docs/               # guide.html ← start here
├── analysis/           # comparison scripts + outputs/
└── sweeps/             # sweep_*.py + tracker.py (Streamlit dashboard)
```
