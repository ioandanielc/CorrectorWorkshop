# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A PyTorch framework for a **Poisson-disk corrector model**: given a 2D point cloud that violates the minimum-distance constraint `rd` between all pairs, model9 predicts per-point displacement vectors to produce a valid cloud. The trained model is then applied to real SPH simulation data to replace or supplement the Transport Velocity (TV) algorithm.

**This is the `simplify` branch** — a stripped-down version of the project: model9 train/infer pipeline plus the obstruction/TV-corrector work, no archived model iterations, no sweep tooling, no old logs/notebooks. The full history (model6-8, model10, sweep scripts, training_artifacts, docs) lives on `main`.

---

## Project structure

```
CorrectorWorkshop/
├── data/
│   ├── data_generator.py       # PoissonDiskDataset + PackedPoissonDiskDataset (online generation)
│   └── data_processor.py       # DataProcessor.make_invariant (center + PCA-rotate, NO scaling)
├── models/
│   ├── fixed_rd/model9.py      # CURRENT production model — violation-weighted edge net, trained at one fixed rd
│   └── variable_rd/model11.py  # experimental — extends model9 to generalize across rd (rd-normalized
│                                # edge features + log10(rd) embedding); no production checkpoint yet
├── training/
│   ├── trainer.py              # training loop: online data, iterative unrolling K steps, dual eval
│   └── loss.py                 # hybrid_loss: linear violation penalty + displacement regulariser
├── inference/
│   ├── pipeline/
│   │   ├── corrector.py         # Corrector — tiling + ghost buffer + scaling + model, one apply() call
│   │   ├── tiling.py            # TilingConfig, tile geometry
│   │   ├── pbc.py               # periodic-boundary distance helpers
│   │   ├── scaling.py           # rd_train / rd_test coordinate scaling
│   │   ├── obstruction.py       # domain obstructions (ellipse/polygon/gear masks + ghost-particle fill)
│   │   └── tv_corrector.py      # TVCorrector (naive O(N^2)) + FastTVCorrector (cKDTree, ~125x faster)
│   ├── visualization/
│   │   ├── comparison.py        # 4-row x 5-col comparison figure
│   │   ├── timeseries.py        # mean nn-distance / CV over time
│   │   ├── ml_vs_tv.py          # 3-panel ML-corrector-vs-TV comparison (quality + timing)
│   │   └── tiling.py, training.py  # secondary plots
│   ├── configs/
│   │   ├── grid_6x6.yaml         # n100 model, 6x6 tiling (recommended)
│   │   └── grid_10x10.yaml       # n50 model, 10x10 tiling
│   ├── corrected/
│   │   ├── inspector.py          # scratch script to inspect .h5part contents
│   │   └── replace_positions.py  # overwrite coords_0/coords_1 in an H5Part file from a positions.txt
│   ├── run_experiment.py       # main entry point: comparison figures across K values
│   ├── apply_corrector.py      # apply the corrector to every timestep of the no-TV SPH trajectory, save output
│   ├── apply_single_cloud.py   # apply model9 to a single bounded cloud with N == N_train (no tiling/PBC)
│   ├── obstruction_demo.py     # visualize the three obstruction types (ellipse/polygon/gear)
│   ├── obstruction_experiment.py  # two SPH-around-obstacle init strategies, corrector applied cumulatively
│   ├── pbc_toy.py               # standalone synthetic proof the ghost-buffer approach is correct
│   └── sph_data/                # positions.npy / positions_without.npy — NOT in git, put your own here
├── configs/                     # dataset/model/loss/trainer YAMLs, plus configs/smoke_test/ (fast CPU variants)
├── utils/                       # config loading, logging, training-time plots
├── weights/                     # the two trained checkpoints, tracked in git (see weights/README.md)
└── README.md                    # quickstart + full structure/config walkthrough
```

---

## Training

**Smoke test (CPU, ~30 s):**
```
.venv\Scripts\python.exe -m training.trainer ^
  --train-config   configs/smoke_test/train_config.yaml ^
  --dataset-config configs/smoke_test/dataset_config.yaml ^
  --loss-config    configs/smoke_test/loss_config.yaml ^
  --model-config   configs/smoke_test/model_config_9.yaml
```

**Full packed run (GPU, N≈231, p=0.5 — produced the `n100_p050` checkpoint):**
```
.venv\Scripts\python.exe -m training.trainer ^
  --train-config   configs/trainer_configs/train_config_packed.yaml ^
  --dataset-config configs/dataset_configs/dataset_config_packed.yaml ^
  --loss-config    configs/loss_configs/loss_config_packed.yaml ^
  --model-config   configs/model_configs/model_config_9.yaml
```

Writes to `training_artifacts/train_run_<timestamp>/` (gitignored — copy `model_final.pt` into `weights/` yourself if you want to keep it).

`dataset_config_3.yaml` + `loss_config_5.yaml` + `train_config_2.yaml` is the N=50 sparse setup that produced `n50_sparse.pt`.

---

## Production checkpoints

Two deployable checkpoints, both tracked in `weights/` (details in `weights/README.md`):

| Name | N_train | rd_train | Tiling config | Notes |
|---|---|---|---|---|
| **n100_p050** | ~100 | 0.076 | `inference/configs/grid_6x6.yaml` | packed p=0.50, hd128_d3 — primary, use by default |
| **n50_sparse** | 50 | 0.05 | `inference/configs/grid_10x10.yaml` | sparse p≈0.11 — for 10×10 tiling |

Don't cross checkpoint and model config — `hidden_dim` / `max_displacement` differ, will either error or silently produce garbage. `n100_p050` beats `n50_sparse` across the board despite the theoretically worse tile-size match.

---

## Key design decisions

**Training:**
- Data generated **online** per batch — no dataset file on disk.
- `make_invariant` = center (subtract mean) + PCA-rotate. Does NOT scale. Applied before every model call.
- Loss operates in invariant space. `lambda2 = lambda1 / (N−1) / 10` keeps displacement reg a constant fraction of violation penalty regardless of N.
- Training uses K-step unrolling (backprop through all K steps). Eval reports both K=1 and K=unroll_steps.

**Inference on SPH data (`rd_test=0.02`, `N=2500`, `domain=[0,1]²`, PBC):**

1. **Tiling**: split domain into a `G×G` grid so each tile has `N_tile ≈ N_train` points.
   - 6×6 → ~107 pts/tile (69 core + 38 ghost) ≈ N_train=100 for n100 model.
   - 10×10 → ~49 pts/tile ≈ N_train=50 for n50 model.

2. **Ghost buffer**: each tile is extended by `ghost_width = ghost_factor * rd_test` on all sides. All 9 periodic images of each particle are checked. Any image falling in the extended tile becomes a ghost. After inference, only *core* (non-ghost) displacements are kept. **Correctness**: `ghost_factor ≥ 1.0` guarantees every PBC-violating pair is visible in at least one tile's ghost buffer. Do not set below 1.0.

3. **Coordinate scaling**: `scale = rd_train / rd_test`. Multiply coords by `scale` before `make_invariant`; divide displacements by `scale` after reverting. This maps violations from `rd_test`-scale to `rd_train`-scale so the model operates in its training distribution.

4. **Iterative application**: `Corrector.apply(pts, k=...)` applies K passes. Higher K = more correction, more compute, diminishing returns. K=3–5 outperforms TV from t≈300 onwards on the SPH trajectory.

**Obstructions (domain obstacles — gears, ellipses, polygons):**
- `inference/pipeline/obstruction.py` fills the obstacle interior with ghost particles at spacing `rd` so the corrector sees a uniform-looking environment and naturally pushes real particles away from the boundary. Ghosts are dropped from the output after `corrector.apply()`.
- `inference/pipeline/tv_corrector.py` provides `TVCorrector` (naive O(N²), faithful port of the reference implementation) and `FastTVCorrector` (cKDTree neighbor list, ~125× faster) as the Transport Velocity baseline to compare against, driven by a `tv:` block in the experiment config (`h_factor`, `nmax`, `dt`).

**Run an experiment:**
```
.venv\Scripts\python.exe inference/run_experiment.py inference/configs/grid_6x6.yaml
.venv\Scripts\python.exe inference/run_experiment.py inference/configs/grid_10x10.yaml
.venv\Scripts\python.exe inference/run_experiment.py inference/configs/grid_6x6.yaml --timestep 300
```

---

## Experiment config schema

```yaml
model:
  checkpoint: weights/n100_p050.pt
  config:     configs/model_configs/model_config_9_n100_p050.yaml
  rd_train:   0.076

data:
  without_tv: inference/sph_data/positions_without.npy
  with_tv:    inference/sph_data/positions.npy
  rd_test:    0.02
  domain:     1.0

tiling:
  grid_size:    6
  ghost_factor: 1.0   # ghost_width = ghost_factor * rd_test

experiment:
  stride:   100
  k_values: [1, 2, 3, 5]
  device:   cpu

tv:
  h_factor: 1.3   # h = h_factor * dx,  dx = domain / sqrt(N)
  nmax:     10
  dt:       0.2   # relaxation factor (matches reference implementation)
```

Copy one of the two files in `inference/configs/` and change what you need — no code changes required to run a variant.

---

## Model architecture (model9 — production)

```
Input: (B, N, D)  — batch of point clouds in invariant space
For every pair (i,j):
  edge_feat = [rel_pos(D), dist(1), violation(1)]  — violation = relu(rd - dist)
  edge_emb  = edge_mlp(edge_feat)                  — 3-layer MLP, hidden_dim=128
  weight_ij = violation_ij / sum_j(violation_ij)   — violation-weighted attention
agg_i  = sum_j(weight_ij * edge_emb_ij)            — zero for non-violating neighbours
disp_i = tanh(output_mlp(agg_i)) * max_displacement
Output: (B, N, D)  — displacement vectors in invariant space
```

`uses_rd = True` — `rd` is passed as a scalar tensor at every forward call. `model11` (`models/variable_rd/`) is the same idea extended to generalize across a ~100x range of `rd` (0.01–1.0) via rd-normalized edge features and a `log10(rd)` embedding — experimental, no production checkpoint yet.

---

## Metrics (training eval)

| Metric | Target |
|---|---|
| `mean_violation` — avg `relu(rd−dist)` | → 0 |
| `illegal_pairs %` | → 0% |
| `viol_reduction %` | → 100% |
| `mean_nn_dist` | ≥ rd, not >> rd |
| `correction_eff` — violation removed ÷ displacement | high |

---

## Known rough edges

- `inference/sph_data/*.npy` must exist locally — not part of this repo.
- `training_artifacts/` and `inference/experiments/` are gitignored run output. Nothing there is source; delete freely.
- `inference/corrected/inspector.py` has a hardcoded local path — scratch script, not a reusable tool.

## Current state

Training is complete on model9. Active work is on SPH inference and obstruction handling:
- **6×6 grid, n100 model**: K=5 reaches mean_nn ≈ 0.018 at t=700 vs TV ≈ 0.016 (+13%)
- **10×10 grid, n50 model**: similar but slightly weaker than 6×6
- Obstruction/TV-corrector work (ghost-particle obstacle fill, TV baseline) just ported from `main`
- A more direct/simplified entry point for applying the corrector to the grid tiling cases is planned but not yet started
