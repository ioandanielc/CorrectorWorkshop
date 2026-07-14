# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A PyTorch framework for a **Poisson-disk corrector model**: given a 2D or 3D point cloud that violates the minimum-distance constraint `rd` between all pairs, model9 predicts per-point displacement vectors to produce a valid cloud. Applied to real SPH simulation data (replacing/supplementing the Transport Velocity algorithm) and to MD initial states (repairing too-close pairs before the LS1 solver runs — the `olga_init` experiment).

**This is the `simplify` branch** — stripped down to the model9 train/infer pipeline plus the experiments. The full history (model6-8, model10, model11/variable-rd, sweep scripts, old logs/notebooks) lives on `main`.

---

## Project structure

All code lives under `src/`; all generated or external data lives under `artifacts/` (gitignored). Run everything from the project root — paths in configs and code are root-relative; scripts put `src/` on `sys.path` themselves.

```
CorrectorWorkshop/
├── src/
│   ├── configs/
│   │   ├── training/            # dataset/ loss/ model/ trainer/ YAML sets + smoke_test/ (fast CPU variants)
│   │   └── experiments/         # one subfolder per experiment, one YAML per variant
│   │       ├── sph_tv/          #   grid_6x6.yaml (n100, recommended), grid_10x10.yaml (n50)
│   │       ├── apply_corrector/ #   grid_6x6.yaml (corrector blocks only)
│   │       ├── obstruction/     #   grid_6x6.yaml (corrector blocks only)
│   │       └── olga_init/       #   n50/n100 × grid/kdtree/grid_then_kdtree variants
│   ├── models/
│   │   ├── architectures/model9/
│   │   │   ├── model9.py        # THE model — violation-weighted edge net; 2D/3D via input_dim
│   │   │   └── invariance.py    # invariant frame (center + PCA-rotate, NO scaling), applied before every model call
│   │   └── weights/             # production checkpoints model9_*.pt, tracked in git (own README.md)
│   ├── training/
│   │   ├── trainer.py           # training loop: online data, K-step unrolling, dual eval
│   │   ├── loss.py              # hybrid_loss: linear violation penalty + displacement regulariser
│   │   └── datagen.py           # PoissonDiskDataset + PackedPoissonDiskDataset (online generation)
│   ├── inference/
│   │   ├── correctors/
│   │   │   ├── base.py          # Corrector / Experiment ABCs
│   │   │   ├── common/          # pbc.py (3^dim images), scaling.py (rd_train/rd_test), tiling.py (n_cells^dim tiles)
│   │   │   ├── grid/corrector.py       # GridCorrector2D/3D — tiling + ghost buffer + scaling + model (shared ND body)
│   │   │   ├── kdtree/kdtree_corrector.py  # KDTreeCorrector2D/3D — violation-targeted greedy sweeps, k = cap
│   │   │   ├── tv/tv_corrector.py      # TVCorrector2D / FastTVCorrector2D (cKDTree, ~125x faster) — TV baseline
│   │   │   └── pure/pure_inference.py  # PureInference2D — bare model9 round trip, N == N_train, no tiling/PBC
│   │   └── experiments/         # one subfolder per experiment, each with a small README
│   │       ├── sph_tv/          # SPHTVExperiment — ML corrector vs TV baseline across K values
│   │       ├── apply_corrector/ # correct every SPH timestep, save output (+ h5part scratch tools)
│   │       ├── obstruction/     # ObstructionExperiment + demo + obstruction.py (masks + ghost fill)
│   │       ├── olga_init/       # OlgaInitExperiment — 3D MD-init repair, RDF before/after
│   │       └── pbc_toy/         # standalone synthetic proof of the ghost-buffer approach
│   └── utils/
│       ├── config.py, logger.py # config loading, logging
│       └── visualizations/
│           ├── training_visualizations/   # sample plots, evolution GIF, finished-run plots
│           ├── inference_visualizations/  # comparison (4x5), timeseries, enhanced (3-panel per-apply), tiling
│           └── misc/                      # empty placeholder
├── artifacts/                   # gitignored — every input and run output
│   ├── training/                # train_run_<timestamp>/ from the trainer
│   └── inference/
│       ├── experiments/<name>/  # data/ (external inputs) + runs/ (outputs), per experiment
│       └── misc/                # enhanced_viz/ diagnostics, one-off figures
├── requirements.txt
└── README.md                    # quickstart + full structure/config walkthrough
```

---

## Training

**Smoke test (CPU, ~30 s):**
```
.venv\Scripts\python.exe src/training/trainer.py ^
  --train-config   src/configs/training/smoke_test/train_config.yaml ^
  --dataset-config src/configs/training/smoke_test/dataset_config.yaml ^
  --loss-config    src/configs/training/smoke_test/loss_config.yaml ^
  --model-config   src/configs/training/smoke_test/model_config_9.yaml
```

Full runs: swap in one of the four config sets (one per checkpoint — table in README "Training"). Writes to `artifacts/training/train_run_<timestamp>/` (gitignored — copy `model_final.pt` into `src/models/weights/` to keep it).

---

## Production checkpoints

Four deployable checkpoints in `src/models/weights/`, named `model9_<training setup>` (details in `src/models/weights/README.md`):

| Checkpoint | Dim | N_train | rd_train | Used by |
|---|---|---|---|---|
| **model9_n100_p050** | 2D | ~100 | 0.076 | `src/configs/experiments/sph_tv/grid_6x6.yaml` — primary 2D, use by default |
| **model9_n50_sparse** | 2D | 50 | 0.05 | `src/configs/experiments/sph_tv/grid_10x10.yaml` |
| **model9_3d_n50** | 3D | 50 | 0.15 | `src/configs/experiments/olga_init/n50_*.yaml` — primary 3D |
| **model9_3d_n100** | 3D | 100 | 0.12 | `src/configs/experiments/olga_init/n100_*.yaml` — for denser clouds |

Don't cross checkpoint and model config — `hidden_dim` / `max_displacement` differ, will either error or silently produce garbage.

---

## Key design decisions

**Training:**
- Data generated **online** per batch — no dataset file on disk.
- Invariant frame (`src/models/architectures/model9/invariance.py`) = center (subtract mean) + PCA-rotate. Does NOT scale. Applied before every model call, at training and inference.
- Loss operates in invariant space. `lambda2 = lambda1 / (N−1) / 10` keeps displacement reg a constant fraction of violation penalty regardless of N.
- Training uses K-step unrolling (backprop through all K steps). Eval reports both K=1 and K=unroll_steps.

**Inference (all correctors):**

0. **Domain inference**: correctors don't take a `domain` config — the input cloud is centered on its own centroid and the domain is the largest axis extent, fresh on every `apply()` call. Works regardless of the input coordinate frame.

1. **Grid corrector**: split the inferred domain into `n_cells^dim` tiles so each tile has N_tile ≈ N_train. Each tile is extended by `ghost_width = ghost_factor * cell_size` (a fraction of a tile's own size, not of rd_test); all `3^dim` periodic images are checked and images falling in the extended tile become ghosts. Only core (non-ghost) displacements are kept. Correctness requires `ghost_width ≥ rd_test` — the corrector warns if violated. In 3D ghost overhead is ~2.7x core, so pick `n_cells` from **total** (core+ghost) ≈ N_train.

2. **Coordinate scaling**: `scale = rd_train / rd_test`. Multiply coords by `scale` before the invariant transform; divide displacements by `scale` after reverting. Maps violations into the model's training distribution.

3. **KDTree corrector**: runs the model only around violations. One sweep: PBC `cKDTree(boxsize)` → violating points worst-first → greedy claiming — a site's neighbourhood is its `total_core` (≈ N_train) nearest points, its `inner_core` most central unclaimed points receive displacements, the outer ring is frozen context → one rectangular batched model call. `apply(points, k)`: k is a **cap** — early stop once clean, sweeps get cheaper as violations shrink. Warns when a site's frozen ring is thinner than rd_test.

4. **Iterative application**: `apply(pts, k=...)` = K passes, diminishing returns. Grid K=3–5 outperforms TV from t≈300 onwards on the SPH trajectory. Measured (2D SPH, N=2500): kdtree k≤10 beats grid K=5 on quality (mean nn 0.0184–0.0185 vs 0.0170–0.0180); grid K=5 then kdtree k≤10 composes best (mean nn ≈ 0.0186–0.0187) — exposed as `corrector: grid_then_kdtree` in olga_init.

**Interfaces (`src/inference/correctors/base.py`):**
- `Corrector` — one method `apply(points, k=1) -> points`. Implemented by `GridCorrector2D/3D`, `KDTreeCorrector2D/3D`, `TVCorrector2D`/`FastTVCorrector2D`, `PureInference2D`. The grid/kdtree 2D/3D pairs are thin `DIM = 2/3` subclasses of private ND bodies. `enhanced_visualization` is 2D-only (warned and ignored on 3D). Every ML corrector's config carries its own `checkpoint` + `model_config` + `rd_train`; all correctors and configs re-export from `inference.correctors`.
- `Experiment` — one method `run()`. Implemented by `SPHTVExperiment`, `ObstructionExperiment`, `OlgaInitExperiment`. Experiment-loop settings (`stride`, `k_values`, `corrector` kind…) live on the experiments, not on corrector configs.

**Obstructions**: `src/inference/experiments/obstruction/obstruction.py` fills obstacle interiors (ellipse/polygon/gear masks) with ghost particles at spacing `rd` so the corrector pushes real particles away from the boundary; ghosts are dropped after `apply()`.

**Run an experiment** (config schema: README "Configs"; every YAML's header comment shows its run command):
```
.venv\Scripts\python.exe src/inference/experiments/sph_tv/sph_tv_experiment.py src/configs/experiments/sph_tv/grid_6x6.yaml
.venv\Scripts\python.exe src/inference/experiments/olga_init/olga_init_experiment.py src/configs/experiments/olga_init/n50_kdtree.yaml
```

---

## Metrics (training eval)

| Metric | Target |
|---|---|
| `mean_violation` — avg `relu(rd−dist)` | → 0 |
| `illegal_pairs %` | → 0% |
| `viol_reduction %` | → 100% |
| `mean_nn_dist` | ≥ rd, not >> rd |
| `correction_eff` — violation removed ÷ displacement | high |

For 3D MD data, `olga_init` reports RDF g(r) + min pair distance + pairs<rd before/after.

---

## Known rough edges

- `artifacts/inference/experiments/<name>/data/` inputs are not in git: SPH trajectories under `sph_tv/data/`, Olga's `data.npy` under `olga_init/data/`.
- Everything under `artifacts/inference/experiments/*/runs/` and `artifacts/training/` is regenerable run output; delete freely.
- `src/inference/experiments/apply_corrector/inspector.py` has a hardcoded local path — scratch script, not a reusable tool.
- Open issue (found via enhanced viz): on lattice-like inputs (e.g. t=0) the inferred domain (max extent) undershoots the true PBC box, pinching the wrap seam into artificial violations — grid corrector degrades mean nn at t=0 (0.0184 → 0.0174); self-heals by t≈300. Fix TBD (pad extent by one nn-spacing?).
- The shipped `dataset_config_packed.yaml` (rd=0.05 → N≈231) does not match what `model9_n100_p050` recorded at training time (rd_train 0.076, N≈100) — the config drifted after that run. The 2D checkpoints also predate the 2026-07-13 frozen-seed data-generation fix (see src/models/weights/README.md).

## Current state

Training is complete on model9 (2D ×2, 3D ×2). Active work is inference-side:
- **2D SPH**: grid 6×6 + n100 K=5 reaches mean_nn ≈ 0.018 at t=700 vs TV ≈ 0.016 (+13%); kdtree and the grid→kdtree composition beat it (numbers above).
- **3D / olga_init**: correct Olga's LS1 initial state (N=5096, L=18, T=1.4, rho=0.874; min pair distance 0.08 spikes potential energy, ~100 extra MD timesteps to equilibrate). Sweep checkpoints (n50/n100) × correctors (grid/kdtree/grid_then_kdtree) × rd_test, compare RDFs; Olga re-runs LS1 on the corrected state and counts timesteps to reach U ≈ −4.61.
- Roadmap: `MixedCorrector` as a first-class corrector (grid→kdtree composition currently lives only in olga_init); domain-inference fix for lattice-like inputs.
