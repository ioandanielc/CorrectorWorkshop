# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A PyTorch framework for a **Poisson-disk corrector model**: given a 2D or 3D point cloud that violates the minimum-distance constraint `rd` between all pairs, a corrector network predicts per-point displacement vectors to produce a valid cloud. Two networks: **model9** (violation-weighted edge net) and **model12** (physics-informed — adds an SPH kernel-gradient-symmetry loss so corrected clouds are valid SPH simulation restarts). Applied to real SPH simulation data, replacing/supplementing the Transport Velocity algorithm.

**This is the `sph-use-case` branch** — the 2D SPH experiments only (SPH-trajectory correction + obstruction), branched from `simplify`. The 3D MD-init experiment (`olga_init`) and its `model9_3d_*` checkpoints live on `simplify`; the full model history (model6-8, model10, model11/variable-rd, sweep scripts, old logs/notebooks) lives on `main`. The corrector code still carries the 2D/3D shared bodies, so future 3D SPH work stays possible.

---

## Project structure

All code lives under `src/`; all generated or external data lives under `artifacts/` (gitignored). Run everything from the project root — paths in configs and code are root-relative; scripts put `src/` on `sys.path` themselves.

```
CorrectorWorkshop/
├── src/
│   ├── configs/
│   │   ├── training/            # dataset/ loss/ model/ trainer/ YAML sets + smoke_test/ (fast CPU variants)
│   │   └── experiments/         # one subfolder per experiment, one YAML per variant
│   │       ├── sph_tv/          #   model9: grid_6x6 (n100, default) / grid_10x10 (n50); model12_{grid,kdtree,grid_then_kdtree,wholecloud}
│   │       ├── apply_corrector/ #   grid_6x6.yaml (corrector blocks only)
│   │       └── obstruction/     #   grid_6x6.yaml (corrector blocks only)
│   ├── models/
│   │   ├── architectures/
│   │   │   ├── model9/          # model9.py (violation-weighted edge net; 2D/3D via input_dim) + invariance.py (center + PCA-rotate)
│   │   │   └── model12/         # model12.py — SPH corrector: L-round message passing + KG-symmetry loss; forward_sparse (whole-cloud)
│   │   └── weights/             # production checkpoints model9_*.pt + model12_sph_l4.pt, tracked in git (own README.md)
│   ├── training/
│   │   ├── trainer.py           # training loop: online data, K-step unrolling, dual eval
│   │   ├── loss.py              # hybrid_loss (model9) + rdsph_loss/sph_loss (model12: + λ3·SPH kernel-gradient symmetry); KG primitive re-exported from utils/metrics.py
│   │   └── datagen.py           # PoissonDiskDataset + PackedPoissonDiskDataset (online generation)
│   ├── inference/
│   │   ├── correctors/
│   │   │   ├── base.py          # Corrector / Experiment ABCs
│   │   │   ├── common/          # pbc.py (3^dim images), scaling.py (rd_train/rd_test), tiling.py (n_cells^dim tiles)
│   │   │   ├── grid/corrector.py       # GridCorrector2D/3D — tiling + ghost buffer + scaling + model (shared ND body)
│   │   │   ├── kdtree/kdtree_corrector.py  # KDTreeCorrector2D/3D — violation-targeted greedy sweeps, k = cap
│   │   │   ├── tv/tv_corrector.py      # TVCorrector2D / FastTVCorrector2D (cKDTree, ~125x faster) — TV baseline
│   │   │   ├── wholecloud/wholecloud_corrector.py  # WholeCloudCorrector2D/3D — one forward_sparse call per pass, no tiles/seams; THE model12 deployment path
│   │   │   └── pure/pure_inference.py  # PureInference2D — bare model9 round trip, N == N_train, no tiling/PBC
│   │   └── experiments/         # one subfolder per experiment, each with a small README
│   │       ├── sph_tv/          # SPHTVExperiment (model9 vs TV) + sph_model12_experiment (model12 quality check) + kg_sweep.py (full-trajectory KG/nn metrics -> metrics.csv)
│   │       ├── apply_corrector/ # correct every SPH timestep, save output (+ h5part scratch tools)
│   │       ├── obstruction/     # ObstructionExperiment + demo + obstruction.py (masks + ghost fill)
│   │       └── pbc_toy/         # standalone synthetic proof of the ghost-buffer approach
│   └── utils/
│       ├── config.py, logger.py # config loading, logging
│       ├── metrics.py           # shared KG primitive (quintic kernel, torch, batched) + numpy helpers (mean_kg, nn_dists, mean_nn, illegal_frac)
│       └── visualizations/
│           ├── training_visualizations/   # sample plots, evolution GIF, finished-run plots
│           └── inference_visualizations/  # comparison (4x5), timeseries, ml_vs_tv, enhanced (3-panel per-apply), tiling
├── tests/                       # test_wholecloud.py — WholeCloudCorrector2D must reproduce the sim-validated artifact bit-exactly (needs artifacts on disk)
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

Three deployable 2D checkpoints in `src/models/weights/` (details in `src/models/weights/README.md`):

| Checkpoint | N_train | rd_train | Used by |
|---|---|---|---|
| **model9_n100_p050** | ~100 | 0.076 | `src/configs/experiments/sph_tv/grid_6x6.yaml` — primary model9, use by default |
| **model9_n50_sparse** | 50 | 0.05 | `src/configs/experiments/sph_tv/grid_10x10.yaml` |
| **model12_sph_l4** | 49 | 0.14 (rd); attention_rd 0.286 | `src/configs/experiments/sph_tv/model12_*.yaml` — physics-informed (SPH KG symmetry) |

Don't cross checkpoint and model config — `hidden_dim` / `max_displacement` differ, will either error or silently produce garbage. model12 also has a distinct calling convention (`rd=attention_rd`, `box=`).

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

4. **Iterative application**: `apply(pts, k=...)` = K passes, diminishing returns. Grid K=3–5 outperforms TV from t≈300 onwards on the SPH trajectory. Measured (2D SPH, N=2500): kdtree k≤10 beats grid K=5 on quality (mean nn 0.0184–0.0185 vs 0.0170–0.0180); grid K=5 then kdtree k≤10 composes best (mean nn ≈ 0.0186–0.0187) — exposed as `corrector: grid_then_kdtree` in the sph_tv model12 experiment (`model12_grid_then_kdtree.yaml`).

**Interfaces (`src/inference/correctors/base.py`):**
- `Corrector` — one method `apply(points, k=1) -> points`. Implemented by `GridCorrector2D/3D`, `KDTreeCorrector2D/3D`, `WholeCloudCorrector2D/3D`, `TVCorrector2D`/`FastTVCorrector2D`, `PureInference2D`. The grid/kdtree/wholecloud 2D/3D pairs are thin `DIM = 2/3` subclasses of private ND bodies. `enhanced_visualization` is 2D-only (warned and ignored on 3D). Every ML corrector's config carries its own `checkpoint` + `model_config` + `rd_train`; all correctors and configs re-export from `inference.correctors`.
- `WholeCloudCorrector2D/3D` — one `forward_sparse` call over the entire cloud per pass (PBC cKDTree edge list at `attention_rd`); requires a model with `forward_sparse` (model12). Takes an optional explicit `data.box` — give the true PBC box when known (the SPH case does), which sidesteps the domain-undershoot issue below. Guarded by `tests/test_wholecloud.py`: must reproduce the sim-validated whole-cloud artifact bit-exactly.
- `Experiment` — one method `run()`. Implemented by `SPHTVExperiment` and `ObstructionExperiment` (the model12 sph_tv script uses the same `experiment.corrector` step-builder pattern). Experiment-loop settings (`stride`, `k_values`, `corrector` kind…) live on the experiments, not on corrector configs.

**Obstructions**: `src/inference/experiments/obstruction/obstruction.py` fills obstacle interiors (ellipse/polygon/gear masks) with ghost particles at spacing `rd` so the corrector pushes real particles away from the boundary; ghosts are dropped after `apply()`.

**Run an experiment** (config schema: README "Configs"; every YAML's header comment shows its run command):
```
.venv\Scripts\python.exe src/inference/experiments/sph_tv/sph_tv_experiment.py src/configs/experiments/sph_tv/grid_6x6.yaml
.venv\Scripts\python.exe src/inference/experiments/sph_tv/sph_model12_experiment.py src/configs/experiments/sph_tv/model12_grid.yaml
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

For model12, SPH-restart quality is additionally judged by `mean_kg_norm` (kernel-gradient symmetry, → 0) — the in-repo SPH-consistency proxy.

---

## Known rough edges

- `artifacts/inference/experiments/<name>/data/` inputs are not in git: the SPH trajectory lives under `sph_tv/data/`.
- Everything under `artifacts/inference/experiments/*/runs/` and `artifacts/training/` is regenerable run output; delete freely — EXCEPT `sph_tv/for_sim/` (sim-validated reference states; `tests/test_wholecloud.py` and `kg_sweep.py` read `positions_model12_corrected.npy`), `apply_corrector/runs/positions_corrected_K5.npy` (model9 series in the KG sweep) and the checkpoint-provenance training runs named in `src/models/weights/README.md`.
- Open issue (found via enhanced viz): on lattice-like inputs (e.g. t=0) the inferred domain (max extent) undershoots the true PBC box, pinching the wrap seam into artificial violations — grid corrector degrades mean nn at t=0 (0.0184 → 0.0174); self-heals by t≈300. Fix TBD (pad extent by one nn-spacing?).
- The shipped `dataset_config_packed.yaml` (rd=0.05 → N≈231) does not match what `model9_n100_p050` recorded at training time (rd_train 0.076, N≈100) — the config drifted after that run. The 2D checkpoints also predate the 2026-07-13 frozen-seed data-generation fix (see src/models/weights/README.md).

## Current state

This branch is the **2D SPH use case**. Two experiments in scope: the SPH-trajectory corrector (many timesteps, `sph_tv`) and the obstruction demo.

- **model9 vs TV** (`sph_tv`): grid 6×6 + n100 K=5 reaches mean_nn ≈ 0.018 at t=700 vs TV ≈ 0.016 (+13%); kdtree and the grid→kdtree composition beat it (numbers above).
- **model12 (physics-informed — the active thrust)**: adds `λ3·|KG|²` (SPH kernel-gradient symmetry) so corrected clouds are valid SPH restarts — model9's are not (it blows KG up ~3.9×: full-sweep disordered mean 1.278 vs raw 0.326). The whole-cloud sparse pass is now a first-class `WholeCloudCorrector2D` (`corrector: wholecloud`, `model12_wholecloud.yaml`), bit-exact against the sim-validated trajectory (tests/), ~20× faster than tiled (~0.26 s vs 5.8 s per N=2500 timestep). Full-trajectory KG sweep (`kg_sweep.py`, 1002 steps, disordered means): raw 0.326 / TV 0.274 / model12_wc **0.128**, KG floor ≈ 0.111. The KG term is a *soft, training-only* constraint — never enforced at inference. PoC for a physics-informed-ML ("Physics in AI") workshop paper.
- **Obstruction**: to be re-run on this branch.
- Roadmap (ON HOLD pending user walkthrough): 3D model12 (needs 3D quintic KG in utils/metrics.py; synthetic data only for now); ablations grouped + λ3=0 control arm; obstruction re-run; `MixedCorrector`; domain-inference fix for lattice-like inputs (t=0 undershoot — the wholecloud corrector's explicit `box` already avoids it).
