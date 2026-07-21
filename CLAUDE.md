# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A PyTorch framework for a **Poisson-disk corrector model**: given a 2D or 3D point cloud that violates the minimum-distance constraint `rd` between all pairs, a corrector network predicts per-point displacement vectors to produce a valid cloud. One network: **model12** (physics-informed — L-round message passing + an SPH kernel-gradient-symmetry loss so corrected clouds are valid SPH simulation restarts). Applied to real SPH simulation data, replacing/supplementing the Transport Velocity algorithm.

**This is the `sph-use-case` branch** — model12 / 2D SPH only (SPH-trajectory correction + obstruction), branched from `simplify`. The **model9 family was removed from this branch** (2026-07-21): its 2D checkpoints, the center+PCA invariant frame, the model9-vs-TV experiment, pure/pbc_toy/apply_corrector — all live on `simplify`/`main` (along with the 3D MD-init `olga_init` and the older model history). model9's role here is historical: its corrected trajectory is kept as an artifact for the KG comparison (it blows KG up ~3.9× — the motivating failure). The corrector code keeps the 2D/3D shared bodies, so future 3D SPH work stays possible.

---

## Project structure

All code lives under `src/`; all generated or external data lives under `artifacts/` (gitignored). Run everything from the project root — paths in configs and code are root-relative; scripts put `src/` on `sys.path` themselves.

```
CorrectorWorkshop/
├── src/
│   ├── configs/
│   │   ├── training/            # dataset/ loss/ model/ trainer/ YAML sets + smoke_test/ (fast CPU variants); loss/ holds the λ3 ablation ladder
│   │   └── experiments/         # one subfolder per experiment, one YAML per variant
│   │       ├── sph_tv/          #   model12_{grid,kdtree,grid_then_kdtree,wholecloud}
│   │       └── obstruction/     #   grid_6x6.yaml (corrector blocks only, retargeted to model12 — untested until the re-run)
│   ├── models/
│   │   ├── architectures/
│   │   │   └── model12/         # model12.py — SPH corrector: L-round message passing + KG-symmetry loss; forward_sparse (whole-cloud)
│   │   └── weights/             # production checkpoint model12_sph_l4.pt, tracked in git (own README.md)
│   ├── training/
│   │   ├── trainer.py           # training loop: online data, K-step unrolling, dual eval
│   │   ├── loss.py              # rdsph_loss / sph_loss (λ3·SPH kernel-gradient symmetry); KG primitive re-exported from utils/metrics.py
│   │   └── datagen.py           # PoissonDiskDataset (online generation; periodic=True = the SPH lattice regime)
│   ├── inference/
│   │   ├── correctors/
│   │   │   ├── base.py          # Corrector / Experiment ABCs
│   │   │   ├── common/          # pbc.py (3^dim images), scaling.py (rd_train/rd_test), tiling.py (n_cells^dim tiles)
│   │   │   ├── grid/corrector.py       # GridCorrector2D/3D — tiling + ghost buffer + scaling + model (shared ND body)
│   │   │   ├── kdtree/kdtree_corrector.py  # KDTreeCorrector2D/3D — violation-targeted greedy sweeps, k = cap
│   │   │   ├── tv/tv_corrector.py      # TVCorrector2D / FastTVCorrector2D (cKDTree, ~125x faster) — TV baseline
│   │   │   └── wholecloud/wholecloud_corrector.py  # WholeCloudCorrector2D/3D — one forward_sparse call per pass, no tiles/seams; THE deployment path
│   │   └── experiments/         # one subfolder per experiment, each with a small README
│   │       ├── sph_tv/          # sph_model12_experiment (grid/kdtree/grid_then_kdtree/wholecloud) + kg_sweep.py (full-trajectory KG/nn metrics -> metrics.csv)
│   │       └── obstruction/     # ObstructionExperiment + demo + obstruction.py (masks + ghost fill)
│   └── utils/
│       ├── config.py, logger.py # config loading, logging
│       ├── metrics.py           # shared KG primitive (quintic kernel, torch, batched) + numpy helpers (mean_kg, nn_dists, mean_nn, illegal_frac)
│       └── visualizations/
│           ├── training_visualizations/   # sample plots, evolution GIF, finished-run plots
│           └── inference_visualizations/  # tiling diagnostics + enhanced (3-panel per-apply)
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

**Smoke test (CPU, ~5 s):**
```
.venv\Scripts\python.exe src/training/trainer.py ^
  --train-config   src/configs/training/smoke_test/train_config.yaml ^
  --dataset-config src/configs/training/smoke_test/dataset_config_sph.yaml ^
  --loss-config    src/configs/training/smoke_test/loss_config_rdsph.yaml ^
  --model-config   src/configs/training/smoke_test/model_config_12.yaml
```

Full run (the `model12_sph_l4` recipe — ~10 min on GPU): `train_config_sph_adamw` + `dataset_config_sph` + `loss_config_rdsph_lam3_0p27` + `model_config_12_sph_L4`. The `loss_config_rdsph_lam3_*.yaml` ladder (0.03/0.09/0.27/0.90) is the λ3 ablation axis. Writes to `artifacts/training/train_run_<timestamp>/` (gitignored — copy the checkpoint into `src/models/weights/` to keep it; `model_best.pt` = best val loss, `model_final.pt` = last iterate).

---

## Production checkpoint

One deployable checkpoint in `src/models/weights/` (details in `src/models/weights/README.md`):

| Checkpoint | N_train | rd_train | Used by |
|---|---|---|---|
| **model12_sph_l4** | 49 (7×7 lattice) | 0.14 (rd); attention_rd 0.286 | `src/configs/experiments/sph_tv/model12_*.yaml` — physics-informed (SPH KG symmetry) |

Don't cross checkpoint and model config — `hidden_dim` / `max_displacement` differ, will either error or silently produce garbage. Calling convention: `rd=attention_rd` (not the constraint rd), `box=` for periodic geometry; all correctors carry the adapter.

---

## Key design decisions

**Training:**
- Data generated **online** per batch — no dataset file on disk. The SPH regime (`periodic: true`) enforces rd under minimum-image on the unit torus; at that packing, clean clouds are a randomly translated square lattice and the noise provides the disorder.
- No invariant-frame transform: model12 is translation-invariant by construction (it only sees relative positions, `rel = x_j − x_i`) and trains in the fixed unit-torus frame. It is NOT rotation-equivariant — deliberate for a fixed simulation frame (the model9-era center+PCA machinery lives on `main`/`simplify`).
- `lambda2 = lambda1 / (N−1) / 10` keeps displacement reg a constant fraction of violation penalty regardless of N; `lambda3` is the violation↔KG-symmetry trade-off dial (λ3 ladder in `configs/training/loss/`).
- Training uses K-step unrolling (backprop through all K steps). Eval reports both K=1 and K=unroll_steps.

**Inference (all correctors):**

0. **Domain inference**: the tiled correctors (grid/kdtree) don't take a `domain` config — the input cloud is centered on its own centroid and the domain is the largest axis extent, fresh on every `apply()` call; works regardless of the input coordinate frame. The wholecloud corrector additionally accepts an explicit `data.box` (preferred when the true PBC box is known — see rough edges).

1. **Grid corrector**: split the inferred domain into `n_cells^dim` tiles so each tile has N_tile ≈ N_train. Each tile is extended by `ghost_width = ghost_factor * cell_size` (a fraction of a tile's own size, not of rd_test); all `3^dim` periodic images are checked and images falling in the extended tile become ghosts. Only core (non-ghost) displacements are kept. Correctness requires `ghost_width ≥ rd_test` — the corrector warns if violated. In 3D ghost overhead is ~2.7x core, so pick `n_cells` from **total** (core+ghost) ≈ N_train.

2. **Coordinate scaling**: `scale = rd_train / rd_test`. Multiply coords by `scale` before the invariant transform; divide displacements by `scale` after reverting. Maps violations into the model's training distribution.

3. **KDTree corrector**: runs the model only around violations. One sweep: PBC `cKDTree(boxsize)` → violating points worst-first → greedy claiming — a site's neighbourhood is its `total_core` (≈ N_train) nearest points, its `inner_core` most central unclaimed points receive displacements, the outer ring is frozen context → one rectangular batched model call. `apply(points, k)`: k is a **cap** — early stop once clean, sweeps get cheaper as violations shrink. Warns when a site's frozen ring is thinner than rd_test.

4. **Iterative application**: `apply(pts, k=...)` = K passes, diminishing returns. Measured with model12 (2D SPH, N=2500): kdtree and grid_then_kdtree edge out grid alone on mean nn (~0.0182 vs ~0.0178); **wholecloud beats everything on KG and speed** (~0.26 s vs ~5.8 s per timestep) and is the deployment path — grid/kdtree remain as the tiling comparison.

**Interfaces (`src/inference/correctors/base.py`):**
- `Corrector` — one method `apply(points, k=1) -> points`. Implemented by `GridCorrector2D/3D`, `KDTreeCorrector2D/3D`, `WholeCloudCorrector2D/3D`, `TVCorrector2D`/`FastTVCorrector2D`. The 2D/3D pairs are thin `DIM = 2/3` subclasses of private ND bodies. All ML correctors require a **box-aware** model (`uses_box`, model12-style) and raise otherwise; `model_file` is required in the model config (no default architecture). `enhanced_visualization` is 2D-only (warned and ignored on 3D). Every ML corrector's config carries its own `checkpoint` + `model_config` + `rd_train`; all correctors and configs re-export from `inference.correctors`.
- `WholeCloudCorrector2D/3D` — one `forward_sparse` call over the entire cloud per pass (PBC cKDTree edge list at `attention_rd`); requires a model with `forward_sparse`. Takes an optional explicit `data.box` — give the true PBC box when known (the SPH case does), which sidesteps the domain-undershoot issue below. Guarded by `tests/test_wholecloud.py`: must reproduce the sim-validated whole-cloud artifact bit-exactly. (Grid/kdtree keep an add-then-subtract displacement arithmetic on purpose — bit-identical to the historically validated outputs.)
- `Experiment` — one method `run()`. Implemented by `ObstructionExperiment`; the sph_tv model12 script uses the same `experiment.corrector` step-builder pattern. Experiment-loop settings (`stride`, `k_*`, `corrector` kind…) live on the experiments, not on corrector configs.

**Obstructions**: `src/inference/experiments/obstruction/obstruction.py` fills obstacle interiors (ellipse/polygon/gear masks) with ghost particles at spacing `rd` so the corrector pushes real particles away from the boundary; ghosts are dropped after `apply()`.

**Run an experiment** (config schema: README "Configs"; every YAML's header comment shows its run command):
```
.venv\Scripts\python.exe src/inference/experiments/sph_tv/sph_model12_experiment.py src/configs/experiments/sph_tv/model12_wholecloud.yaml
.venv\Scripts\python.exe src/inference/experiments/sph_tv/kg_sweep.py
.venv\Scripts\python.exe tests/test_wholecloud.py
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

SPH-restart quality is additionally judged by `mean_kg_norm` (kernel-gradient symmetry, → 0) — the in-repo SPH-consistency proxy. The primitive lives in `src/utils/metrics.py` (shared by the loss and the experiments); note lattice-validation KG (N=49, ~0.02) and real-data disordered KG (N=2500, ~0.13) are different regimes — never compare them directly.

---

## Known rough edges

- `artifacts/inference/experiments/<name>/data/` inputs are not in git: the SPH trajectory lives under `sph_tv/data/`.
- Everything under `artifacts/inference/experiments/*/runs/` and `artifacts/training/` is regenerable run output; delete freely — EXCEPT `sph_tv/for_sim/` (sim-validated reference states; `tests/test_wholecloud.py` and `kg_sweep.py` read `positions_model12_corrected.npy`), `apply_corrector/runs/positions_corrected_K5.npy` (model9 series in the KG sweep) and the checkpoint-provenance training runs named in `src/models/weights/README.md`.
- Open issue: on lattice-like inputs (e.g. t=0) the inferred domain (max extent) undershoots the true PBC box, pinching the wrap seam into artificial violations — affects the tiled correctors (grid/kdtree). The wholecloud corrector avoids it by taking the true box explicitly (`data.box`); measured at t=0 it leaves the cloud essentially untouched (nn 0.0200 → 0.0200) where the grid corrector degraded it.
- `obstruction/grid_6x6.yaml` was retargeted from the removed model9 checkpoint to model12 — not yet re-run/tuned (tiling + ghost_factor unvalidated for this checkpoint).

## Current state

This branch is the **model12 / 2D SPH use case**, production-shaped (2026-07-21): model9 removed, whole-cloud path first-class, results test-guarded and persisted.

- **model12 (physics-informed)**: `λ3·|KG|²` (SPH kernel-gradient symmetry) makes corrected clouds valid SPH restarts — the removed model9's were not (it blows KG up ~3.9×: full-sweep disordered mean 1.278 vs raw 0.326; kept as an artifact-only comparison series). `WholeCloudCorrector2D` (`corrector: wholecloud`, `model12_wholecloud.yaml`) is the deployment path: bit-exact against the sim-validated trajectory (`tests/test_wholecloud.py`), ~20× faster than tiled (~0.26 s vs 5.8 s per N=2500 timestep). Full-trajectory KG sweep (`kg_sweep.py`, 1002 steps, disordered means): raw 0.326 / TV 0.274 / model12_wc **0.128**, KG floor ≈ 0.111. The KG term is a *soft, training-only* constraint — never enforced at inference. PoC for a physics-informed-ML ("Physics in AI") workshop paper.
- **Obstruction**: config retargeted to model12; re-run pending.
- Roadmap (ON HOLD pending user walkthrough): 3D model12 (needs 3D quintic KG in utils/metrics.py; synthetic data only for now); ablations grouped + λ3=0 control arm; obstruction re-run; disorder-level ("physical temperature") study; paper/ figure scripts (deferred until the rest is cleared).
