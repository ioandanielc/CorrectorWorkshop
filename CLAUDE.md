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
│   │   ├── training/            # dataset/ loss/ model/ trainer/ YAML sets + smoke_test/ (fast CPU variants); one YAML each = the production recipe
│   │   └── experiments/         # one subfolder per experiment, one YAML per variant
│   │       ├── sph_tv/          #   model12_wholecloud.yaml
│   │       └── obstruction/     #   wholecloud.yaml (corrector blocks only)
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
│   │   │   ├── common/          # scaling.py (rd_train/rd_test coordinate scaling)
│   │   │   └── wholecloud/wholecloud_corrector.py  # WholeCloudCorrector2D/3D — one forward_sparse call per pass, no tiles/seams; THE deployment path
│   │   └── experiments/         # one subfolder per experiment, each with a small README
│   │       ├── sph_tv/          # sph_model12_experiment (wholecloud) + kg_sweep.py (full-trajectory KG/nn metrics -> metrics.csv)
│   │       └── obstruction/     # ObstructionExperiment (wholecloud + ghost fill) + obstruction.py (masks + fill)
│   └── utils/
│       ├── config.py, logger.py # config loading, logging
│       ├── metrics.py           # shared KG primitive (quintic kernel, torch, batched) + numpy helpers (mean_kg, nn_dists, mean_nn, illegal_frac)
│       └── visualizations/
│           └── training_visualizations/   # sample plots, evolution GIF, finished-run plots
├── tests/                       # test_wholecloud.py — WholeCloudCorrector2D must reproduce the sim-validated artifact bit-exactly (needs artifacts on disk)
├── artifacts/                   # gitignored — every input and run output
│   ├── training/                # train_run_<timestamp>/ from the trainer
│   └── inference/
│       ├── experiments/<name>/  # data/ (external inputs) + runs/ (outputs), per experiment
│       └── misc/                # one-off figures
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

Full run (the `model12_sph_l4` recipe — ~10 min on GPU): `train_config_sph_adamw` + `dataset_config_sph` + `loss_config_rdsph_lam3_0p27` + `model_config_12_sph_L4` — the only YAML in each folder. The swept λ3/L/optimizer arms were purged from src/; each arm's exact config set survives in its training run's `configs/` snapshot under `artifacts/training/train_run_2026-07-15_*` and on `main`. Writes to `artifacts/training/train_run_<timestamp>/` (gitignored — copy the checkpoint into `src/models/weights/` to keep it; `model_best.pt` = best val loss, `model_final.pt` = last iterate).

---

## Production checkpoint

One deployable checkpoint in `src/models/weights/` (details in `src/models/weights/README.md`):

| Checkpoint | N_train | rd_train | Used by |
|---|---|---|---|
| **model12_sph_l4** | 49 (7×7 lattice) | 0.14 (rd); cutoff_rd 0.286 | `src/configs/experiments/sph_tv/model12_*.yaml` — physics-informed (SPH KG symmetry) |

Don't cross checkpoint and model config — `hidden_dim` / `max_displacement` differ, will either error or silently produce garbage. Calling convention: `rd=cutoff_rd` (not the constraint rd), `box=` for periodic geometry; all correctors carry the adapter.

---

## Key design decisions

**Training:**
- Data generated **online** per batch — no dataset file on disk. The SPH regime (`periodic: true`) enforces rd under minimum-image on the unit torus; at that packing, clean clouds are a randomly translated square lattice and the noise provides the disorder.
- No invariant-frame transform: model12 is translation-invariant by construction (it only sees relative positions, `rel = x_j − x_i`) and trains in the fixed unit-torus frame. It is NOT rotation-equivariant — deliberate for a fixed simulation frame (the model9-era center+PCA machinery lives on `main`/`simplify`).
- `lambda2 = lambda1 / (N−1) / 10` keeps displacement reg a constant fraction of violation penalty regardless of N; `lambda3` is the violation↔KG-symmetry trade-off dial (swept arm configs purged — provenance in the kept run snapshots under `artifacts/training/`).
- Training uses K-step unrolling (backprop through all K steps). Eval reports both K=1 and K=unroll_steps.

**Inference (all correctors):**

0. **Domain**: give the true PBC box explicitly via `data.box` when known (the SPH case: 1.0; obstruction: its domain) — extent-based inference (`box: null`) undershoots on lattice-like clouds (see rough edges).

1. **Whole-cloud sparse pass**: per pass, scale the cloud to the model's training rd, build the PBC edge list (`cKDTree.query_pairs` at `cutoff_rd`, both directions), one `forward_sparse` call, unscale, wrap. No tiles, no ghosts, no seams; measured 0.189 s/timestep on CPU and 0.049 s on CUDA at N=2500, k=5. (The tiled grid/kdtree correctors were removed 2026-07-21 — on `main` if a tiled comparison is ever needed again; their measured numbers are recorded in the kg_sweep artifacts and memory.)

2. **Coordinate scaling**: `scale = rd_train / rd_test`. Multiply coords by `scale` before the model call; divide displacements by `scale` after. Maps violations into the model's training distribution. Verified to extrapolate: obstruction runs at rd_test=0.012 (scale ≈ 11.7) on a bounded scene and converges to ~0.98·rd.

3. **Ghost-particle obstacles** (obstruction experiment): the obstacle interior is filled with fixed ghost particles at spacing rd (`obstruction.py`); each pass concatenates ghosts to the real particles, applies the whole-cloud corrector once, and keeps only the real rows — ghosts are re-pinned every pass so the solid never drifts.

4. **Iterative application**: `apply(pts, k=...)` = k passes, diminishing returns; k=5 is the validated deployment setting.

**Interfaces (`src/inference/correctors/base.py`):**
- `Corrector` — one method `apply(points, k=1) -> points`. Implemented by `WholeCloudCorrector2D/3D` (thin `DIM = 2/3` subclasses of a private ND body). Requires a **box-aware** model with `forward_sparse` (`uses_box`, model12-style) and raises otherwise; `model_file` is required in the model config (no default architecture). The config carries `checkpoint` + `model_config` + `rd_train`; everything re-exports from `inference.correctors`. (TV / grid / kdtree implementations were purged — TV comparisons use the precomputed `positions.npy` trajectory; tiled correctors live on `main`.)
- `WholeCloudCorrector2D/3D` — one `forward_sparse` call over the entire cloud per pass (PBC cKDTree edge list at `cutoff_rd`). Takes an optional explicit `data.box` — give the true PBC box when known (the SPH case does), which sidesteps the domain-undershoot issue below. Guarded by `tests/test_wholecloud.py`: must reproduce the sim-validated whole-cloud artifact bit-exactly.
- `Experiment` — one method `run()`. Implemented by `ObstructionExperiment`; the sph_tv model12 script uses the same `experiment.corrector` step-builder pattern. Experiment-loop settings (`stride`, `k_*`, `corrector` kind…) live on the experiments, not on corrector configs.

**Obstructions**: `src/inference/experiments/obstruction/obstruction.py` fills obstacle interiors (ellipse/polygon/gear masks) with ghost particles at spacing `rd` so the corrector pushes real particles away from the boundary; ghosts are re-pinned every pass and dropped from the output.

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
- Open issue (only when `data.box` is omitted): extent-based domain inference undershoots the true PBC box on lattice-like inputs, pinching the wrap seam into artificial violations. With the box given explicitly (all shipped configs do) the issue is moot — at t=0 the wholecloud corrector leaves the cloud essentially untouched (nn 0.0200 → 0.0200).
- `obstruction/grid_6x6.yaml` was retargeted from the removed model9 checkpoint to model12 — not yet re-run/tuned (tiling + ghost_factor unvalidated for this checkpoint).

## Current state

This branch is the **model12 / 2D SPH use case**, production-shaped (2026-07-21): model9 removed, whole-cloud path first-class, results test-guarded and persisted.

- **model12 (physics-informed)**: `λ3·|KG|²` (SPH kernel-gradient symmetry) makes corrected clouds valid SPH restarts — the removed model9's were not (it blows KG up ~3.9×: full-sweep disordered mean 1.278 vs raw 0.326; kept as an artifact-only comparison series). `WholeCloudCorrector2D` (`corrector: wholecloud`, `model12_wholecloud.yaml`) is the deployment path: bit-exact against the sim-validated trajectory (`tests/test_wholecloud.py`). Full-trajectory KG sweep (`kg_sweep.py`, 1002 steps, disordered means): raw 0.326 / TV 0.274 / model12_wc **0.128** at k=5. The KG term is a *soft, training-only* constraint — never enforced at inference. PoC for a physics-informed-ML ("Physics in AI") workshop paper.
- **Obstruction**: re-run 2026-07-21 with the wholecloud corrector — 6100 real + 832 ghosts, mean nn 0.0076 → 0.0117 (rd 0.012, scale ≈ 11.7, bounded non-periodic scene) in 5 ghost-re-pinned passes. Strong scale-extrapolation evidence.

### 2026-08-10 — ablation suite for the workshop paper (`paper/`)

**Read `paper/RESULTS.md` first** for what is established, `paper/JOURNAL.md` for the run
log and the claim-by-claim audit. Four claims that used to live in this file were retracted
by measurement that day; do not re-import them from memory or older docs:

| retracted | replaced by |
|---|---|
| "KG floor ≈ 0.111" | an artifact of k=5. |KG| falls monotonically to 0.0675 by k=40 with no floor; k=1 is *worse than no correction*; illegal% bottoms at k=8 |
| "~0.26 s / ~20× faster than tiled" | measured deployment cost, N=2500, k=5: **0.189 s/step CPU, 0.049 s/step CUDA** |
| "model12 is 3× cheaper than GNS" | that is dense *training* at N=49; at deployment it is 1.13× CPU / 1.39× CUDA |
| "per-particle normalisation drives size transfer" | the `nonorm` rung removes it and transfers *better*; the fixed kernel is the candidate mechanism, and is marked suggestive rather than established |

Headline additions: the λ3=0 ablation **reproduces model9's failure on the real trajectory**
(|KG| 1.471 vs raw 0.331) and collapses in 4/4 seeds; and the small-N synthetic benchmark
**misranks architectures against deployment** — GNS and the `maxagg` bridge rung both beat
model12 at N=49 and both lose at N=2500.

New in the tree: `paper/` (results, journal, architecture report, exhibits),
`src/configs/training/ablations/` (architecture, bridge, cardinality, packing, cutoff),
`src/inference/experiments/ablations/` (`score_arm.py`, `build_exhibits.py`),
four baseline architectures (`pointnet`, `pointnet2`, `dgcnn`, `gns`) plus
`model12_ablate` (component flags; its baseline rung is bit-identical to `model12`),
and `tests/test_sparse_paths.py`.

**Traps this branch now guards against**: the trainer has a degenerate mode — roughly half
of some architecture variants collapse into a *saturated uniform translation*, which every
loss term is blind to (they all depend only on relative positions) and which |KG| and
illegal% both report as "unchanged". `trainer.py` logs `bulk_drift` and warns above 50%.
The trainer is now seeded (`seed: 0`) because that mode is initialisation-dependent, so
unseeded runs were coin flips. **A single collapsed run means nothing — collapse claims
need ≥3 seeds.**

- Roadmap: 3D model12 (needs 3D quintic KG in utils/metrics.py; synthetic data only for now); disorder-level ("physical temperature") study; re-simulating at k>5 to validate the better operating point (external solver); paper prose.
