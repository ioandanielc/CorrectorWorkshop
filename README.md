# Poisson Disk Corrector

Neural net that takes a 2D point cloud violating a minimum-distance constraint
`rd` and predicts per-point displacements that fix it. One model: `model12` —
physics-informed message passing trained with an SPH kernel-gradient-symmetry
loss (`λ3·|KG|²`), so corrected clouds are valid SPH simulation restarts.
Applied to SPH simulation output as a Transport Velocity replacement.

This is the `sph-use-case` branch — model12 / 2D SPH only (SPH-trajectory
correction + obstruction), branched from `simplify`. The model9 family
(violation-weighted edge net, its checkpoints, the center+PCA invariant frame
and its experiments) was removed from this branch and lives on
`simplify`/`main`, along with the 3D MD-init experiment (`olga_init`) and the
older model history. The corrector code keeps its 2D/3D shared bodies, so
future 3D SPH work stays open.

---

## Quickstart

```bash
# Smoke test — ~5s on CPU, just checks the training pipeline runs
.venv\Scripts\python.exe src/training/trainer.py ^
  --train-config   src/configs/training/smoke_test/train_config.yaml ^
  --dataset-config src/configs/training/smoke_test/dataset_config_sph.yaml ^
  --loss-config    src/configs/training/smoke_test/loss_config_rdsph.yaml ^
  --model-config   src/configs/training/smoke_test/model_config_12.yaml

# Whole-cloud corrector (the deployment path) on the SPH trajectory
.venv\Scripts\python.exe src/inference/experiments/sph_tv/sph_model12_experiment.py src/configs/experiments/sph_tv/model12_wholecloud.yaml --timestep 300

# Full-trajectory KG metrics over all precomputed series -> metrics.csv
.venv\Scripts\python.exe src/inference/experiments/sph_tv/kg_sweep.py

# Regression test: the corrector must reproduce the sim-validated artifact bit-exactly
.venv\Scripts\python.exe tests/test_wholecloud.py
```

Run everything from the project root — all paths in configs and code are
relative to it. Each experiment reads its input data from
`artifacts/inference/experiments/<experiment>/data/` (gitignored — see
"Artifacts" below for what goes where). Dependencies: `pip install -r
requirements.txt` into `.venv`.

---

## Structure

Everything that is code lives under `src/`; everything generated or
externally provided lives under `artifacts/` (gitignored). Inside both,
training and inference are separate trees.

```
src/
  configs/
    training/                 dataset/ loss/ model/ trainer/ + smoke_test/ (fast CPU variants)
                              — one YAML each: the model12_sph_l4 production recipe
    experiments/              one subfolder per experiment, one YAML per variant
      sph_tv/  obstruction/
  models/
    architectures/
      model12/                model12.py — L-round message passing + smooth proximity
                              attention; forward() dense (tile-sized N), forward_sparse()
                              edge-list (whole-cloud N)
    weights/                  production checkpoint model12_sph_l4.pt, tracked in git (see its README.md)
  training/
    trainer.py                training loop: online data, K-step unrolling, dual eval (K=1 vs K=unroll)
    loss.py                   rdsph_loss / sph_loss (λ3·SPH kernel-gradient symmetry);
                              KG primitive re-exported from utils/metrics.py
    datagen.py                PoissonDiskDataset — online generation, no dataset files;
                              periodic=True = the SPH lattice regime
  inference/
    correctors/
      base.py                 Corrector / Experiment ABCs — every corrector and experiment implements these
      common/                 pbc.py, scaling.py, tiling.py — shared PBC / rd-scaling / tile geometry
      grid/                   GridCorrector2D/3D — tiles the whole domain, ghost buffers (tiling comparison)
      kdtree/                 KDTreeCorrector2D/3D — model only around violations, k = cap w/ early stop
      wholecloud/             WholeCloudCorrector2D/3D — one forward_sparse call per pass,
                              no tiles/seams; THE deployment path
    experiments/              one subfolder per experiment, each with its own README:
      sph_tv/                 sph_model12_experiment (corrector variants) + kg_sweep (full-trajectory metrics)
      obstruction/            corrector around domain obstacles (gear mask + ghost fill)
  utils/
    config.py, logger.py      config loading, logging
    metrics.py                shared KG primitive (quintic kernel, torch, batched) +
                              numpy helpers: mean_kg, nn_dists, mean_nn, illegal_frac
    visualizations/
      training_visualizations/    sample plots, evolution GIF, finished-run plots
      inference_visualizations/   enhanced (3-panel per-apply diagnostics)

tests/
  test_wholecloud.py          WholeCloudCorrector2D must reproduce the sim-validated
                              whole-cloud trajectory bit-exactly (needs artifacts on disk)

artifacts/                    gitignored — every input and run output
  training/                   train_run_<timestamp>/ dirs from the trainer
  inference/
    experiments/<name>/       data/ (external inputs) + runs/ (outputs), per experiment
    misc/                     enhanced-viz diagnostics, one-off figures

requirements.txt              the package list — pip install -r requirements.txt
```

Scripts insert `src/` on `sys.path` themselves; all file paths in configs and
code are relative to the project root, so a fresh clone works as long as you
run from the root.

---

## Checkpoint

One in `src/models/weights/`. Details + provenance in
`src/models/weights/README.md`.

| Checkpoint | Model config | Used by |
|---|---|---|
| `src/models/weights/model12_sph_l4.pt` | `src/configs/training/model/model_config_12_sph_L4.yaml` | `src/configs/experiments/sph_tv/model12_*.yaml` |

Never cross a checkpoint with another model config — different `hidden_dim` /
`max_displacement`, will either error or silently produce garbage. Calling
convention: `rd=attention_rd` (0.286 — the attention radius, not the
constraint rd 0.14) and `box=` for periodic geometry; every corrector carries
the adapter, and every corrector requires a box-aware model (raises otherwise).

---

## Configs

Two trees under `src/configs/`: **`training/`** — four YAMLs passed as
separate flags to the trainer — and **`experiments/`** — one subfolder per
experiment, one YAML per variant. `src/configs/training/smoke_test/` holds fast
CPU variants of the training set. Running a variant of anything means copying
a YAML and editing it, not changing code.

### Dataset config (`src/configs/training/dataset/`)

Parametrises the online data generator (`PoissonDiskDataset` — clouds are
generated per batch, never stored on disk):

```yaml
points_per_cloud: 49    # N; a perfect square + periodic → 7x7 unit-torus lattice
dim: 2                  # 2 or 3
rd: 0.14                # minimum pair distance the clean clouds satisfy
seed: 42
periodic: true          # rd enforced under minimum-image; noise wraps mod 1.
                        # Dense regime: clean clouds are a randomly translated
                        # square lattice — the noise provides all the disorder
                        # (mirrors the SPH data regime).
noise_scale_min: 0.0    # per-cloud Gaussian σ drawn uniformly from [min, max];
noise_scale_max: 0.084  # keep min at 0 so the model also sees clean clouds
                        # and learns to stay idle. 0.084 = 0.6·rd.
```

### Model config (`src/configs/training/model/`)

```yaml
architecture: iterative_message_passing_corrector  # descriptive label, not parsed
model_file: models/architectures/model12/model12   # import path — REQUIRED

hidden_dim: 128            # MLP width
num_layers: 4              # message-passing rounds; receptive field ~ L·attention_rd
norm: layer                # 'layer' for LayerNorm; any other value = no norm
activation: GELU           # any torch.nn activation class name
max_displacement: 0.168    # tanh output clamp; recipe 1.2·rd
attention_rd: 0.286        # attention radius passed at every call (2·dx = h);
                           # part of the checkpoint's calling convention
```

The model's `input_dim` comes from the dataset config's `dim`, and weight
`initialization` from the trainer config — neither is set here.

### Loss config (`src/configs/training/loss/`)

```yaml
name: rdsph_loss      # violation + displacement reg (periodic distances)
                      # + λ3 · SPH kernel-gradient symmetry;
                      # sph_loss = the pure-symmetry variant (λ1 = 0 analogue)

params:
  lambda1: 7.14       # linear violation penalty; recipe lambda1 = 1/rd
  lambda1_quad: 0     # quadratic violation term; 0 in every production config
  lambda2: 0.0149     # displacement regulariser; recipe 0.1·lambda1/(N−1)
  lambda3: 0.27       # violation ↔ KG-symmetry trade-off dial — THE ablation
                      # axis; the swept arms (0.03/0.09/0.90) were purged — their
                      # exact YAMLs live in the kept training runs' configs/ snapshots
  h_factor: 2.0       # h = h_factor·dx, dx = box/sqrt(N) = lattice spacing
  box: 1.0            # unit torus
```

### Trainer config (`src/configs/training/trainer/`)

```yaml
batch_size: 32        # clouds per iteration; dense edges are O(N²) per cloud
num_iterations: 10000
initialization: xavier_uniform   # weight init scheme
device: cuda
unroll_steps: 3       # K-step unrolling: model applied K times per iteration,
                      # loss summed over all steps, backprop through everything

optimizer:
  name: AdamW         # any torch.optim class
  params:             # its constructor kwargs
    lr: 0.001
    weight_decay: 1.0e-4

lr_scheduler:
  name: CosineAnnealingLR   # any torch.optim.lr_scheduler class
  params:
    T_max: 10000
    eta_min: 1.0e-6

eval:
  validation_size: 256    # clouds in the fixed validation set
  num_visual_samples: 6   # clouds shown in the periodic sample figures
  log_interval: 100       # iterations between metric logs
  sample_interval: 500    # iterations between sample figures + val-loss
                          # checkpointing (model_best.pt on improvement)
```

### Experiment configs (`src/configs/experiments/<experiment>/`)

Tie model + checkpoint + corrector settings + data together. Every consumer
reads the common blocks plus its own block(s) and ignores the rest, so a file
only needs the blocks its experiment uses — each shipped YAML states its
consumer and run command in its header comment.

#### Common — required by every consumer

```yaml
model:
  checkpoint: src/models/weights/model12_sph_l4.pt
  config:     src/configs/training/model/model_config_12_sph_L4.yaml  # must match the checkpoint
  rd_train:   0.14   # constraint rd (drives coordinate scaling; attention_rd
                     # is read from the model config automatically)

data:
  rd_test: 0.02   # minimum distance the test data should satisfy
```

`experiment.device` (below) is also read by every corrector's `from_yaml`
(default `cpu`).

#### GridCorrector — `tiling:` + `visualization:`

```yaml
tiling:
  n_cells:      7      # grid is n_cells^dim; for a KG (symmetry) model size
                       # tiles from the SPH kernel support (3h), not rd —
                       # see model12_grid.yaml's header for the sizing argument
  ghost_factor: 0.45   # ghost_width = ghost_factor * cell_size (fraction of a
                       # tile, not of rd_test); must give ghost_width ≥ rd_test

visualization:         # optional, 2D only
  enhanced: true       # save a 3-panel diagnostic figure per apply() call
  dir: artifacts/inference/misc/enhanced_viz
```

#### KDTreeCorrector — `kdtree:`

```yaml
kdtree:                # optional — 2D defaults shown (3D defaults: 50 / 12)
  total_core: 100      # points per model input
  inner_core: 25       # most central points that receive displacements
```

#### WholeCloudCorrector — `data.box`

```yaml
data:                  # extends the common data: block
  box: 1.0             # the TRUE periodic box — give it when known; omitting it
                       # falls back to extent-based domain inference, which
                       # undershoots on lattice-like clouds (see CLAUDE.md)
```

#### sph_model12 experiment — `experiment:` corrector step-builder

`sph_model12_experiment.py` selects its corrector composition via
`experiment.corrector`, reading the common blocks plus `tiling:` / `kdtree:` /
`data.box` for the paths that need them. It is the consumer of the SPH
trajectory path:

```yaml
data:
  without_tv: artifacts/inference/experiments/sph_tv/data/positions_without.npy

experiment:
  corrector: wholecloud     # grid | kdtree | grid_then_kdtree | wholecloud
  k_wholecloud: 5           # passes for the wholecloud path
  k_grid:    5              # passes for the grid path(s)
  k_kdtree:  10             # cap — stops early once clean
  stride:    200            # evaluate every stride-th SPH timestep
```

### Corrector config dataclasses

Each corrector owns a config dataclass in its own module; build it with
`from_yaml()` from an experiment YAML, or construct it directly (see
Programmatic use). Every ML corrector config carries `checkpoint` /
`model_config` / `rd_train` / `rd_test` (required) and `device` (`'cpu'`).
Beyond those, with dataclass defaults shown:

```python
GridCorrector2DConfig(       # GridCorrector3DConfig: same fields
    n_cells      = 6,
    ghost_factor = 1.0,      # dataclass default — the shipped YAMLs use tuned values
    enhanced_visualization = False,
    viz_dir      = 'artifacts/inference/misc/enhanced_viz')

KDTreeCorrector2DConfig(
    total_core = 100,
    inner_core = 25)

WholeCloudCorrector2DConfig(
    box = None)              # None = infer domain from extent; set the true
                             # PBC box when known (the SPH case: 1.0)

ObstructionExperimentConfig( # standalone experiment config, not a corrector's
    corrector_config = 'src/configs/experiments/obstruction/grid_6x6.yaml',
    rd          = 0.012,
    domain      = 1.0,
    cx          = 0.5,       # obstacle centre
    cy          = 0.5,
    noise_scale = 0.3,       # noise_std = noise_scale * rd
    k_values    = [1, 3, 5],
    seed        = 42,
    device      = 'cuda')
```

---

## Training

One trainer, four YAML flags. The `model12_sph_l4` recipe (~10 min on GPU,
10k iters):

```bash
.venv\Scripts\python.exe src/training/trainer.py ^
  --train-config   src/configs/training/trainer/train_config_sph_adamw.yaml ^
  --dataset-config src/configs/training/dataset/dataset_config_sph.yaml ^
  --loss-config    src/configs/training/loss/loss_config_rdsph_lam3_0p27.yaml ^
  --model-config   src/configs/training/model/model_config_12_sph_L4.yaml
```

Writes to `artifacts/training/train_run_<timestamp>/` (gitignored) —
`model_best.pt` (best val loss — what shipped as `model12_sph_l4.pt`) and
`model_final.pt` (last iterate); copy the one you want into
`src/models/weights/` to keep it.

Ablation axes already explored (winning arm shipped, losing arms purged from
src/): λ3 ∈ {0.03, 0.09, 0.27, 0.90} — 0.27 won; L ∈ {3, 4} — L4 won; Adam vs
AdamW — AdamW won. Every arm's exact config set survives in its training run's
`configs/` snapshot under `artifacts/training/train_run_2026-07-15_*` (kept),
and on `main`.

Recipe for scaling the constants to any new rd/N: `lambda1 = 1/rd`,
`lambda2 = 0.1 * lambda1 / (N-1)`, `max_displacement = 1.2 * rd`,
`noise_scale_max = 0.6 * rd`, `h_factor = 2.0` (h = 2·dx).

---

## Experiments

One subfolder per experiment under `src/inference/experiments/`, each with its
own README, its configs under `src/configs/experiments/<name>/`, and its data
+ run outputs under `artifacts/inference/experiments/<name>/`.

| Experiment | What it does | Run |
|---|---|---|
| `sph_tv` (corrector variants) | model12 corrector quality per sampled timestep | `python src/inference/experiments/sph_tv/sph_model12_experiment.py src/configs/experiments/sph_tv/model12_wholecloud.yaml` |
| `sph_tv` (KG sweep) | full-trajectory mean\|KG\|/nn/ill% for raw / TV / model9-K5 / wholecloud → metrics.csv | `python src/inference/experiments/sph_tv/kg_sweep.py` |
| `obstruction` | corrector around a gear obstacle via ghost-particle fill (retargeted to model12 — re-run pending) | `python src/inference/experiments/obstruction/obstruction_experiment.py` |

Headline result (full 1002-step sweep, disordered regime t ≥ 300, mean|KG| —
lower = better SPH restart): raw 0.326, TV 0.274, model9-K5 1.278 (the
motivating failure — kept as an artifact, regenerable only on `main`),
**model12 whole-cloud 0.128** (KG floor ≈ 0.111). Validated end-to-end by an
actual SPH re-simulation from the whole-cloud corrected start states.

---

## Programmatic use

Every corrector satisfies the `Corrector` ABC
(`src/inference/correctors/base.py`): one method, `apply(points, k=1) -> points`,
same coordinate frame in and out. All correctors and their configs re-export
from the package root. For your own scripts, run from the project root with
`src/` on `sys.path` (the shipped experiment scripts do this themselves):

```python
from inference.correctors import (WholeCloudCorrector2D, WholeCloudCorrector2DConfig,
                                  GridCorrector2D, GridCorrector2DConfig)
import numpy as np

pts = np.load('artifacts/inference/experiments/sph_tv/data/positions_without.npy')[300]  # (2500, 2)

# wholecloud — the deployment path: one forward_sparse call per pass, ~0.26s/timestep
wc = WholeCloudCorrector2D(WholeCloudCorrector2DConfig.from_yaml(
    'src/configs/experiments/sph_tv/model12_wholecloud.yaml'))
corrected = wc.apply(pts.astype(np.float32), k=5)                # (2500, 2)

# grid — tiled comparison path (~5.8s/timestep at N=2500)
gc = GridCorrector2D(GridCorrector2DConfig.from_yaml(
    'src/configs/experiments/sph_tv/model12_grid.yaml'))
corrected = gc.apply(pts.astype(np.float32), k=5)
```

`k` = number of passes; higher K = more correction, diminishing returns.
KDTree's `k` is a cap — it stops early once clean. Grid/kdtree remain as the
tiling comparison; wholecloud wins on both KG and speed (no tile seams).

3D gotcha for the grid (for future 3D work): ghost overhead is ~2.7x the core
count, so choose `n_cells` so that **total** (core+ghost) points per tile ≈
N_train, not core alone.

Per-apply diagnostics (2D only): `visualization: {enhanced: true}` in the
YAML (or `enhanced_visualization=True` on the config) saves a 3-panel figure
per `apply()` to `artifacts/inference/misc/enhanced_viz/`.

---

## Artifacts

`artifacts/` is gitignored and holds everything that is not source, split
into a training side and an inference side; the inference side is divided by
experiment name:

```
artifacts/
  training/                     train_run_<timestamp>/ dirs from the trainer;
                                train_run_2026-07-15_11-18-44 = model12_sph_l4
                                provenance, the other 2026-07-15 dirs = its
                                λ3/L/optimizer sweep arms — keep these
  inference/
    experiments/
      sph_tv/
        data/                   positions.npy (TV), positions_without.npy (raw) —
                                the SPH trajectories; not in the repo, obtain separately
        for_sim/                sim-validated start states + the whole-cloud corrected
                                trajectory — KEEP: tests/ and kg_sweep read these
        runs/                   experiment outputs (kg_sweep_*/, model12_*/)
      apply_corrector/runs/     positions_corrected_K5.npy — the model9-era corrected
                                trajectory, kept as the KG sweep's comparison series
      obstruction/runs/         obstacle experiment figures
    misc/
      enhanced_viz/             per-apply() diagnostic figures (when enabled)
```

Run outputs are regenerable — delete freely, EXCEPT `sph_tv/for_sim/` (test +
sweep dependencies), `apply_corrector/runs/positions_corrected_K5.npy`
(regenerable only on `main`), and the provenance training runs above. The
`sph_tv/data/` folder holds externally provided inputs — keep those.
