# Poisson Disk Corrector

Neural net that takes a 2D point cloud violating a minimum-distance constraint
`rd` and predicts per-point displacements that fix it. Two models: `model9`
(violation-weighted edge net) and `model12` (physics-informed — an SPH
kernel-gradient-symmetry loss so corrected clouds are valid SPH simulation
restarts). Applied to SPH simulation output as a Transport Velocity replacement.

This is the `sph-use-case` branch — the 2D SPH experiments only (SPH-trajectory
correction + obstruction), branched from `simplify`. The 3D MD-init experiment
(`olga_init`) lives on `simplify`; the full model history lives on `main`. The
corrector code keeps its 2D/3D shared bodies, so future 3D SPH work stays open.

---

## Quickstart

```bash
# Smoke test — ~30s on CPU, just checks the pipeline runs
.venv\Scripts\python.exe src/training/trainer.py ^
  --train-config   src/configs/training/smoke_test/train_config.yaml ^
  --dataset-config src/configs/training/smoke_test/dataset_config.yaml ^
  --loss-config    src/configs/training/smoke_test/loss_config.yaml ^
  --model-config   src/configs/training/smoke_test/model_config_9.yaml

# model9 ML corrector vs TV baseline on the SPH trajectory
.venv\Scripts\python.exe src/inference/experiments/sph_tv/sph_tv_experiment.py src/configs/experiments/sph_tv/grid_6x6.yaml --timestep 300

# model12 physics-informed (SPH kernel-gradient symmetry) corrector on the same trajectory
.venv\Scripts\python.exe src/inference/experiments/sph_tv/sph_model12_experiment.py src/configs/experiments/sph_tv/model12_grid.yaml --timestep 300
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
    training/                 one YAML set per checkpoint (see "Training")
      dataset/  loss/  model/  trainer/  smoke_test/
    experiments/              one subfolder per experiment, one YAML per variant
      sph_tv/  apply_corrector/  obstruction/
  models/
    architectures/
      model9/                 model9.py (violation-weighted edge MLP, tanh clamp; 2D/3D via input_dim) + invariance.py (center + PCA-rotate)
      model12/                model12.py — SPH corrector: L-round message passing + KG-symmetry loss; forward_sparse for whole-cloud
    weights/                  production checkpoints model9_*.pt + model12_sph_l4.pt, tracked in git (see its README.md)
  training/
    trainer.py                training loop: online data, K-step unrolling, dual eval (K=1 vs K=unroll)
    loss.py                   hybrid_loss (model9) + rdsph_loss/sph_loss (model12) + kernel_gradient/mean_kg_norm
    datagen.py                PoissonDiskDataset + PackedPoissonDiskDataset — online generation, no dataset files
  inference/
    correctors/
      base.py                 Corrector / Experiment ABCs — every corrector and experiment implements these
      common/                 pbc.py, scaling.py, tiling.py — shared PBC / rd-scaling / tile geometry
      grid/                   GridCorrector2D/3D — tiles the whole domain, ghost buffers, every pass full price
      kdtree/                 KDTreeCorrector2D/3D — runs the model only around violations, k = cap w/ early stop
      tv/                     TVCorrector2D / FastTVCorrector2D — Transport Velocity baseline
      pure/                   PureInference2D — bare model round trip, N == N_train, no tiling/PBC
    experiments/              one subfolder per experiment, each with its own README:
      sph_tv/                 model9 vs TV baseline + sph_model12 (physics-informed) over the SPH trajectory
      apply_corrector/        correct every SPH timestep, save the trajectory (+ h5part tools)
      obstruction/            corrector around domain obstacles (gear mask + ghost fill)
      pbc_toy/                synthetic proof the tiling + ghost-buffer approach is correct
  utils/
    config.py, logger.py      config loading, logging
    visualizations/
      training_visualizations/    sample plots, evolution GIF, finished-run plots
      inference_visualizations/   comparison / timeseries / enhanced / tiling figures
      misc/                       (empty placeholder)

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

## Checkpoints

Three in `src/models/weights/`, all 2D. Details in
`src/models/weights/README.md`.

| Checkpoint | Model | Model config | Used by |
|---|---|---|---|
| `src/models/weights/model9_n100_p050.pt` | model9 | `src/configs/training/model/model_config_9_n100_p050.yaml` | `src/configs/experiments/sph_tv/grid_6x6.yaml` |
| `src/models/weights/model9_n50_sparse.pt` | model9 | `src/configs/training/model/model_config_9.yaml` | `src/configs/experiments/sph_tv/grid_10x10.yaml` |
| `src/models/weights/model12_sph_l4.pt` | model12 | `src/configs/training/model/model_config_12_sph_L4.yaml` | `src/configs/experiments/sph_tv/model12_*.yaml` |

Never cross a checkpoint with another checkpoint's model config — different
`hidden_dim` / `max_displacement`, will either error or silently produce
garbage. Default 2D corrector is `model9_n100_p050`; `model12_sph_l4` is the
physics-informed option for SPH-restart quality (distinct calling convention —
`rd=attention_rd`, `box=`).

---

## Configs

Two trees under `src/configs/`: **`training/`** — four YAMLs passed as
separate flags to the trainer — and **`experiments/`** — one subfolder per
experiment, one YAML per variant. `src/configs/training/smoke_test/` holds fast
CPU variants of the training set. Running a variant of anything means copying
a YAML and editing it, not changing code.

### Dataset config (`src/configs/training/dataset/`)

Parametrises the online data generator (clouds are generated per batch,
never stored on disk). `type` selects the generator class.

#### `type: standard` — `PoissonDiskDataset` (default when `type` is omitted)

```yaml
points_per_cloud: 50    # N, points per cloud
dim: 2                  # 2 or 3
rd: 0.05                # minimum pair distance the clean clouds satisfy
seed: 42
noise_scale_min: 0.0    # per-cloud Gaussian noise σ drawn uniformly from
noise_scale_max: 0.03   # [min, max]; keep min at 0 so the model also sees
                        # clean clouds and learns to stay idle
```

#### `type: packed` — `PackedPoissonDiskDataset`

Same fields, but N is derived from a density instead of set directly:

```yaml
type: packed
packedness: 0.5         # fraction of max lattice density (triangular in 2D,
                        # FCC in 3D); N follows — 0.5 at rd=0.05 → N≈231
dim: 2
rd: 0.05
seed: 42
noise_scale_min: 0.0
noise_scale_max: 0.02
```

### Model config (`src/configs/training/model/`)

```yaml
architecture: violation_weighted_edge_network_v2   # descriptive label, not parsed
model_file: models/architectures/model9/model9  # import path of the model module

hidden_dim: 128            # MLP width (128 in every checkpoint)
edge_depth: 3              # layers in the edge MLP (default 3)
norm: layer                # 'layer' for LayerNorm; any other value = no norm
activation: GELU           # any torch.nn activation class name
max_displacement: 0.0912   # tanh output clamp; recipe 1.2·rd
```

The model's `input_dim` comes from the dataset config's `dim`, and weight
`initialization` from the trainer config — neither is set here.

### Loss config (`src/configs/training/loss/`)

```yaml
name: hybrid_loss     # model9's loss; model12 uses rdsph_loss (+ λ3·SPH
                      # kernel-gradient symmetry) or sph_loss — see
                      # src/configs/training/loss/loss_config_rdsph*.yaml

params:
  lambda1: 20         # linear violation penalty; calibrate as lambda1·rd = 1
  lambda1_quad: 0     # quadratic violation term; 0 in every production config
  lambda2: 0.0087     # displacement regulariser; recipe 0.1·lambda1/(N−1)
                      # keeps a 10× violation preference at any N
```

### Trainer config (`src/configs/training/trainer/`)

```yaml
batch_size: 8         # clouds per iteration; edge memory is O(N²) per cloud,
                      # so shrink as N grows (8 at N≈231, 32 at N=100)
num_iterations: 10000
initialization: xavier_uniform   # weight init scheme
device: cuda
unroll_steps: 3       # K-step unrolling: the model is applied K times per
                      # iteration, loss summed over all steps, backprop
                      # through everything

optimizer:
  name: Adam          # any torch.optim class
  params:             # its constructor kwargs
    lr: 0.001
    weight_decay: 1.0e-4
    betas: [0.9, 0.999]

lr_scheduler:
  name: LinearLR      # any torch.optim.lr_scheduler class
  params:             # its constructor kwargs
    start_factor: 1.0
    end_factor: 0.001
    total_iters: 10000

eval:
  validation_size: 64     # clouds in the fixed validation set
  num_visual_samples: 6   # clouds shown in the periodic sample figures
  log_interval: 100       # iterations between metric logs
  sample_interval: 500    # iterations between sample figures
```

### Experiment configs (`src/configs/experiments/<experiment>/`)

Tie model + checkpoint + corrector settings + data together. Every consumer
reads the common blocks plus its own block(s) and ignores the rest, so a file
only needs the blocks its experiment uses — each shipped YAML states its
consumer and run command in its header comment.

#### Common — required by every consumer

```yaml
model:
  checkpoint: src/models/weights/model9_n100_p050.pt
  config:     src/configs/training/model/model_config_9_n100_p050.yaml  # must match the checkpoint
  rd_train:   0.076

data:
  rd_test: 0.02   # minimum distance the test data should satisfy
```

`experiment.device` (below) is also read by every corrector's `from_yaml`
(default `cpu`).

#### GridCorrector — `tiling:` + `visualization:`

```yaml
tiling:
  n_cells:      6      # grid is n_cells^dim; pick so pts/tile ≈ N_train
  ghost_factor: 0.13   # ghost_width = ghost_factor * cell_size (fraction of a
                       # tile, not of rd_test); must give ghost_width ≥ rd_test

visualization:         # optional, 2D only
  enhanced: true       # save a 3-panel diagnostic figure per apply() call
  dir: artifacts/inference/misc/enhanced_viz
```

#### KDTreeCorrector — `kdtree:`

```yaml
kdtree:                # optional — 2D defaults shown (3D defaults: 50 / 12)
  total_core: 100      # points per model input, ≈ N_train of the checkpoint
  inner_core: 25       # most central points that receive displacements
```

#### SPHTVExperiment — trajectory paths + `experiment:` + `tv:`

Builds a grid corrector and the TV baseline from the same file, so it needs
the common + `tiling:` blocks too. It is the only consumer of the SPH
trajectory paths:

```yaml
data:             # extends the common data: block
  without_tv: artifacts/inference/experiments/sph_tv/data/positions_without.npy
  with_tv:    artifacts/inference/experiments/sph_tv/data/positions.npy

experiment:
  stride:   100           # evaluate every stride-th SPH timestep
  k_values: [1, 2, 3, 5]  # K sweep for the grid corrector
  device:   cpu

tv:                    # TVCorrector2D / FastTVCorrector2D baseline
  h_factor: 1.3        # smoothing length h = h_factor * dx, dx = domain / sqrt(N)
  nmax:     10         # TV iterations
  dt:       0.2        # relaxation factor
```

#### sph_model12 experiment — `experiment:` corrector step-builder

The model12 SPH experiment (`sph_model12_experiment.py`) selects its corrector
composition via `experiment.corrector`, reading the common + `tiling:` (and
`kdtree:` for the kdtree paths) blocks. `model:` points at `model12_sph_l4`;
`attention_rd` is read from the model config, distinct from the constraint
`rd_train`:

```yaml
experiment:
  corrector: grid_then_kdtree   # grid | kdtree | grid_then_kdtree
  k_grid:    5
  k_kdtree:  10                 # cap — stops early once clean
  stride:    200                # evaluate every stride-th SPH timestep
```

No `domain` field for the correctors — they center the input cloud on its
own centroid and infer the domain from its extent, fresh on every `apply()`
call.

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
    total_core = 100,        # sized for model9_n100_p050
    inner_core = 25)

PureInference2DConfig(
    rd_test = ...)           # the only required field — checkpoint /
                             # model_config / rd_train default to the
                             # model9_n100_p050 production paths

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

TV correctors take plain constructor args (`h_factor`, `nmax`, `dt`) — no
dataclass.

---

## Training

One trainer, four YAML flags:

```bash
.venv\Scripts\python.exe src/training/trainer.py ^
  --train-config   src/configs/training/trainer/train_config_sph_adamw.yaml ^
  --dataset-config src/configs/training/dataset/dataset_config_sph.yaml ^
  --loss-config    src/configs/training/loss/loss_config_rdsph_lam3_0p27.yaml ^
  --model-config   src/configs/training/model/model_config_12_sph_L4.yaml
```

Writes to `artifacts/training/train_run_<timestamp>/` (gitignored — copy
`model_final.pt` into `src/models/weights/` to keep it).

Each checkpoint has its config set (names relative to their
`src/configs/training/` subfolder):

| Checkpoint | dataset | loss | trainer | model |
|---|---|---|---|---|
| `model9_n50_sparse` (2D, N=50, rd 0.05) | `dataset_config_3` | `loss_config_5` | `train_config_2` | `model_config_9` |
| `model9_n100_p050` (2D, packed p=0.5) | `dataset_config_packed` | `loss_config_packed` | `train_config_packed` | `model_config_9_n100_p050` |
| `model12_sph_l4` (2D SPH, N=49 lattice) | `dataset_config_sph` | `loss_config_rdsph_lam3_0p27` | `train_config_sph_adamw` | `model_config_12_sph_L4` |

`model12_sph_l4` trains in ~10 min on GPU (10k iters); its trainer config uses
AdamW plus deterministic val-loss checkpointing (saves `model_best.pt` on every
validation improvement, alongside `model_final.pt`).

Recipe for scaling the model9 constants to any new rd/N: `lambda1 = 1/rd`,
`lambda2 = 0.1 * lambda1 / (N-1)`, `max_displacement = 1.2 * rd`,
`noise_scale_max = 1.0 * rd`. Density: `N_max = 2/(sqrt(3)*rd^2)` in 2D,
`sqrt(2)/rd^3` in 3D — pick rd so N/N_max lands where you want it.

---

## Experiments

One subfolder per experiment under `src/inference/experiments/`, each with its
own README, its configs under `src/configs/experiments/<name>/`, and its data
+ run outputs under `artifacts/inference/experiments/<name>/`.

| Experiment | What it does | Run |
|---|---|---|
| `sph_tv` (model9) | ML corrector vs TV baseline over the 2D SPH trajectory, K sweep | `python src/inference/experiments/sph_tv/sph_tv_experiment.py src/configs/experiments/sph_tv/grid_6x6.yaml` |
| `sph_tv` (model12) | physics-informed (SPH KG-symmetry) corrector quality check | `python src/inference/experiments/sph_tv/sph_model12_experiment.py src/configs/experiments/sph_tv/model12_grid.yaml` |
| `apply_corrector` | correct every SPH timestep, save the corrected trajectory | `python src/inference/experiments/apply_corrector/apply_corrector.py --k 5` |
| `obstruction` | corrector around a gear obstacle via ghost-particle fill | `python src/inference/experiments/obstruction/obstruction_experiment.py` |
| `pbc_toy` | synthetic proof of the tiling + ghost-buffer approach | `python src/inference/experiments/pbc_toy/pbc_toy.py` |

The model12 SPH experiment is the physics-informed path: model12 was trained
with an SPH kernel-gradient-symmetry loss so its corrected clouds work as SPH
simulation restarts, which model9's do not — model9 destroys kernel-gradient
symmetry (blows |KG| up 3–4.5×). It reports before/after quality per sampled
timestep for `corrector: grid | kdtree | grid_then_kdtree`;
`src/configs/experiments/sph_tv/` ships `model12_{grid,kdtree,grid_then_kdtree}.yaml`.

---

## Programmatic use

Every corrector satisfies the `Corrector` ABC
(`src/inference/correctors/base.py`): one method, `apply(points, k=1) -> points`,
same coordinate frame in and out. All correctors and their configs re-export
from the package root. For your own scripts, run from the project root with
`src/` on `sys.path` (the shipped experiment scripts do this themselves):

```python
from inference.correctors import (GridCorrector2D, GridCorrector2DConfig,
                                  KDTreeCorrector2D, KDTreeCorrector2DConfig)
import numpy as np

# grid — tiles the whole domain, every pass costs the same
cfg       = GridCorrector2DConfig.from_yaml('src/configs/experiments/sph_tv/grid_6x6.yaml')
corrector = GridCorrector2D(cfg)
pts       = np.load('artifacts/inference/experiments/sph_tv/data/positions_without.npy')[300]   # (2500, 2)
corrected = corrector.apply(pts, k=5)                                     # (2500, 2)

# kdtree — model only around violations; k is a cap, stops early once clean
kd = KDTreeCorrector2D(KDTreeCorrector2DConfig(
    checkpoint='src/models/weights/model9_n100_p050.pt',
    model_config='src/configs/training/model/model_config_9_n100_p050.yaml',
    rd_train=0.076, rd_test=0.02))
corrected = kd.apply(pts, k=10)   # (2500, 2) -> (2500, 2)
```

Grid `k` = number of passes; higher K = more correction, diminishing returns
(K=3–5 beats TV on the SPH data from t≈300 onwards). KDTree is ~30x faster
than the grid on sparse violations and reaches better quality on packed data
at ~5x the per-pass cost; best measured 2D quality is grid K=5 then kdtree
k≤10 — that composition is `corrector: grid_then_kdtree` in the sph_tv model12
experiment.

3D gotcha for the grid (for future 3D work): ghost overhead is ~2.7x the core
count, so choose `n_cells` so that **total** (core+ghost) points per tile ≈
N_train, not core alone.

Per-apply diagnostics (2D only): `visualization: {enhanced: true}` in the
YAML (or `enhanced_visualization=True` on the config) saves a 3-panel figure
per `apply()` to `artifacts/inference/misc/enhanced_viz/`.

`corrector.apply_shifted_grid()` — two K=1 passes on grids offset by half a
tile — is numerically equivalent to plain K=2. Prefer `apply(k=...)`.

---

## Artifacts

`artifacts/` is gitignored and holds everything that is not source, split
into a training side and an inference side; the inference side is divided by
experiment name:

```
artifacts/
  training/                     train_run_<timestamp>/ dirs from the trainer
  inference/
    experiments/
      sph_tv/
        data/                   positions.npy, positions_without.npy — the SPH
                                trajectories; not in this repo, obtain separately
        runs/                   exp_<timestamp>/ comparison figures + reports
      apply_corrector/runs/     corrected trajectories (.npy + .txt)
      obstruction/runs/         obstacle experiment figures
      pbc_toy/                  proof figure
    misc/
      enhanced_viz/             per-apply() diagnostic figures (when enabled)
```

Everything under `runs/` is regenerable — delete freely. The `sph_tv/data/`
folder holds externally provided inputs — keep those.
