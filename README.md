# Poisson Disk Corrector

Neural net that takes a 2D point cloud violating a minimum-distance constraint
`rd` and predicts per-point displacements that fix it. One model: `model12` —
physics-informed message passing trained with an SPH kernel-gradient-symmetry
loss (`λ3·|KG|²`), so corrected clouds are valid SPH simulation restarts.
Applied to SPH simulation output as a Transport Velocity replacement.

This is the `sph-use-case` branch — model12 / 2D SPH only. Everything not on
the deployment path was removed (2026-07-21) and lives on `simplify`/`main`:
the model9 family and its invariant frame, the tiled grid/kdtree correctors,
the TV corrector reimplementation, the 3D MD-init experiment, the older model
history. The remaining code is one linear pipeline:

```
datagen ──> model12 ──> loss ──> trainer ──> model12_sph_l4.pt
                                                  │
                                                  ▼
                                       WholeCloudCorrector2D/3D
                                          │              │
                                          ▼              ▼
                                       sph_tv        obstruction
                                  (quality + kg_sweep)  (ghost fill)
```

---

## Quickstart

```bash
# Smoke test — ~5s on CPU, just checks the training pipeline runs
.venv\Scripts\python.exe src/training/trainer.py ^
  --train-config   src/configs/training/smoke_test/train_config.yaml ^
  --dataset-config src/configs/training/smoke_test/dataset_config_sph.yaml ^
  --loss-config    src/configs/training/smoke_test/loss_config_rdsph.yaml ^
  --model-config   src/configs/training/smoke_test/model_config_12.yaml

# Whole-cloud corrector on the SPH trajectory
.venv\Scripts\python.exe src/inference/experiments/sph_tv/sph_model12_experiment.py src/configs/experiments/sph_tv/model12_wholecloud.yaml --timestep 300

# Full-trajectory KG metrics over all precomputed series -> metrics.csv
.venv\Scripts\python.exe src/inference/experiments/sph_tv/kg_sweep.py

# Obstacle initialization demo (gear + ghost fill)
.venv\Scripts\python.exe src/inference/experiments/obstruction/obstruction_experiment.py

# Regression test: the corrector must reproduce the sim-validated artifact bit-exactly
.venv\Scripts\python.exe tests/test_wholecloud.py
```

Run everything from the project root — all paths in configs and code are
relative to it; scripts put `src/` on `sys.path` themselves. Dependencies:
`pip install -r requirements.txt` into `.venv`.

---

## Every script, and why it exists

### The model — `src/models/architectures/`

| File | What | Why we need it |
|---|---|---|
| `model12/model12.py` | L rounds of message passing with a fixed SPH-kernel-shaped proximity weight `(1−(d/rd)²)²` over pairs within `attention_rd`; minimum-image geometry via `box=`. Two entry points: `forward()` (dense `(B,N,N)` edges, training-scale N) and `forward_sparse()` (edge list, whole-cloud N). | The corrector itself. Message passing sees *non-violating* pairwise asymmetry — the part of the KG signal a violation-gated net is blind to. `forward_sparse` removes the dense-edge memory ceiling, enabling one call over an entire N=2500 cloud (the deployment path). Translation-invariant by construction (only `rel = x_j−x_i` enters) and permutation-equivariant; NOT rotation-equivariant — deliberate for a fixed simulation frame. |

### Training — `src/training/`

| File | What | Why we need it |
|---|---|---|
| `datagen.py` | `PoissonDiskDataset` — online per-batch cloud generation, no dataset files. `periodic: true` enforces rd under minimum-image on the unit torus; in the dense regime clean clouds are a randomly translated square lattice + capped jitter, and the noise step provides all disorder. | Training data must mirror the SPH regime: near-max packing on a torus, disorder from perturbation. Online generation means every batch is fresh and there is nothing to version on disk. |
| `loss.py` | `rdsph_loss` = λ1·violations (periodic distances) + λ2·displacement reg + **λ3·\|KG\|²** (quintic kernel-gradient symmetry). `sph_loss` = the pure-symmetry variant. | The physics-informed objective — the core contribution. λ3 is the violation↔symmetry trade-off dial (0.27 shipped). `sph_loss` is kept as the ablation arm showing the original corrector idea is load-bearing: symmetry alone doesn't work, the illegality + displacement terms are necessary. |
| `trainer.py` | Training loop driven by 4 YAML flags: online batches, K-step unrolling (backprop through all K applications), dual eval (K=1 deploy vs K=unroll), deterministic val-loss checkpointing (`model_best.pt`), loss.csv + sample figures + evolution GIF per run. | The model is *applied iteratively* at inference, so it must be *trained* through iterated application — single-step training does not produce a good k=5 corrector. Val-loss checkpointing is why the shipped weights are `model_best.pt`, not the last iterate. |

### Shared utilities — `src/utils/`

| File | What | Why we need it |
|---|---|---|
| `metrics.py` | The KG primitive (2D quintic spline, torch, batched: `kernel_gradient`, `kg_norm`, `mean_kg_norm`) + numpy helpers (`mean_kg`, `nn_dists`, `mean_nn`, `illegal_frac`). | Single source of truth for the physics: the *same* kernel-gradient formula is the training objective (via loss.py) and the evaluation metric (via kg_sweep / experiments). That identity is the physics-informed claim — one file makes it checkable. |
| `config.py` | YAML loaders for the four training config kinds. | One place that defines what a config file is. |
| `logger.py` | Run-dir creation + logger with ETA. | Every training run gets a self-contained `train_run_<timestamp>/` with a configs snapshot — this is how sweep provenance survived the purges. |
| `visualizations/training_visualizations/visualizations.py` | Sample-cloud comparison plots + evolution GIF, called by the trainer. | Visual sanity during training: the corrector's behaviour on fixed validation clouds, every `sample_interval` iterations. |

### Inference — `src/inference/correctors/`

| File | What | Why we need it |
|---|---|---|
| `base.py` | `Corrector` ABC (`apply(points, k=1) -> points`) + `Experiment` ABC (`run()`). | The seam between model machinery and experiments: experiments consume the interface, never the model directly. This is what made three rounds of corrector swaps possible without touching experiment logic. |
| `common/scaling.py` | `compute_scale(rd_train, rd_test)` — multiply coords by `rd_train/rd_test` before the model, divide displacements after. | The model trains at one rd (0.14) and deploys at others (SPH 0.02 → scale 7.0; obstruction 0.012 → scale 11.7). Scaling maps every deployment into the training distribution — verified to extrapolate. |
| `wholecloud/wholecloud_corrector.py` | `WholeCloudCorrector2D/3D` + config dataclass. Per pass: scale → PBC `cKDTree.query_pairs(attention_rd)` edge list → one `forward_sparse` call → unscale → wrap. Optional explicit `data.box` (give the true PBC box when known — extent inference undershoots on lattice-like clouds). Requires a box-aware model with `forward_sparse`, raises otherwise. | THE deployment path: no tiles, no ghosts, no seams, ~0.26 s per N=2500 timestep (~20× faster than the removed tiled path) and equal-or-better KG everywhere. Dimension-generic body — `WholeCloudCorrector3D` is `DIM = 3`, so future 3D work costs nothing here. |

### Experiments — `src/inference/experiments/`

| File | What | Why we need it |
|---|---|---|
| `sph_tv/sph_model12_experiment.py` | Applies the whole-cloud corrector to sampled timesteps of the real SPH trajectory; reports mean nn + illegal% before/after; timeseries figure + report per run. | The real-data check: model12 trains only on synthetic 49-point lattices — this proves it transfers to N=2500 disordered SPH frames. |
| `sph_tv/kg_sweep.py` | Full-trajectory (1002-step) mean\|KG\|/nn/ill% for the four stored series: raw (non-TV), TV baseline, model9-K5 (motivating failure), model12 whole-cloud. Pure measurement — no corrector runs. Incremental CSV (kill-safe), GPU-batched KG (~25 s total). | The paper's evidence base as a regenerable artifact instead of scrolled-away stdout. Headline: disordered-regime KG raw 0.326 / TV 0.274 / model9 1.278 / **model12 0.128**, floor ≈ 0.111. Also why nn alone is never trusted: nn cannot discriminate methods that KG separates cleanly. |
| `obstruction/obstruction.py` | Scenario creation: `is_inside` masks (ellipse/circle/gear/polygon) + `fill_obstruction` — ghost particles at spacing rd filling the obstacle interior, eroded by rd so real particles can sit on the contour. | The obstacle-handling idea itself: the corrector never learns about walls — it sees a uniform-looking environment where some particles (ghosts) happen to be immovable, and pushes real particles into conformance with any shape. |
| `obstruction/obstruction_experiment.py` | Noisy-grid initialization outside a gear obstacle; k whole-cloud passes with ghosts re-pinned each pass; initial-vs-corrected figure + report. | Second use case + scale-extrapolation evidence: bounded non-periodic scene at rd=0.012 (scale ≈ 11.7), mean nn 0.0076 → 0.0117 in 5 passes, particles conforming to the tooth contour. |

### Tests — `tests/`

| File | What | Why we need it |
|---|---|---|
| `test_wholecloud.py` | `WholeCloudCorrector2D` must reproduce the sim-validated whole-cloud trajectory (`positions_model12_corrected.npy`) **bit-exactly** at t=0/300/600/1000 (`np.array_equal`, not allclose). Needs the gitignored artifacts on disk. | The central artifact was produced by a scratchpad script; this test pins the promoted class to it. If any refactor changes the corrector's output by one ulp, this fails — it has already guarded three rounds of surgery. |

---

## Checkpoint

One in `src/models/weights/`. Details + provenance in
`src/models/weights/README.md`.

| Checkpoint | Model config | N_train | rd_train | attention_rd |
|---|---|---|---|---|
| `model12_sph_l4.pt` | `src/configs/training/model/model_config_12_sph_L4.yaml` | 49 (7×7 lattice) | 0.14 | 0.286 |

Never cross a checkpoint with another model config — different `hidden_dim` /
`max_displacement`, will either error or silently produce garbage. Calling
convention: `rd=attention_rd` (the attention radius, not the constraint rd)
and `box=`; the corrector carries the adapter and requires a box-aware model.
Shipped weights = the run's `model_best.pt` (best val loss), sha-verified.

---

## Configs

Two trees under `src/configs/`: **`training/`** — four YAMLs passed as
separate flags to the trainer, exactly one per folder = the `model12_sph_l4`
production recipe (+ `smoke_test/` fast CPU variants) — and **`experiments/`**
— one YAML per experiment. Running a variant means copying a YAML and editing
it, not changing code.

### Training set (`src/configs/training/`), condensed

```yaml
# dataset/dataset_config_sph.yaml — the online generator
points_per_cloud: 49    # N; perfect square + periodic → 7x7 unit-torus lattice
dim: 2
rd: 0.14                # minimum pair distance clean clouds satisfy
periodic: true          # rd under minimum-image; noise wraps mod 1
noise_scale_min: 0.0    # per-cloud σ ~ U[min,max]; min 0 → model sees clean
noise_scale_max: 0.084  # clouds and learns to stay idle; 0.084 = 0.6·rd

# loss/loss_config_rdsph_lam3_0p27.yaml — the PIML objective
name: rdsph_loss
params:
  lambda1: 7.14         # 1/rd — violation penalty
  lambda2: 0.0149       # 0.1·lambda1/(N−1) — displacement reg
  lambda3: 0.27         # KG-symmetry weight — THE ablation dial (0.27 won the
                        # sweep; other arms live in the run snapshots + main)
  h_factor: 2.0         # h = 2·dx (dx = lattice spacing); box: 1.0

# model/model_config_12_sph_L4.yaml — architecture (input_dim comes from dataset dim)
model_file: models/architectures/model12/model12   # import path — REQUIRED
hidden_dim: 128
num_layers: 4           # receptive field ~ L·attention_rd
max_displacement: 0.168 # tanh clamp; 1.2·rd
attention_rd: 0.286     # 2·dx — part of the checkpoint's calling convention

# trainer/train_config_sph_adamw.yaml — optimization
batch_size: 32; num_iterations: 10000; unroll_steps: 3   # K-step unrolling
optimizer: AdamW        # decoupled decay fixed a late-training regression
lr_scheduler: CosineAnnealingLR
eval: {validation_size: 256, sample_interval: 500}       # val-loss checkpointing
```

Scaling recipe for a new rd/N: `lambda1 = 1/rd`, `lambda2 = 0.1·lambda1/(N−1)`,
`max_displacement = 1.2·rd`, `noise_scale_max = 0.6·rd`, `h_factor = 2.0`.

### Experiment configs (`src/configs/experiments/`)

Corrector blocks read by `from_yaml()`; each YAML's header states its consumer
and run command.

```yaml
model:
  checkpoint: src/models/weights/model12_sph_l4.pt
  config:     src/configs/training/model/model_config_12_sph_L4.yaml  # must match
  rd_train:   0.14      # constraint rd — drives coordinate scaling

data:
  rd_test: 0.02         # minimum distance the test data should satisfy
  box:     1.0          # TRUE periodic box — give it when known; omitting falls
                        # back to extent inference (undershoots on lattices)

experiment:             # experiment-loop settings, not corrector settings
  k_wholecloud: 5       # correction passes (k=5 = validated deployment)
  stride: 200           # evaluate every stride-th timestep
  device: cpu
```

`sph_tv/model12_wholecloud.yaml` adds `data.without_tv` (the trajectory path);
`obstruction/wholecloud.yaml` is corrector blocks only (the experiment
overrides rd_test/box/device at runtime).

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

Writes `artifacts/training/train_run_<timestamp>/` (gitignored): configs
snapshot, loss.csv, samples, `model_best.pt` (what ships) + `model_final.pt`.
Copy the checkpoint into `src/models/weights/` to keep it.

Ablation axes already explored (winners shipped, losing arms purged from src/):
λ3 ∈ {0.03, 0.09, 0.27, 0.90} → 0.27; L ∈ {3, 4} → 4; Adam vs AdamW → AdamW.
Every arm's exact config set survives in its run's `configs/` snapshot under
`artifacts/training/train_run_2026-07-15_*` (kept) and on `main`.

---

## Results

Headline (kg_sweep, full 1002-step trajectory, disordered regime t ≥ 300,
mean|KG| — lower = more SPH-consistent = better restart):

| Series | mean\|KG\| | vs raw |
|---|---|---|
| raw (non-TV input) | 0.326 | — |
| TV baseline (in-simulator) | 0.274 | −16% |
| model9-K5 (motivating failure, artifact-only) | 1.278 | **+292%** |
| **model12 whole-cloud, k=5** | **0.128** | **−61%** |

KG floor ≈ 0.111: the corrector helps wherever raw KG exceeds it (t ≳ 250) and
should not be applied to already-ordered frames. Validated end-to-end by an
actual SPH re-simulation from the whole-cloud corrected start states.
Obstruction: mean nn 0.0076 → 0.0117 at rd 0.012 on a bounded scene (scale ≈
11.7 — the coordinate-scaling extrapolation result).

Note on metrics: nn cannot discriminate methods that KG separates — judge
SPH-restart quality on KG, and never compare lattice-validation KG (N=49,
~0.02) with real-data disordered KG (N=2500, ~0.13): different regimes.

---

## Programmatic use

```python
from inference.correctors import WholeCloudCorrector2D, WholeCloudCorrector2DConfig
import numpy as np

pts = np.load('artifacts/inference/experiments/sph_tv/data/positions_without.npy')[300]  # (2500, 2)

wc = WholeCloudCorrector2D(WholeCloudCorrector2DConfig.from_yaml(
    'src/configs/experiments/sph_tv/model12_wholecloud.yaml'))
corrected = wc.apply(pts.astype(np.float32), k=5)                # (2500, 2)
```

Every corrector satisfies the `Corrector` ABC: `apply(points, k=1) -> points`,
same coordinate frame in and out (wrapped into `[0, box)` when `box` is set).

---

## Artifacts

`artifacts/` is gitignored and holds everything that is not source. **These
files back the paper and exist only on this machine — keep an off-machine
copy.**

```
artifacts/
  training/
    train_run_2026-07-15_11-18-44/   model12_sph_l4 provenance — KEEP
    train_run_2026-07-15_*/          its λ3/L/optimizer sweep arms — KEEP
  inference/experiments/
    sph_tv/
      data/                          positions.npy (TV), positions_without.npy (raw)
                                     — external inputs, irreplaceable — KEEP
      for_sim/                       sim-validated states + whole-cloud corrected
                                     trajectory; model12_wc_t* = wholecloud slices,
                                     model12_t* = historical grid output (see its
                                     README.txt) — tests + kg_sweep read these — KEEP
      runs/                          kg_sweep_*/ (metrics.csv) + experiment outputs
    apply_corrector/runs/            positions_corrected_K5.npy — model9 comparison
                                     series, regenerable only on main — KEEP
    obstruction/runs/                obstacle figures + reports
```

Run outputs are regenerable — delete freely, except the KEEP-marked items.
