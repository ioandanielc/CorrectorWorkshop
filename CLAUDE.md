# CorrectorWorkshop — Claude Onboarding

## What this project is

A PyTorch training framework for a **Poisson disk corrector model**: given a noisy point cloud that violates the Poisson disk constraint (minimum distance `rd` between all point pairs), the model predicts per-point displacement vectors to produce a valid cloud.

The active model is **model9** (violation-weighted edge network, output-clamped). Training data is generated online using `scipy.stats.qmc.PoissonDisk`.

---

## Project structure

```
CorrectorWorkshop/
├── data/
│   ├── data_generator.py       # PoissonDiskDataset + PackedPoissonDiskDataset
│   └── data_processor.py       # DataProcessor — make_invariant (centers + normalises)
├── models/
│   └── fixed_rd/
│       ├── model6.py           # ViolationAwareEdgeNetwork — uniform-push baseline
│       ├── model7.py           # ViolationWeightedEdgeNetwork — violation-weighted agg
│       ├── model9.py           # model7 + deeper edge MLP (3→3 layers) + tanh clamping (CURRENT)
│       └── archive/            # model1–5 (superseded)
├── training/
│   ├── trainer.py              # Main training loop (iterative unrolling, dual K=1/K=K eval)
│   └── loss.py                 # hybrid_loss (linear viol penalty + displacement reg)
├── utils/
│   ├── config.py               # load_*_config helpers (yaml → dict)
│   ├── logger.py               # setup_logger, create_run_dir
│   └── visualizations.py       # plot_comparison, make_sample_gif
├── configs/
│   ├── trainer_configs/        # train_config_2 (K=3), _3 (K=5), _packed (packed regime)
│   ├── dataset_configs/        # dataset_config_3 (50pts sparse), dataset_config_packed (N≈231, p=0.5)
│   ├── loss_configs/           # loss_config_5 (principled sparse), loss_config_packed (N≈231)
│   ├── model_configs/          # model_config_7, model_config_8, model_config_9 (current)
│   ├── smoke_test/             # fast CPU configs for quick testing
│   ├── sweep/                  # auto-generated per-run configs (sweep_packedness.py)
│   └── archive/                # superseded configs
├── analysis/                   # comparison outputs + scripts (comparison_report.md, *.png)
├── docs/                       # architecture diagrams, PLANS.md
├── logs/                       # stdout/stderr logs from background training runs
├── training_artifacts/         # auto-created per run (see below)
├── sweep_packedness.py         # sweeper — packedness in {0.75, 0.90}, restart-safe
├── tracker.py                  # Streamlit live dashboard — run with: streamlit run tracker.py
├── COMMANDS.md                 # reference commands for common tasks
└── CLAUDE.md                   # this file
```

---

## Running

**Smoke test (fast, CPU-friendly):**
```
.venv\Scripts\python.exe -m training.trainer ^
  --train-config   configs/smoke_test/train_config.yaml ^
  --dataset-config configs/smoke_test/dataset_config.yaml ^
  --loss-config    configs/smoke_test/loss_config.yaml ^
  --model-config   configs/smoke_test/model_config_9.yaml
```

**Packed regime (model9, K=3, packedness=0.5):**
```
.venv\Scripts\python.exe -m training.trainer ^
  --train-config   configs/trainer_configs/train_config_packed.yaml ^
  --dataset-config configs/dataset_configs/dataset_config_packed.yaml ^
  --loss-config    configs/loss_configs/loss_config_packed.yaml ^
  --model-config   configs/model_configs/model_config_9.yaml
```

**Packedness sweep (0.75 → 0.90, restart-safe):**
```
.venv\Scripts\python.exe sweep_packedness.py
```

---

## Training artifacts

Each run creates `training_artifacts/train_run_YYYY-MM-DD_HH-MM-SS/` containing:
```
run_dir/
├── configs/              # copies of all 4 configs used
├── samples/              # sample_XXXXXX.png — side-by-side noisy vs corrected
├── validation_set.npy    # fixed validation set (raw, pre-processing)
├── loss.csv              # per-iteration metrics
├── training.log          # full logger output
├── evolution.gif         # animated sample progression
└── model_final.pt        # saved model weights
```

---

## Metrics logged

Eval runs at every `log_interval`, reporting **K=1 (deploy)** and **K=unroll_steps (train)** columns side by side.

| Metric | Meaning | Target |
|---|---|---|
| `loss` | hybrid loss (linear viol penalty + displacement reg) | decreasing |
| `mean_violation` | avg `relu(rd − dist)` per pair | → 0 |
| `illegal_pairs %` | % of point pairs closer than `rd` | → 0% |
| `viol_per_cloud` | mean count of unique illegal pairs per cloud | → 0 |
| `viol_reduction %` | `(viol_pre − viol_post) / viol_pre × 100` | → 100% |
| `mean_nn_dist` | mean nearest-neighbour distance | ≥ rd, not >> rd |
| `displacement` | mean L2 displacement (in rd units) | low |
| `correction_eff` | violation removed ÷ displacement | high |

---

## Config reference

**train_config**: `batch_size`, `num_iterations`, `initialization`, `device`, `unroll_steps` (K), `optimizer`, `lr_scheduler`, `eval`

**dataset_config**: `type` (packed | standard), `dim`, `rd`, `packedness` OR `points_per_cloud`, `seed`, `noise_scale_min`, `noise_scale_max`

**loss_config**: `name: hybrid_loss`, `params: {lambda1, lambda1_quad, lambda2}`
- Principled: `lambda1 = 1/rd`, `lambda1_quad = 0`, `lambda2 = lambda1 / (N−1) / 10`

**model_config**: `architecture`, `model_file`, `hidden_dim`, `norm`, `activation`, `max_displacement` (model9 only)

---

## Model lineage

| Model | Architecture | Key change | Status |
|---|---|---|---|
| model6 | uniform-push edge net | baseline | comparison only |
| model7 | violation-weighted agg | surgical corrections | superseded |
| model8 | model7 + GELU | activation swap | superseded |
| model9 | model8 + deeper MLP + tanh clamp | 3-layer edge MLP, bounded output | **CURRENT** |

---

## Experiment history

| Experiment | Config | Result |
|---|---|---|
| Sparse sparse (N=50, p≈0.11) | dataset_3 + loss_5 + train_2 | model9: 73.9% viol reduction ×1 |
| Packed (N≈231, p=0.5, rd=0.05) | dataset_packed + loss_packed + train_packed | model9: 78.7% viol reduction ×1, 96% ×3 |
| Model comparison | analysis/measure_comparison.py | model9 best efficiency (0.0288); model6 best raw clearance |

---

## Current state

- **model9** is active (violation-weighted, 3-layer edge MLP, tanh-clamped output)
- Packed regime benchmarked at packedness=0.5 (N≈231)
- **Running:** packedness sweep 0.75 → 0.90 to find the capacity limit of model9
- Next after sweep: width/depth capacity grid at the breaking-point packedness

---

## Key design decisions

- Data generated **online** (each batch is a fresh sample) — no dataset file
- `PackedPoissonDiskDataset` parameterised by `packedness` (fraction of triangular-lattice max), rd-independent
- Validation set generated once at run start, saved as `.npy` — fixed across training
- `DataProcessor.make_invariant` centers and normalises — do NOT repeat in the model
- Loss operates in **processed (invariant) space**, not original coordinates
- Intended deployment: **single-pass (K=1)** — unrolling is a training trick only
- `lambda2 = lambda1 / (N−1) / 10`: scales with N so violation penalty always dominates displacement reg by 10×

---

## Next steps

1. ✅ model7 — violation-weighted edge network
2. ✅ Iterative unrolling (K=3 training, K=1 eval)
3. ✅ model9 — deeper MLP + output clamping
4. ✅ Packed regime (N≈231, packedness=0.5)
5. ✅ Model comparison (model6 vs 7 vs 8 vs 9, sparse + packed)
6. ▶ **Packedness sweep** — 0.75 → 0.90 to find where model9 breaks
7. **Width/depth grid** — hidden_dim ∈ {128, 256, 512} × depth ∈ {3, 4} at the hard packedness
8. **Periodic boundary conditions** — toroidal domain (MIC wrapping)
