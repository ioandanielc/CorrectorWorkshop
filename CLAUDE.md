# CorrectorWorkshop — Claude Onboarding

## What this project is

A PyTorch training framework for a **Poisson disk corrector model**: given a noisy point cloud that violates the Poisson disk constraint (minimum distance `rd` between all point pairs), the model predicts per-point displacement vectors to produce a valid cloud.

The model is a self-attention MLP over point sets (PointNet-style). Training data is generated online using `scipy.stats.qmc.PoissonDisk`.

---

## Project structure

```
CorrectorWorkshop/
├── data/
│   ├── data_generator.py     # PoissonDiskDataset — generates clean + noisy clouds
│   └── data_processor.py     # DataProcessor — make_invariant (centers + normalises)
├── models/
│   └── fixed_rd/
│       ├── model1.py         # Baseline: BatchNorm, concatenative skip
│       ├── model2.py         # + additive residuals (NOT YET USED in trainer)
│       └── model3.py         # + LayerNorm + tanh output (NOT YET USED in trainer)
├── training/
│   ├── trainer.py            # Main training loop
│   └── loss.py               # classic_loss, rd_weighted_loss
├── utils/
│   ├── config.py             # load_*_config helpers (yaml → dict)
│   ├── logger.py             # setup_logger, create_run_dir
│   └── visualizations.py     # plot_poisson_disk, plot_comparison
├── configs/
│   ├── trainer_configs/      # train_config_1.yaml
│   ├── dataset_configs/      # dataset_config_1.yaml
│   ├── loss_configs/         # loss_config_1.yaml
│   ├── model_configs/        # model_config_1/2/3.yaml
│   └── smoke_test/           # low-cost configs for quick testing
├── training_artifacts/       # auto-created per run (see below)
├── PLANS.md                  # pending model architecture improvements
└── CLAUDE.md                 # this file
```

---

## Running

**Smoke test (fast, CPU-friendly):**
```bash
.venv/bin/python -m training.trainer \
  --train-config  configs/smoke_test/train_config.yaml \
  --dataset-config configs/smoke_test/dataset_config.yaml \
  --loss-config   configs/smoke_test/loss_config.yaml \
  --model-config  configs/smoke_test/model_config.yaml
```

**Full training run:**
```bash
.venv/bin/python -m training.trainer \
  --train-config  configs/trainer_configs/train_config_1.yaml \
  --dataset-config configs/dataset_configs/dataset_config_1.yaml \
  --loss-config   configs/loss_configs/loss_config_1.yaml \
  --model-config  configs/model_configs/model_config_1.yaml
```

---

## Training artifacts

Each run creates `training_artifacts/train_run_YYYY-MM-DD_HH-MM-SS/` containing:
```
run_dir/
├── configs/          # copies of all 4 configs used
├── samples/          # sample_XXXXXX.png — side-by-side noisy vs corrected
├── validation_set.npy  # fixed validation set (raw, pre-processing)
├── loss.csv          # per-iteration metrics
├── training.log      # full logger output
└── model_final.pt    # saved model weights
```

---

## Metrics logged (all post-correction, on training batch)

| Metric | Meaning | Target |
|---|---|---|
| `loss` | weighted sum of illegality + displacement terms | decreasing |
| `mean_violation` | avg `relu(rd - dist)` per pair | 0 |
| `illegal_pairs` | % of point pairs closer than `rd` | 0% |
| `legal_clouds` | % of clouds with zero violations | 100% |
| `mean_nn_dist` | mean nearest-neighbour distance | ≥ rd |
| `displacement` | mean L2 displacement magnitude | low |

---

## Config structure

**train_config**: `batch_size`, `num_iterations`, `initialization`, `device`, `optimizer`, `lr_scheduler`, `eval` (validation_size, num_visual_samples, log_interval, sample_interval)

**dataset_config**: `dim`, `points_per_cloud`, `rd`, `seed`, `noise_scale_min`, `noise_scale_max`

**loss_config**: `name` (classic_loss / rd_weighted_loss), `params` (lambda1, lambda2, ...)

**model_config**: `hidden_dim`, `num_attention_modules`, `batch_norm` or `norm`, `activation`, optionally `max_displacement`

---

## Current state

- Trainer is working and smoke-tested on CPU
- model1 is used in the trainer — model2 and model3 exist but are NOT wired up yet
- The model import in `trainer.py` is hardcoded to `models.fixed_rd.model1`
- NumPy 2.x / torch compiled-with-NumPy-1.x warning appears on startup — non-fatal, just a warning

---

## Next steps (in order)

1. **Switch trainer to model2** — change import in `trainer.py`, run a proper training on GPU
2. **Switch trainer to model3** — change import, add `max_displacement` to model3 smoke config
3. **Evaluate** — compare loss curves and sample images across model1/2/3 runs
4. See `PLANS.md` for further architecture improvements beyond model3

---

## Key design decisions

- Data is generated **online** (each batch is a fresh sample) — no pre-built dataset file
- Validation set is generated once at run start and saved as `.npy` — always the same clouds
- First 6 clouds of the validation set are used for visualisation images
- `DataProcessor.make_invariant` centers and normalises the cloud — do NOT also do this in the model forward pass (model1 has redundant centroid subtraction; model2/3 removed it)
- Loss operates on the **processed** (invariant) space, not the original coordinates
- Device is set in `train_config` (e.g. `device: cpu` / `device: cuda` / `device: mps`)
