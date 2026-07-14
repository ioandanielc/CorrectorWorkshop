# Production model weights

Four checkpoints, named `model9_<training setup>`: two 2D (SPH inference) and
two 3D (MD-init repair). **Each checkpoint pairs with exactly one model
config** — do not swap them; `max_displacement` / `hidden_dim` differ and
you'll get an error or silent garbage.

## 2D

| File | Model config | N_train | rd_train | scale (rd_test=0.02) | Used by |
|---|---|---|---|---|---|
| `model9_n100_p050.pt` | `src/configs/training/model/model_config_9_n100_p050.yaml` | ~100 | 0.076 | 3.8 | `src/configs/experiments/sph_tv/grid_6x6.yaml` |
| `model9_n50_sparse.pt` | `src/configs/training/model/model_config_9.yaml` | 50 | 0.05 | 2.5 | `src/configs/experiments/sph_tv/grid_10x10.yaml` |

The experiment configs already wire the correct model config and tiling
together. Use those directly:

```bash
# model9_n100_p050 + 6x6 grid (recommended)
.venv\Scripts\python.exe src/inference/experiments/sph_tv/sph_tv_experiment.py src/configs/experiments/sph_tv/grid_6x6.yaml

# model9_n50_sparse + 10x10 grid
.venv\Scripts\python.exe src/inference/experiments/sph_tv/sph_tv_experiment.py src/configs/experiments/sph_tv/grid_10x10.yaml
```

`model9_n100_p050.pt` beats `model9_n50_sparse.pt` across all K values and
timesteps on the SPH data — reach for it by default. The packed training
regime (p=0.5, dense violations) better reflects the SPH violation structure
at t > 300.

## 3D

| File | Model config | N_train | rd_train | Trained on |
|---|---|---|---|---|
| `model9_3d_n50.pt` | `src/configs/training/model/model_config_9_3d_n50.yaml` | 50 | 0.15 | Poisson spheres, ~12% FCC density, noise σ ∈ [0, 1.0·rd] |
| `model9_3d_n100.pt` | `src/configs/training/model/model_config_9_3d_n100.yaml` | 100 | 0.12 | Poisson spheres, ~12% FCC density, noise σ ∈ [0, 1.0·rd] |

Used by the `olga_init` experiment (`src/configs/experiments/olga_init/`), or via
the correctors directly (README "Programmatic use").
`KDTreeCorrector3DConfig` defaults (`total_core=50`, `inner_core=12`) are
sized for `model9_3d_n50.pt`; pass `total_core=100, inner_core=25` for
`model9_3d_n100.pt`.

On sparse synthetic 3D clouds (N=1400, σ up to 1.0·rd) the two perform
identically — violations at ~12% density are pairwise-local, so a 50-point
neighbourhood already suffices. Prefer `model9_3d_n50.pt` (cheaper);
`model9_3d_n100.pt` exists for denser / clustered data where a 50-point
context may not contain the conflict.

## Provenance

| File | Training run | Training config |
|---|---|---|
| `model9_n100_p050.pt` | `artifacts/training/train_run_2026-05-28_14-53-45/` | `dataset_config_packed` + `loss_config_packed` + `train_config_packed` |
| `model9_n50_sparse.pt` | `artifacts/training/train_run_2026-05-26_17-34-01/` | `dataset_config_3` + `loss_config_5` + `train_config_2` |
| `model9_3d_n50.pt` | `artifacts/training/train_run_2026-07-13_20-06-41/` | `dataset_config_3d_n50` + `loss_config_3d_n50` + `train_config_3d` |
| `model9_3d_n100.pt` | `artifacts/training/train_run_2026-07-13_20-29-23/` | `dataset_config_3d_n100` + `loss_config_3d_n100` + `train_config_3d` |

Note: the 2D checkpoints predate the 2026-07-13 data-generation fix — they
were effectively trained on a single frozen batch (32 and 8 clouds
respectively) due to a fixed-seed bug. They work well regardless, but a
retrain with the fixed generator may improve them. The 3D checkpoints trained
with the fix. The shipped `dataset_config_packed.yaml` (rd=0.05 → N≈231) has
also drifted from what `model9_n100_p050` recorded (rd_train 0.076, N≈100).
