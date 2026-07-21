# Production model weights

Three checkpoints, all 2D. Two `model9_<training setup>` (SPH inference) plus
one `model12_sph_l4` (SPH kernel-gradient symmetry corrector). **Each
checkpoint pairs with exactly one model config** — do not swap them;
`max_displacement` / `hidden_dim` differ and you'll get an error or silent
garbage.

> This is the `sph-use-case` branch — 2D SPH only. The 3D MD-init
> checkpoints (`model9_3d_*`) live on `simplify`/`main`.

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

## SPH kernel-gradient corrector (model12)

| File | Model config | N_train | rd_train | attention_rd |
|---|---|---|---|---|
| `model12_sph_l4.pt` | `src/configs/training/model/model_config_12_sph_L4.yaml` | 49 (7×7 periodic lattice) | 0.14 | 0.286 |

model12 (`src/models/architectures/model12/model12.py`) does 4 rounds of
message passing with smooth proximity attention (not violation-gated), so it
sees the non-violating pairwise asymmetry that drives the SPH
kernel-gradient symmetry term — trained against `rdsph_loss` (hybrid
violation + displacement reg + λ3 × kernel-gradient symmetry). Calling
convention differs from model9: pass `rd=attention_rd` (not the constraint
rd) and `box=1.0` (periodic minimum-image geometry) — see model12.py's
docstring. `GridCorrector2D`/`KDTreeCorrector2D` now carry the box-aware
adapter for the `attention_rd`/`box` calling convention (run it via
`src/configs/experiments/sph_tv/model12_*.yaml`). Note the dense-edge tile
ceiling: `forward()` materialises `(B, N, N)` edges (N ~ 50-250); the
`forward_sparse` edge-list path removes it for whole-cloud inference.

Final validated metrics (K=5, full 256-cloud fixed validation set):
viol_reduction 89.6%, illegal_pairs 0.39%, mean|KG| 0.0220 — vs. the
first-cut baseline's 84.8% / 0.57% / 0.0394. Reached via a same-day sweep:
num_layers 3→4, λ3 0.09→0.27 (violation-vs-symmetry trade-off knob), and
Adam→AdamW (decoupled weight decay). Full history: `[[sph-model12-sweep]]`
memory.

## Provenance

| File | Training run | Training config |
|---|---|---|
| `model9_n100_p050.pt` | `artifacts/training/train_run_2026-05-28_14-53-45/` | `dataset_config_packed` + `loss_config_packed` + `train_config_packed` |
| `model9_n50_sparse.pt` | `artifacts/training/train_run_2026-05-26_17-34-01/` | `dataset_config_3` + `loss_config_5` + `train_config_2` |
| `model12_sph_l4.pt` | `artifacts/training/train_run_2026-07-15_11-18-44/` (`model_best.pt` — NOT `model_final.pt`; sha256-verified 2026-07-21) | `dataset_config_sph` + `loss_config_rdsph_lam3_0p27` + `train_config_sph_adamw` + `model_config_12_sph_L4` |

Note: the two model9 checkpoints predate the 2026-07-13 data-generation fix
— they were effectively trained on a single frozen batch (32 and 8 clouds
respectively) due to a fixed-seed bug. They work well regardless, but a
retrain with the fixed generator may improve them. `model12_sph_l4` trained
with the fix. The shipped `dataset_config_packed.yaml` (rd=0.05 → N≈231) has
also drifted from what `model9_n100_p050` recorded (rd_train 0.076, N≈100).
