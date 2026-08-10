# Production model weights

Three deployable checkpoints. **A checkpoint pairs with exactly one model
config** — never swap; `max_displacement` / `hidden_dim` / `cutoff_rd` differ
and you'll get an error or silent garbage.

| File | Deployment mean\|KG\| (N=2500, t>=300, k=5) | Status |
|---|---|---|
| `model12_sph_n100.pt` | **0.0871** | **best deployer** (2026-08-10), n=1, NOT sim-validated |
| `model12_sph_l4_noise1p0.pt` | 0.1237 | best N=49 arm, n=3 (0.1230-0.1273) |
| `model12_sph_l4.pt` | 0.1269 | **the sim-validated one** — quote this for validated claims |

Read that table carefully: **the best checkpoint and the validated checkpoint are
not the same run.** The SPH re-simulation that confirmed these clouds are usable
restarts was performed with `model12_sph_l4.pt` at k=5. `model12_sph_n100.pt` is
31% better on the KG metric but has never been through a solver. Quote both and
say which is which.

The two 2026-08-10 checkpoints were rescued out of `artifacts/training/`, which
this repo documents as "delete freely" — they would otherwise have been lost.
Full results: `paper/RESULTS.md`.

> This is the `sph-use-case` branch — model12 / 2D SPH only. The model9
> family (2D checkpoints + the 3D MD-init `model9_3d_*`) lives on
> `simplify`/`main`.

## SPH kernel-gradient corrector (model12)

| File | Model config | N_train | rd_train | cutoff_rd |
|---|---|---|---|---|
| `model12_sph_l4.pt` | `src/configs/training/model/model_config_12_sph_L4.yaml` | 49 (7×7 periodic lattice) | 0.14 | 0.286 |
| `model12_sph_l4_noise1p0.pt` | same as above | 49 | 0.14 | 0.286 |
| `model12_sph_n100.pt` | `src/configs/training/ablations/cardinality/model_config_12_sph_L4_n100.yaml` | 100 (10×10) | 0.098 | 0.200 |

`model12_sph_l4_noise1p0.pt` differs from `model12_sph_l4.pt` only in training
data — noise 1.0·rd instead of 0.6·rd (`ablations/architecture/dataset_config_sph_noise1p0.yaml`).
`model12_sph_n100.pt` has a **different model config** (rd 0.098, cutoff 0.200,
λ3 0.070) — pairing it with the N=49 config will silently produce garbage.

model12 (`src/models/architectures/model12/model12.py`) does 4 rounds of
message passing with a smooth proximity kernel (not violation-gated), so it
sees the non-violating pairwise asymmetry that drives the SPH
kernel-gradient symmetry term — trained against `rdsph_loss` (hybrid
violation + displacement reg + λ3 × kernel-gradient symmetry). Calling
convention: pass `rd=cutoff_rd` (not the constraint rd) and `box=`
(periodic minimum-image geometry) — see model12.py's docstring.
`WholeCloudCorrector2D` carries the box-aware adapter; run via
`src/configs/experiments/sph_tv/model12_wholecloud.yaml`. `forward()`
materialises dense `(B, N, N)` edges (training-scale N only); the
`forward_sparse` edge-list path removes that ceiling — the corrector uses it
for whole-cloud inference (guarded bit-exactly by `tests/test_wholecloud.py`).

Final validated metrics (K=5, full 256-cloud fixed validation set):
viol_reduction 89.6%, illegal_pairs 0.39%, mean|KG| 0.0220 — vs. the
first-cut baseline's 84.8% / 0.57% / 0.0394. Reached via a same-day sweep:
num_layers 3→4, λ3 0.09→0.27 (violation-vs-symmetry trade-off knob), and
Adam→AdamW (decoupled weight decay). (Those are lattice-validation numbers,
N=49 — NOT comparable to the real-data disordered-regime mean|KG| ≈ 0.13 at
N=2500.) Full history: `[[sph-model12-sweep]]` memory.

## Provenance

| File | Training run | Training config |
|---|---|---|
| `model12_sph_l4.pt` | `artifacts/training/train_run_2026-07-15_11-18-44/` (`model_best.pt` — NOT `model_final.pt`; sha256-verified 2026-07-21) | `dataset_config_sph` + `loss_config_rdsph_lam3_0p27` + `train_config_sph_adamw` + `model_config_12_sph_L4` |
| `model12_sph_l4_noise1p0.pt` | `artifacts/training/train_run_2026-08-10_11-05-02/model_best.pt` | as above but `ablations/architecture/dataset_config_sph_noise1p0` |
| `model12_sph_n100.pt` | `artifacts/training/train_run_2026-08-10_19-15-20/model_best.pt` | `ablations/cardinality/{dataset,loss,model}_*_n100` + `train_config_sph_adamw` |

Both 2026-08-10 runs were trained **unseeded** (the `seed: 0` default was added
to the trainer config later the same day), so re-running will not reproduce them
byte-for-byte.

`model12_sph_l4` trained with the 2026-07-13 data-generation fix (fresh
batches). The λ3/L/optimizer sweep arms it was selected from survive as their training
runs' `configs/` snapshots under `artifacts/training/train_run_2026-07-15_*`
(the arm YAMLs themselves were purged from src/; also on `main`).

Historical (removed from this branch, still on `main`): the model9 2D
checkpoints `model9_n100_p050` / `model9_n50_sparse` and their configs. Their
KG-blowup failure (full-sweep disordered mean|KG| 1.278 vs raw 0.326) is the
motivating comparison in the kg_sweep — the model9-K5 corrected trajectory is
kept as an artifact at
`artifacts/inference/experiments/apply_corrector/runs/positions_corrected_K5.npy`.
