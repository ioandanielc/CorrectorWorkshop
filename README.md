# Corrector — code for "A graph machine learning approach for correcting SPH simulations"

## Layout
- `src/models/architectures/` — the corrector (`model12/`) and the four baselines
- `src/training/` — data generation (`datagen.py`), loss (`loss.py`), training loop
- `src/inference/` — whole-cloud deployment, trajectory and obstacle experiments
- `src/configs/training/` — one config triple (dataset / model / loss) per arm
- `src/models/weights/` — the three checkpoints named in Appendix B

## Paper → config

| Paper | Config |
|---|---|
| Production model (Tables 1, 2, 5) | `training/{model/model_config_12_sph_L4, loss/loss_config_rdsph_lam3_0p27, dataset/dataset_config_sph, trainer/train_config_sph_adamw}.yaml` |
| Table 3, baselines | `training/ablations/architecture/` |
| Table 4, mechanism variants | `training/ablations/bridge/` |
| Table 2, loss ablation | `training/ablations/loss_config_rdsph_lam{2,3}_0.yaml` |
| Table 6, cardinality | `training/ablations/cardinality/` |

## Checkpoints
| File | Deployment mean\|KG\| (N=2500, t≥300, k=5) |
|---|---|
| `model12_sph_l4.pt` | 0.127 — the sim-validated one, quoted for all validated claims |
| `model12_sph_l4_noise1p0.pt` | 0.123 — best N=49 run |
| `model12_sph_n100.pt` | 0.087 — best deployer, single run, not solver-validated |

A checkpoint pairs with exactly one model config; `hidden_dim`, `cutoff_rd` and
`max_displacement` differ between them.

All baselines are trained with the same loss, the same recipe, and parameter
counts matched to 350k ±0.8%. No per-architecture hyperparameter search.
