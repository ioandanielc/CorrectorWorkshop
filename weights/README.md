# Production model weights

Two checkpoints for SPH inference. **They use different architectures and configs** —
do not swap the model config between them.

| File | Model config | N_train | r_d_train | scale | Tiling config |
|---|---|---|---|---|---|
| `n100_p050.pt` | `configs/model_configs/model_config_9_n100_p050.yaml` | ~100 | 0.076 | 3.8 | `inference/configs/grid_6x6.yaml` |
| `n50_sparse.pt` | `configs/model_configs/model_config_9.yaml` | 50 | 0.05 | 2.5 | `inference/configs/grid_10x10.yaml` |

The experiment configs (`inference/configs/`) already wire the correct model config
and tiling config together. Use those directly:

```bash
# n100_p050 + 6x6 grid (recommended)
.venv\Scripts\python.exe inference/run_experiment.py inference/configs/grid_6x6.yaml

# n50_sparse + 10x10 grid
.venv\Scripts\python.exe inference/run_experiment.py inference/configs/grid_10x10.yaml
```

**Do not mix** `n50_sparse.pt` with `model_config_9_n100_p050.yaml` or vice versa —
the `max_displacement` and hidden_dim differ between them.

## Recommended checkpoint

`n100_p050.pt` consistently outperforms `n50_sparse.pt` across all K values and
timesteps on the SPH data, despite `n50_sparse.pt` having a theoretically better
tile-size match at 10×10. The packed training regime (p=0.5, dense violations)
better reflects the SPH violation structure at t > 300.

## Provenance

| File | Training run | Training config |
|---|---|---|
| `n100_p050.pt` | `training_artifacts/train_run_2026-05-28_14-53-45/` | `dataset_config_packed` + `loss_config_packed` + `train_config_packed` |
| `n50_sparse.pt` | `training_artifacts/train_run_2026-05-26_17-34-01/` | `dataset_config_3` + `loss_config_5` + `train_config_2` |
