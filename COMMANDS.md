# CorrectorWorkshop — Command Reference

## Environment setup (first time only)

```bash
# Create venv
py -3.12 -m venv .venv

# Install PyTorch (CUDA 12.6 — compatible with driver CUDA 13.0+)
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu126

# Install remaining dependencies
.venv/Scripts/pip install scipy matplotlib pyyaml numpy streamlit plotly pillow
```

---

## Smoke test (CPU, completes in seconds — verify the setup works)

```bash
.venv/Scripts/python -m training.trainer \
  --train-config   configs/smoke_test/train_config.yaml \
  --dataset-config configs/smoke_test/dataset_config.yaml \
  --loss-config    configs/smoke_test/loss_config.yaml \
  --model-config   configs/smoke_test/model_config.yaml
```

### Smoke test — model4 variant

```bash
.venv/Scripts/python -m training.trainer \
  --train-config   configs/smoke_test/train_config.yaml \
  --dataset-config configs/smoke_test/dataset_config.yaml \
  --loss-config    configs/smoke_test/loss_config.yaml \
  --model-config   configs/smoke_test/model_config_4.yaml
```

---

## Full training runs (GPU)

### model1 — baseline (BatchNorm, concatenative skip, 2-D, 1024 pts)

```bash
.venv/Scripts/python -m training.trainer \
  --train-config   configs/trainer_configs/train_config_1.yaml \
  --dataset-config configs/dataset_configs/dataset_config_1.yaml \
  --loss-config    configs/loss_configs/loss_config_1.yaml \
  --model-config   configs/model_configs/model_config_1.yaml
```

### model4 — scaled for RTX 4080 SUPER (hidden_dim=512, 2-D, 50 pts)

```bash
.venv/Scripts/python -m training.trainer \
  --train-config   configs/trainer_configs/train_config_4.yaml \
  --dataset-config configs/dataset_configs/dataset_config_2.yaml \
  --loss-config    configs/loss_configs/loss_config_1.yaml \
  --model-config   configs/model_configs/model_config_4.yaml
```

---

## Live Training Tracker

Launch in a separate terminal while a training run is active:

```bash
.venv/Scripts/streamlit run tracker.py
```

Opens at http://localhost:8501 — auto-refreshes every 5 seconds.  
Shows: progress bar, metric cards, loss / violation / displacement / LR charts, and the latest sample image.

---

## Notes

- The model used in a run is controlled entirely by `--model-config` (the `model_file` key inside it). No code changes needed to switch models.
- Training artifacts are written to `training_artifacts/train_run_YYYY-MM-DD_HH-MM-SS/`.
- To run on CPU instead of CUDA, set `device: cpu` in the train config.
