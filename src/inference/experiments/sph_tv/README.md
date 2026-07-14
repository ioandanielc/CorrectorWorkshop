# sph_tv

ML corrector vs the Transport Velocity baseline on the 2D SPH trajectory
(N=2500, PBC), across a sweep of K values. Comparison figures, timeseries and
a report go to `artifacts/inference/experiments/sph_tv/runs/exp_<timestamp>/`.

```bash
.venv\Scripts\python.exe src/inference/experiments/sph_tv/sph_tv_experiment.py src/configs/experiments/sph_tv/grid_6x6.yaml
```
