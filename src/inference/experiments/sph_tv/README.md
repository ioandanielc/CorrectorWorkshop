# sph_tv

The model12 SPH-corrector experiments on the real 2D SPH trajectory (N=2500,
T=1002, PBC). Two scripts:

**sph_model12_experiment.py** — apply the whole-cloud corrector to sampled
timesteps; reports mean nn-distance + illegal% before/after. Output:
`artifacts/inference/experiments/sph_tv/runs/model12_wholecloud_<timestamp>/`.

```bash
.venv\Scripts\python.exe src/inference/experiments/sph_tv/sph_model12_experiment.py src/configs/experiments/sph_tv/model12_wholecloud.yaml
```

**kg_sweep.py** — full-trajectory metrics (mean|KG|, mean nn, illegal frac at
every timestep) for the four precomputed series: raw (non-TV), TV baseline,
model9-K5 (kept artifact — regenerable only on main), model12 whole-cloud.
No corrector runs — pure measurement of existing artifacts. Output:
`runs/kg_sweep_<timestamp>/metrics.csv` + report.

```bash
.venv\Scripts\python.exe src/inference/experiments/sph_tv/kg_sweep.py
```

Headline (disordered regime, t >= 300, full resolution): mean|KG| raw 0.326 /
TV 0.274 / model9 1.278 / model12 whole-cloud **0.128**; KG floor ~0.111.
