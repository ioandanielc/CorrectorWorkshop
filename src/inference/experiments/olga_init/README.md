# olga_init

Corrector for Olga's MD initialization. A generative model proposes an initial
LS1 state (3D, LJ units: N=5096, cubic box L=18, T=1.4, rho=0.874) with a few
far-too-close pairs (min distance ~0.08) that spike the potential energy and
cost ~100 extra MD timesteps of equilibration. This experiment corrects the
positions (velocities untouched), saves the corrected `(N, 6)` state for LS1,
and plots the RDF before/after.

Input: `artifacts/inference/experiments/olga_init/data/data.npy`, shape `(N, 6)` = x y z vx vy vz.
Output: `artifacts/inference/experiments/olga_init/runs/<config>_<timestamp>/` — `data_corrected.npy`,
`rdf.png`, `report.txt`.

One config per corrector variant (n50/n100 checkpoint × grid/kdtree/both) in
`src/configs/experiments/olga_init/` — run them all and compare the RDFs:

```bash
.venv\Scripts\python.exe src/inference/experiments/olga_init/olga_init_experiment.py src/configs/experiments/olga_init/n50_grid.yaml
.venv\Scripts\python.exe src/inference/experiments/olga_init/olga_init_experiment.py src/configs/experiments/olga_init/n50_kdtree.yaml
.venv\Scripts\python.exe src/inference/experiments/olga_init/olga_init_experiment.py src/configs/experiments/olga_init/n50_grid_kdtree.yaml
```
