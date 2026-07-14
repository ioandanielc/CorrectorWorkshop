# olga_init

Corrector for Olga's MD initialization. A generative model proposes an initial
LS1 state (3D, LJ units: N=5096, cubic box L=18, T=1.4, rho=0.874) with a few
far-too-close pairs that spike the potential energy and cost ~100 extra MD
timesteps of equilibration (a well-separated start needs ~40). This experiment
corrects the positions (velocities untouched), saves the corrected `(N, 6)`
state for LS1, and plots the RDF before/after.

Input: `artifacts/inference/experiments/olga_init/data/data.npy`, shape `(N, 6)` = x y z vx vy vz.
Output: `artifacts/inference/experiments/olga_init/runs/<config>_<timestamp>/` — `data_corrected.npy`,
`rdf.png`, `report.txt`.

One config per variant in `src/configs/experiments/olga_init/`:

```bash
.venv\Scripts\python.exe src/inference/experiments/olga_init/olga_init_experiment.py src/configs/experiments/olga_init/n50_kdtree.yaml
```

## Results on Olga's data.npy (2026-07-14)

Input: min pair distance 0.1631; 10 pairs < 0.6, 477 < 0.85, 1483 < 0.90 —
the equilibrated liquid's g(r) takes off at ~0.85, so everything below 0.85
is defect tail and everything above is real structure.

| Setting | Result | Perturbation | Time |
|---|---|---|---|
| rd 0.6, any variant (n50/n100 × grid/kdtree/both) | 10 → 0 pairs, min 0.603–0.604, PBC-verified pure | 19–20 particles moved, max 0.24 | kdtree 0.03 s, grid 2–4 s |
| rd 0.85, kdtree defaults (`inner_core: 12`) | 477 → 2–3 pairs at 0.836+ (thin frozen-ring warnings) | ~900 moved, max 0.38 | 0.2 s |
| rd 0.85, `n50_kdtree_rd085.yaml` (`inner_core: 8`, k ≤ 25) | 477 → 0 pairs, min 0.8500, PBC-verified pure | 872 moved, max 0.36 | 0.2 s |

In every case g(r) is unchanged from the first peak outward; the rd-0.85
correction parks displaced pairs in a narrow spike just above the cutoff,
which the first few MD steps thermalize. At rd 0.6 the violations are sparse
→ all variants agree and kdtree is ~100× faster than grid; model choice
(n50 vs n100) makes no difference. Purity was verified two ways on the final
files: `cKDTree(boxsize)` pair queries and a brute-force minimum-image scan
over all ~13M pairs.

The package sent to Olga is in
`artifacts/inference/experiments/olga_init/for_olga/` — both variants as
`.npy` + self-documenting `.txt` (same data, header with method + stats) +
RDF figures. Her acceptance test: LS1 timesteps to reach U ≈ −4.61, vs her
~100 (raw) and ~40 (resampled reference) baselines; the rd-0.85 state is the
one expected to match ~40.
