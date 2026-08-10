# vibecoding

An almost-exclusively **vibecoded resources sandbox**.

Nothing in here touches the critical infrastructure (the `src/` linear pipeline,
the shipped checkpoint, or the test-guarded correctors). It's the scratch space
for the fast, throwaway, exploratory work that shouldn't clutter the production
tree: early prototyping, one-off visualizations, parameter sweeps, and sundry
experiments. Code here is prototype-grade — not reviewed, not tested, not
imported by anything in `src/`. Treat it as disposable.

## Structure

The folder is organized by **goal**, not by module:

| Subfolder | What lives here |
|---|---|
| `visualizations/` | Figures and plotting scripts — illustrate how something works, produce paper/slide candidates, sanity-check data by eye. |
| `sweeps/` | Parameter/ablation sweep drivers and their scratch outputs — quick grids over λ3, N, noise, etc. |
| `misc/` | Everything else — throwaway notebooks, probes, and one-shot scripts that don't fit the two above. |

Each script drops its own outputs next to itself (e.g. `visualizations/outputs/`).

## Running

Everything assumes the repo root as the working directory and the project venv,
same as `src/`:

```
.venv\Scripts\python.exe vibecoding/visualizations/datagen_stages.py
```

Scripts put `src/` on `sys.path` themselves, so they can import the real
`training` / `inference` code — the whole point is to drive the production
pipeline from quick, unpolished harnesses.

## Contents

- **`visualizations/datagen_stages.py`** — side-by-side panels of the SPH data
  *generation* pipeline (datagen.py): lattice → + jitter → torus placement (clean)
  → + noise (noisy). Pre-torus steps are drawn flat; the on-torus stages are drawn
  on an actual 3D torus (isometric, surface grid, depth-cued points). Points are
  colored by nearest-neighbor distance. No correction step — that's inference, not
  the dataset.
