# apply_corrector

Applies the grid corrector to every timestep of the no-TV SPH trajectory and
saves the corrected positions (`.npy` + human-readable `.txt`) to
`artifacts/inference/experiments/apply_corrector/runs/`. `inspector.py` / `replace_positions.py` are scratch
tools for pushing the output back into an `.h5part` file.

```bash
.venv\Scripts\python.exe src/inference/experiments/apply_corrector/apply_corrector.py --k 5
```
