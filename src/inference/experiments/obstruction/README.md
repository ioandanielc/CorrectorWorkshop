# obstruction

Corrector behaviour around domain obstacles (gear mask): the obstacle interior
is filled with fixed ghost particles so the corrector pushes real particles
away from the boundary. Two initialization strategies, corrected cumulatively
per K. `obstruction_demo.py` just visualizes the three obstacle types.
Figures go to `artifacts/inference/experiments/obstruction/runs/`.

```bash
.venv\Scripts\python.exe src/inference/experiments/obstruction/obstruction_experiment.py
```
