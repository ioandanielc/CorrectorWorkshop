# obstruction

Corrector behaviour around domain obstacles (gear mask): the obstacle interior
is filled with fixed ghost particles (`obstruction.py` — masks + eroded fill)
so the corrector pushes real particles away from the boundary. Scenario: noisy
grid outside the obstacle; correction: k whole-cloud passes with the ghosts
re-pinned each pass. Output (initial-vs-corrected figure + report) goes to
`artifacts/inference/experiments/obstruction/runs/wholecloud_<timestamp>/`.

Re-run 2026-07-21 with the wholecloud corrector: 6100 real + 832 ghost
particles, mean nn 0.0076 → 0.0117 (rd = 0.012) in 5 passes — on a bounded,
non-periodic scene at an rd ~12× below the training scale.

```bash
.venv\Scripts\python.exe src/inference/experiments/obstruction/obstruction_experiment.py
```
