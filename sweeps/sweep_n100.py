"""
sweep_n100.py — Sequential sweep of N=100 corrector models at packedness 0.50/0.75/1.00.

Each variant uses a rd computed so that N_target=100 at the given packedness:
  p=0.50 → rd=0.0760,  p=0.75 → rd=0.0931,  p=1.00 → rd=0.1075

Tracks state in n100_state.json — re-run to resume after failure.

Usage:  .venv\\Scripts\\python.exe sweep_n100.py
"""
import json, subprocess, sys
from pathlib import Path

ROOT   = Path(__file__).parent.parent
STATE  = Path(__file__).parent / "n100_state.json"
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")
TRAIN  = "configs/sweep/train_small_n.yaml"

VARIANTS = [
    ("n100_p050", "configs/sweep/dataset_n100_p050.yaml",
                  "configs/sweep/loss_n100_p050.yaml",
                  "configs/model_configs/model_config_9_n100_p050.yaml"),
    ("n100_p075", "configs/sweep/dataset_n100_p075.yaml",
                  "configs/sweep/loss_n100_p075.yaml",
                  "configs/model_configs/model_config_9_n100_p075.yaml"),
    ("n100_p100", "configs/sweep/dataset_n100_p100.yaml",
                  "configs/sweep/loss_n100_p100.yaml",
                  "configs/model_configs/model_config_9_n100_p100.yaml"),
]

def load_state(): return json.loads(STATE.read_text()) if STATE.exists() else {}
def save_state(s): STATE.write_text(json.dumps(s, indent=2))

def main():
    state = load_state()
    for name, dataset_cfg, loss_cfg, model_cfg in VARIANTS:
        if state.get(name) == "done":
            print(f"[skip] {name}"); continue

        print(f"\n{'='*58}\n[run]  {name}\n{'='*58}")
        rc = subprocess.run([
            PYTHON, "-m", "training.trainer",
            "--train-config",   TRAIN,
            "--dataset-config", dataset_cfg,
            "--loss-config",    loss_cfg,
            "--model-config",   model_cfg,
        ], cwd=ROOT).returncode

        state[name] = "done" if rc == 0 else "failed"
        save_state(state)

        if rc != 0:
            print(f"\n[FAIL] {name} exit={rc} — re-run to retry"); sys.exit(1)
        print(f"[ok]   {name}")

    print("\nAll variants complete.")

if __name__ == "__main__":
    main()
