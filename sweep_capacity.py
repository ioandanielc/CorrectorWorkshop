"""
Capacity grid at packedness=0.90 (N~416): vary hidden_dim and edge_depth.
Uses the p090 dataset/loss/train configs from sweep_packedness.py.
Tracks state in capacity_state.json — re-run to resume after failure.

Usage:  .venv\\Scripts\\python.exe sweep_capacity.py
"""
import json, subprocess, sys
from pathlib import Path

ROOT   = Path(__file__).parent
STATE  = ROOT / "capacity_state.json"
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")

VARIANTS = [
    # (name, model_cfg, train_cfg_override or None)
    ("hd128_d3", "configs/model_configs/model_config_9_hd128_d3.yaml", None),   # baseline
    ("hd256_d3", "configs/model_configs/model_config_9_hd256_d3.yaml", None),   # 2x width
    ("hd512_d3", "configs/model_configs/model_config_9_hd512_d3.yaml", None),   # 4x width
    ("hd256_d4", "configs/model_configs/model_config_9_hd256_d4.yaml",          # 2x width + deeper
     "configs/sweep/train_p090_bs4.yaml"),                                       # bs=4: OOM at bs=8
]

# p090 configs from packedness sweep
TRAIN   = "configs/sweep/train_p090.yaml"
DATASET = "configs/sweep/dataset_p090.yaml"
LOSS    = "configs/sweep/loss_p090.yaml"

def load_state(): return json.loads(STATE.read_text()) if STATE.exists() else {}
def save_state(s): STATE.write_text(json.dumps(s, indent=2))

def main():
    state = load_state()
    for name, model_cfg, train_override in VARIANTS:
        if state.get(name) in ("done", "skipped"):
            print(f"[skip] {name}"); continue

        train_cfg = train_override if train_override else TRAIN
        print(f"\n{'='*58}\n[run]  {name}  (train={train_cfg})\n{'='*58}")

        rc = subprocess.run([
            PYTHON, "-m", "training.trainer",
            "--train-config",   train_cfg,
            "--dataset-config", DATASET,
            "--loss-config",    LOSS,
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
