"""
Packedness sweep: model9 at rd=0.05 — packedness in {0.75, 0.90}.

Generates per-run configs under configs/sweep/, tracks completion in
sweep_state.json.  Re-run at any time to resume after a failure.

Usage:  .venv\\Scripts\\python.exe sweep_packedness.py
"""
import json, math, subprocess, sys
from pathlib import Path
import yaml

ROOT   = Path(__file__).parent.parent          # project root
SWEEP  = ROOT / "configs" / "sweep"
STATE  = Path(__file__).parent / "sweep_state.json"   # state lives in sweeps/
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")

# ── sweep definition ────────────────────────────────────────────────────────
RD           = 0.05
PACKEDNESS   = [0.75, 0.90, 1.00]
SQRT3        = math.sqrt(3)

def n_target(p):   return round(p * 2.0 / (SQRT3 * RD**2))
def val_size(N):   return max(16, int(64 * (231 / N)**2))   # scale val set with N^2

# ── config generation ────────────────────────────────────────────────────────
def make_configs(p):
    N, bs, vs = n_target(p), 8, val_size(n_target(p))
    tag = f"p{int(p * 100):03d}"
    SWEEP.mkdir(parents=True, exist_ok=True)

    dataset = {
        "type": "packed", "packedness": p, "dim": 2, "rd": RD,
        "seed": 42, "noise_scale_min": 0.0, "noise_scale_max": 0.02,
    }
    loss = {"name": "hybrid_loss", "params": {
        "lambda1": 20, "lambda1_quad": 0,
        "lambda2": round(20.0 / (N - 1) / 10, 6),
    }}
    # base train config — override only memory-sensitive fields
    train = yaml.safe_load(
        (ROOT / "configs/trainer_configs/train_config_packed.yaml").read_text(encoding="utf-8")
    )
    train["eval"]["validation_size"] = vs

    paths = {}
    for name, cfg in [("dataset", dataset), ("loss", loss), ("train", train)]:
        path = SWEEP / f"{name}_{tag}.yaml"
        path.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")
        paths[name] = str(path.relative_to(ROOT))
    return paths, N, vs

# ── state helpers ────────────────────────────────────────────────────────────
def load_state(): return json.loads(STATE.read_text()) if STATE.exists() else {}
def save_state(s): STATE.write_text(json.dumps(s, indent=2))

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    state = load_state()
    for p in PACKEDNESS:
        tag = f"p{int(p * 100):03d}"

        if state.get(tag) == "done":
            print(f"[skip] {tag} — already done"); continue

        paths, N, vs = make_configs(p)
        print(f"\n{'='*58}")
        print(f"[run]  packedness={p}  N~{N}  batch=8  val={vs}")
        print(f"{'='*58}")

        rc = subprocess.run([
            PYTHON, "-m", "training.trainer",
            "--train-config",   paths["train"],
            "--dataset-config", paths["dataset"],
            "--loss-config",    paths["loss"],
            "--model-config",   "configs/model_configs/model_config_9.yaml",
        ], cwd=ROOT).returncode

        state[tag] = "done" if rc == 0 else "failed"
        save_state(state)

        if rc != 0:
            print(f"\n[FAIL] {tag} exited with code {rc}")
            print("Fix the issue and re-run — completed runs will be skipped.")
            sys.exit(1)

        print(f"[ok]   {tag} done")

    print("\nAll runs complete.")

if __name__ == "__main__":
    main()
