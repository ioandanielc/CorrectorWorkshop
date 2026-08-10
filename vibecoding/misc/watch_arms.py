"""
Live status of every training arm under artifacts/training/.

Reads each run's config snapshot to name the arm (architecture, N, noise level,
lambda3) and its training.log for the most recent K=5 eval block. Running arms
show progress and ETA; finished ones show their final numbers.

    .venv\\Scripts\\python.exe vibecoding/misc/watch_arms.py
    .venv\\Scripts\\python.exe vibecoding/misc/watch_arms.py --today --at 1500
"""
import argparse
import re
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'artifacts' / 'training'

ITER_RE = re.compile(r'iter=\s*(\d+)\s+([\d.]+)s/iter')
K5_RE = {
    'viol_red': re.compile(r'viol_reduction\s+=\s+\S*\s+-?[\d.]+%\s+(-?[\d.]+)%'),
    'illegal':  re.compile(r'illegal_pairs\s+=\s+[\d.]+%\s+->\s+[\d.]+%\s+([\d.]+)%'),
    'nn':       re.compile(r'mean_nn_dist\s+=\s+[\d.]+\s+->\s+[\d.]+\s+([\d.]+)'),
    'kg':       re.compile(r'mean_\|KG\|\s+=\s+([\d.]+)\s+->\s+[\d.]+\s+([\d.]+)'),
}


def arm_name(run: Path):
    cfg_dir = run / 'configs'
    if not cfg_dir.is_dir():
        return None
    arch, N, noise, rd, lam3 = '?', '?', None, None, None
    for p in cfg_dir.glob('*.yaml'):
        try:
            c = yaml.safe_load(open(p))
        except Exception:
            continue
        if not isinstance(c, dict):
            continue
        if 'model_file' in c:
            arch = c['model_file'].split('/')[-1]
        elif 'points_per_cloud' in c:
            N, rd, noise = c['points_per_cloud'], c['rd'], c['noise_scale_max']
        elif 'params' in c and isinstance(c['params'], dict):
            lam3 = c['params'].get('lambda3', c['params'].get('lambda_sph'))
    ratio = f'{noise / rd:.1f}' if (noise is not None and rd) else '?'
    # rd/spacing identifies a packing rung, where noise/rd is constant by construction
    pack = f'/rd{rd * round(N ** 0.5):.2f}' if (rd and isinstance(N, int)) else ''
    return f'{arch}/N{N}{pack}/noise{ratio}/lam3={lam3}'


def latest_block(log_text, at=None):
    """Most recent (or nearest-to-`at`) eval block: iteration, s/iter, K=5 metrics."""
    blocks = []
    for m in ITER_RE.finditer(log_text):
        it, sec = int(m.group(1)), float(m.group(2))
        tail = log_text[m.end():m.end() + 900]
        vals = {}
        for key, rx in K5_RE.items():
            g = rx.search(tail)
            if g:
                vals[key] = float(g.group(2) if key == 'kg' else g.group(1))
                if key == 'kg':
                    vals['kg_pre'] = float(g.group(1))
        blocks.append((it, sec, vals))
    if not blocks:
        return None
    if at is None:
        return blocks[-1]
    return min(blocks, key=lambda b: abs(b[0] - at))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--today', action='store_true', help='only runs started today')
    ap.add_argument('--at', type=int, default=None,
                    help='report the eval block nearest this iteration (fair cross-arm view)')
    args = ap.parse_args()

    today = datetime.now().strftime('%Y-%m-%d')
    runs = sorted(p for p in RUNS.glob('train_run_*') if p.is_dir())
    if args.today:
        runs = [r for r in runs if today in r.name]

    print(f'{"arm":42s} {"iter":>6s} {"s/it":>5s} {"viol_red":>9s} '
          f'{"illeg%":>7s} {"nn":>7s} {"|KG|pre":>8s} {"|KG|K5":>7s}  state')
    print('-' * 110)
    for run in runs:
        name = arm_name(run)
        if name is None:
            continue
        log = run / 'training.log'
        if not log.exists():
            continue
        text = log.read_text(errors='ignore')
        # seed arrives via CLI, so it is in the log rather than the config snapshot
        m = re.search(r'Torch seed: (\d+)', text)
        if m:
            name += f'/seed{m.group(1)}'
        blk = latest_block(text, args.at)
        done = 'Training complete' in text
        if blk is None:
            print(f'{name:42s} {"-":>6s} {"-":>5s} {"":>9s} {"":>7s} {"":>7s} '
                  f'{"":>8s} {"":>7s}  {"starting"}')
            continue
        it, sec, v = blk
        print(f'{name:42s} {it:6d} {sec:5.3f} {v.get("viol_red", float("nan")):8.1f}% '
              f'{v.get("illegal", float("nan")):7.2f} {v.get("nn", float("nan")):7.4f} '
              f'{v.get("kg_pre", float("nan")):8.4f} {v.get("kg", float("nan")):7.4f}  '
              f'{"done" if done else "RUNNING"}  {run.name}')


if __name__ == '__main__':
    main()
