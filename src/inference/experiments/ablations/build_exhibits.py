"""
build_exhibits.py
-----------------
Turns `paper/results.csv` (written by score_arm.py) into the paper's tables, in
both Markdown and LaTeX. Every number is traced to the CSV row it came from, so
nothing in a figure can be quoted that is not in the results file.

    .venv\\Scripts\\python.exe src/inference/experiments/ablations/build_exhibits.py

Exhibits, matching the agreed structure — three in the main body, the per-axis
breakdown in an appendix:

  E1  architecture      one row per architecture at matched N and noise
  E2  loss 2x2          full / lambda3=0 / pure-symmetry / lambda2=0
  E3  cardinality       train-N x deploy-N matrix (KG and violation reduction)
  E4  packing           rd/spacing 0.80 -> 1.00 x architecture; the margin column
                        is what the "closer to a hard constraint" claim rests on
  E5  bridge            model12 -> DGCNN, one mechanism per rung

Arms are selected by name prefix, so `score_arm.py --arm` naming is the contract:
    arch_<model>_n<N>_noise<r>      loss_<variant>      card_n<N>
    pack_<model>_rd<frac>           bridge_<rung>
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SEED_SPREAD = {'viol_red': 1.4, 'kg_rel': 0.10}   # measured, see paper/JOURNAL.md


def load(csv_path):
    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if k not in ('arm', 'regime', 'checkpoint') and v not in ('', None):
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
    return rows


def md_table(headers, rows):
    esc = lambda c: str(c).replace('|', r'\|')   # |KG| would otherwise split the cell
    out = ['| ' + ' | '.join(esc(h) for h in headers) + ' |',
           '|' + '|'.join('---' for _ in headers) + '|']
    out += ['| ' + ' | '.join(esc(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def tex_table(headers, rows, caption, label):
    spec = 'l' + 'r' * (len(headers) - 1)
    body = ' \\\\\n'.join(' & '.join(str(c).replace('%', r'\%') for c in r) for r in rows)
    return (f'\\begin{{table}}[t]\n\\centering\n\\begin{{tabular}}{{{spec}}}\n\\hline\n'
            + ' & '.join(h.replace('%', r'\%') for h in headers)
            + f' \\\\\n\\hline\n{body} \\\\\n\\hline\n\\end{{tabular}}\n'
            f'\\caption{{{caption}}}\n\\label{{tab:{label}}}\n\\end{{table}}')


def _fmt(v, nd=4):
    return f'{v:.{nd}f}' if isinstance(v, float) else str(v)


def exhibit_generic(rows, prefix, title, key_label, sort_key=None):
    """One row per matching arm: the shared shape of E1/E2/E5."""
    sel = [r for r in rows if r['arm'].startswith(prefix) and r['regime'] == 'synthetic']
    if not sel:
        return None
    sel.sort(key=sort_key or (lambda r: -_num(r, 'viol_red_k5')))
    best_kg = min((_num(r, 'kg_k5') for r in sel), default=None)
    headers = [key_label, 'params', 'viol_red K5', 'illegal% K5', 'mean nn', '|KG| K5', 'vs best']
    body = []
    for r in sel:
        kg = _num(r, 'kg_k5')
        rel = '' if best_kg in (None, 0) else (
            'best' if kg == best_kg else f'+{100 * (kg - best_kg) / best_kg:.0f}%')
        body.append([r['arm'][len(prefix):].lstrip('_') or r['arm'],
                     f'{int(_num(r, "n_params")):,}',
                     f'{_num(r, "viol_red_k5"):.1f}%',
                     f'{_num(r, "ill_pct_k5"):.2f}',
                     _fmt(_num(r, 'nn_k5')),
                     _fmt(kg), rel])
    return title, headers, body


def _num(row, key):
    v = row.get(key, float('nan'))
    return v if isinstance(v, float) else float('nan')


def exhibit_cardinality(rows):
    """train-N x deploy-N matrix; cells are |KG| K5 with viol_red underneath."""
    sel = [r for r in rows if r['arm'].startswith('card_') and r['regime'] == 'synthetic']
    if not sel:
        return None
    trains = sorted({int(_num(r, 'train_N')) for r in sel})
    deploys = sorted({int(_num(r, 'deploy_N')) for r in sel})
    cell = {(int(_num(r, 'train_N')), int(_num(r, 'deploy_N'))): r for r in sel}
    headers = ['train N \\ deploy N'] + [str(d) for d in deploys]
    body = []
    for t in trains:
        row = [str(t)]
        for d in deploys:
            r = cell.get((t, d))
            row.append('—' if r is None
                       else f'{_num(r, "kg_k5"):.4f} / {_num(r, "viol_red_k5"):.0f}%')
        body.append(row)
    return ('E3 Cardinality: |KG| K5 / violation reduction, diagonal = in-distribution',
            headers, body)


def exhibit_packing(rows):
    """rd/spacing x architecture, with model12's margin over each baseline."""
    sel = [r for r in rows if r['arm'].startswith('pack_') and r['regime'] == 'synthetic']
    if not sel:
        return None
    by = defaultdict(dict)
    for r in sel:
        _, model, frac = r['arm'].split('_', 2)
        by[frac][model] = r
    models = sorted({m for d in by.values() for m in d})
    headers = ['rd/spacing'] + [f'{m} |KG|' for m in models] + ['model12 margin vs best baseline']
    body = []
    for frac in sorted(by, reverse=True):
        row = [frac.replace('rd', '')]
        kgs = {m: _num(by[frac][m], 'kg_k5') for m in models if m in by[frac]}
        for m in models:
            row.append(_fmt(kgs.get(m, float('nan'))))
        base = [v for m, v in kgs.items() if m != 'model12']
        if 'model12' in kgs and base and min(base) > 0:
            row.append(f'{100 * (min(base) - kgs["model12"]) / min(base):.0f}% better')
        else:
            row.append('—')
        body.append(row)
    return ('E4 Packing limit: |KG| K5 by constraint rigidity', headers, body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(ROOT / 'paper/results.csv'))
    ap.add_argument('--out', default=str(ROOT / 'paper/EXHIBITS.md'))
    ap.add_argument('--tex', default=str(ROOT / 'paper/exhibits.tex'))
    args = ap.parse_args()

    path = Path(args.results)
    if not path.exists():
        raise SystemExit(f'{path} does not exist yet — run score_arm.py first')
    rows = load(path)

    exhibits = [
        exhibit_generic(rows, 'arch_', 'E1 Architecture (matched params, identical loss)', 'architecture'),
        exhibit_generic(rows, 'loss_', 'E2 Loss ablation', 'loss variant'),
        exhibit_cardinality(rows),
        exhibit_packing(rows),
        exhibit_generic(rows, 'bridge_', 'E5 Architecture bridge (one mechanism per rung)', 'rung'),
    ]
    exhibits = [e for e in exhibits if e]

    md = ['# Paper exhibits',
          '',
          f'Generated from `{path.relative_to(ROOT)}` by `build_exhibits.py`. Every number',
          'here is a row in that file - do not hand-edit.',
          '',
          f'**Seed variance** (measured, two runs of the identical recipe): viol_red '
          f'+-{SEED_SPREAD["viol_red"]} points, |KG| +-{100 * SEED_SPREAD["kg_rel"]:.0f}% relative.',
          'Differences smaller than that are not separable from initialisation noise.',
          '']
    tex = []
    for title, headers, body in exhibits:
        md += [f'## {title}', '', md_table(headers, body), '']
        tex.append(tex_table(headers, body, title, title.split()[0].lower()))

    Path(args.out).write_text('\n'.join(md), encoding='utf-8')
    Path(args.tex).write_text('\n\n'.join(tex), encoding='utf-8')
    print(f'{len(exhibits)} exhibits from {len(rows)} rows')
    print(f'-> {args.out}')
    print(f'-> {args.tex}')


if __name__ == '__main__':
    main()
