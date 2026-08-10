"""Build the final self-contained HTML report."""
import base64
from pathlib import Path

ROOT = Path(r'e:\VSCode\CorrectorWorkshop')
FIGS = ROOT / 'vibecoding/visualizations/outputs'
OUT = Path(r'C:\Users\ioand\AppData\Local\Temp\claude\e--VSCode-CorrectorWorkshop'
           r'\d723bad1-170c-4fd5-a8d3-892d03187a81\scratchpad\report.html')


def uri(name):
    return 'data:image/png;base64,' + base64.b64encode((FIGS / name).read_bytes()).decode()


CSS = """
:root{
  --bg:#fbfbf9; --surface:#f2f4f0; --line:#dee3db; --line2:#eceee9;
  --ink:#13171b; --muted:#59645d;
  --accent:#2d6a8e; --good:#2c7a57; --warn:#a94e46; --gold:#8a6d1f;
}
@media (prefers-color-scheme: dark){
  :root{ --bg:#0d1014; --surface:#151a20; --line:#252c35; --line2:#1b2129;
         --ink:#e6eae6; --muted:#8a948c;
         --accent:#5aa8cf; --good:#4fbb87; --warn:#e07b72; --gold:#d4b25e; }
}
:root[data-theme="dark"]{ --bg:#0d1014; --surface:#151a20; --line:#252c35; --line2:#1b2129;
  --ink:#e6eae6; --muted:#8a948c; --accent:#5aa8cf; --good:#4fbb87; --warn:#e07b72; --gold:#d4b25e; }
:root[data-theme="light"]{ --bg:#fbfbf9; --surface:#f2f4f0; --line:#dee3db; --line2:#eceee9;
  --ink:#13171b; --muted:#59645d; --accent:#2d6a8e; --good:#2c7a57; --warn:#a94e46; --gold:#8a6d1f; }

*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  font-size:16px;line-height:1.62;-webkit-font-smoothing:antialiased}
.mono,code{font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace}
code{font-size:.88em;background:var(--surface);padding:.1em .35em;border-radius:3px}

.wrap{max-width:1120px;margin:0 auto;padding:52px 26px 110px;
  display:flex;flex-direction:column;gap:0}

header{border-bottom:2px solid var(--ink);padding-bottom:22px;margin-bottom:8px}
.eyebrow{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11.5px;
  letter-spacing:.16em;text-transform:uppercase;color:var(--accent);margin-bottom:12px}
h1{margin:0 0 12px;font-size:clamp(28px,4vw,42px);line-height:1.1;font-weight:640;
  letter-spacing:-.022em;text-wrap:balance;max-width:20ch}
.lede{margin:0;color:var(--muted);max-width:68ch;font-size:17px}

section{padding:38px 0 4px;border-bottom:1px solid var(--line2)}
section:last-of-type{border-bottom:none}
h2{margin:0 0 6px;font-size:13px;font-weight:600;letter-spacing:.1em;
  text-transform:uppercase;color:var(--accent);
  font-family:ui-monospace,Menlo,Consolas,monospace}
h3{margin:0 0 14px;font-size:25px;font-weight:620;letter-spacing:-.014em;
  text-wrap:balance;max-width:30ch}
h4{margin:26px 0 8px;font-size:16.5px;font-weight:620}
p{margin:0 0 14px;max-width:72ch}
ul{margin:0 0 14px;padding-left:20px;max-width:72ch}
li{margin-bottom:6px}
.muted{color:var(--muted)}

.tablewrap{overflow-x:auto;margin:16px 0 20px}
table{border-collapse:collapse;width:100%;font-size:14px;
  font-variant-numeric:tabular-nums;min-width:520px}
th,td{padding:8px 13px;text-align:right;border-bottom:1px solid var(--line2);white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal}
thead th{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:10.5px;
  letter-spacing:.09em;text-transform:uppercase;color:var(--muted);font-weight:500;
  border-bottom:1px solid var(--line)}
tbody td:not(:first-child){font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
tr.best{background:color-mix(in srgb,var(--good) 9%,transparent)}
tr.best td{color:var(--good);font-weight:600}
tr.bad td{color:var(--warn)}
tr.hl{background:color-mix(in srgb,var(--accent) 8%,transparent)}
tr.sep td{border-top:2px solid var(--line)}
td.l{text-align:left}

figure{margin:22px 0 8px}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:5px;background:#fff}
.scroll img{display:block;width:100%;min-width:720px;height:auto}
figcaption{color:var(--muted);font-size:13.5px;margin-top:9px;max-width:78ch}

.callout{border-left:3px solid var(--accent);background:var(--surface);
  padding:14px 18px;margin:18px 0;border-radius:0 4px 4px 0;max-width:76ch}
.callout.warn{border-left-color:var(--warn)}
.callout.good{border-left-color:var(--good)}
.callout p:last-child{margin-bottom:0}
.callout strong{color:var(--ink)}

.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:14px;margin:18px 0 22px}
.stat{background:var(--surface);border:1px solid var(--line);border-radius:5px;padding:15px 17px}
.stat .k{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:10.5px;
  letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}
.stat .v{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:27px;
  font-weight:600;line-height:1.25;margin-top:5px;font-variant-numeric:tabular-nums}
.stat .n{font-size:12.5px;color:var(--muted);margin-top:3px;line-height:1.45}
.v.good{color:var(--good)} .v.warn{color:var(--warn)} .v.acc{color:var(--accent)}

.retraction{border:1px solid var(--line);border-left:3px solid var(--warn);
  border-radius:0 5px 5px 0;padding:14px 18px;margin-bottom:12px;background:var(--surface)}
.retraction .was{color:var(--warn);font-weight:600}
.retraction .now{color:var(--good)}
.retraction div{max-width:78ch}

footer{margin-top:52px;padding-top:22px;border-top:1px solid var(--line);
  color:var(--muted);font-size:13px}
"""


def tbl(headers, rows, cls=''):
    h = ''.join(f'<th>{c}</th>' for c in headers)
    b = ''
    for r in rows:
        c = r[0] if isinstance(r[0], str) and r[0] in ('best', 'bad', 'hl', 'sep', '') else ''
        cells = r[1:] if c or (isinstance(r[0], str) and r[0] == '') else r
        b += f'<tr class="{c}">' + ''.join(f'<td>{x}</td>' for x in cells) + '</tr>'
    return (f'<div class="tablewrap"><table class="{cls}"><thead><tr>{h}</tr></thead>'
            f'<tbody>{b}</tbody></table></div>')


def stat(k, v, n, cls=''):
    return f'<div class="stat"><div class="k">{k}</div><div class="v {cls}">{v}</div><div class="n">{n}</div></div>'


# ── deployment table ─────────────────────────────────────────────────────────
deploy = tbl(
    ['series / arm', 'mean |KG|', 'mean nn', 'illegal %', 'train N'],
    [['bad', 'pure symmetry loss (λ1=λ2=0)', '3.0089', '0.01059', '96.9', '49'],
     ['bad', 'λ3 = 0 &mdash; physics term ablated', '1.4712', '0.01671', '95.6', '49'],
     ['bad', 'model9 &mdash; prior work, no symmetry term', '1.278', '0.0178', '96.9', '&mdash;'],
     ['', 'PointNet', '0.3338', '0.01457', '99.4', '49'],
     ['sep', 'raw &mdash; no correction', '0.3331', '0.01457', '99.4', '&mdash;'],
     ['', 'Transport Velocity &mdash; classical baseline', '0.2735', '0.01684', '98.9', '&mdash;'],
     ['', 'GNS @ production training', '0.2417', '0.01864', '86.4', '49'],
     ['', 'GNS @ best training (n=3)', '0.1319&ndash;0.1427', '0.01965', '78.1', '49'],
     ['hl', 'model12 @ production &mdash; <strong>sim-validated</strong>', '0.1269', '0.01947', '82.0', '49'],
     ['hl', 'model12 @ noise 1.0 (n=3)', '0.1230&ndash;0.1273', '0.01950', '80.6', '49'],
     ['best', 'model12 @ N=100 &mdash; best deployer', '0.0871', '0.01957', '77.3', '100'],
     ['best', 'model12 @ N=49, k=12 passes', '0.0846', '0.01966', '80.4', '49'],
     ['best', 'model12 @ N=49, k=40 passes', '0.0675', '0.01965', '88.4', '49']])

# ── architecture benchmark vs deployment ─────────────────────────────────────
arch = tbl(
    ['architecture', 'params', 'N=49 viol_red', 'N=49 |KG|', 'N=49 nn CV', 'N=2500 |KG|'],
    [['best', 'model12', '350,594', '82.9%', '0.0216', '2.3%', '<strong>0.1269</strong>'],
     ['', 'GNS-style (Sanchez-Gonzalez 2020)', '347,966', '62.5% / 91.5%*', '0.0365 / 0.0205*', '3.9%', '0.1319'],
     ['', 'DGCNN', '348,692', '77.1%', '0.0272', '3.1%', 'cannot deploy'],
     ['', 'PointNet++', '350,240', '19.1% / 27.1%*', '0.1065', '23.8%', 'not run'],
     ['bad', 'PointNet', '351,914', '0.2%', '0.2258', '34.9%', '0.3338']])

# ── loss 2x2 ─────────────────────────────────────────────────────────────────
loss = tbl(
    ['loss variant', 'viol_red', '|KG| N=49', 'knn_keep', 'trajectory |KG|', 'seeds'],
    [['best', 'full &mdash; λ1 + λ2 + λ3', '82.9%', '0.0216', '0.661', '<strong>0.127</strong>', '3'],
     ['bad', 'λ3 = 0 &mdash; no physics term', '17.6%', '0.1407', '0.701', '1.471', '4/4 collapse'],
     ['bad', 'λ1 = λ2 = 0 &mdash; pure symmetry', '&minus;19.5%', '0.3948', '0.132', '3.009', '1'],
     ['bad', 'λ2 = 0 &mdash; no displacement reg', '0.0%', '0.2260', '1.000', 'n/a', '1 (collapsed)']])

# ── mechanism ────────────────────────────────────────────────────────────────
mech = tbl(
    ['rung', 'kernel', 'aggregation', 'N=49 |KG|', 'N=2500 |KG|'],
    [['best', '<code>nonorm</code> s2', 'fixed', 'sum', '0.0184', '<strong>0.1177</strong>'],
     ['best', '<code>nonorm</code>', 'fixed', 'sum', '0.0196', '0.1204'],
     ['best', 'model12 @ noise1.0', 'fixed', 'weighted mean', '0.0308', '0.1237'],
     ['best', 'model12 @ production', 'fixed', 'weighted mean', '0.0216', '0.1269'],
     ['sep bad', 'GNS', 'learned latent', 'sum', '0.0205', '0.1319'],
     ['bad', '<code>maxagg</code>', 'inert', 'max', '0.0106', '0.1434'],
     ['bad', '<code>nokernel</code> s2', 'learned gate', 'weighted mean', '0.0244', '0.1486'],
     ['bad', '<code>wmax</code>', 'fixed', 'max', '0.0159', '0.1538'],
     ['bad', '<code>nokernel</code> s3', 'learned gate', 'weighted mean', '0.0252', '0.1703']])

# ── weights ──────────────────────────────────────────────────────────────────
weights = tbl(
    ['checkpoint', 'model config', 'N_train', 'deployment |KG|', 'status'],
    [['best', '<code>model12_sph_n100.pt</code>', '<code>ablations/cardinality/..._n100.yaml</code>',
      '100', '<strong>0.0871</strong>', 'best, n=1, <em>not</em> sim-validated'],
     ['', '<code>model12_sph_l4_noise1p0.pt</code>', '<code>model/model_config_12_sph_L4.yaml</code>',
      '49', '0.1237', 'best N=49 arm, n=3'],
     ['hl', '<code>model12_sph_l4.pt</code>', '<code>model/model_config_12_sph_L4.yaml</code>',
      '49', '0.1269', '<strong>sim-validated</strong> &mdash; quote this']])

# ── k sweep ──────────────────────────────────────────────────────────────────
ksweep = tbl(
    ['k', '|KG|', 'mean nn', 'illegal %', 's / timestep'],
    [['', '0 (raw)', '0.3331', '0.01457', '99.4', '&mdash;'],
     ['bad', '1', '0.4661', '0.01691', '97.1', '0.084'],
     ['', '3', '0.1860', '0.01902', '87.9', '0.090'],
     ['hl', '5 &mdash; shipped, sim-validated', '0.1269', '0.01947', '82.0', '0.110'],
     ['best', '8 &mdash; best illegal %', '0.0984', '0.01963', '<strong>79.2</strong>', '0.143'],
     ['', '12', '0.0846', '0.01966', '80.4', '0.177'],
     ['', '20', '0.0769', '0.01966', '83.6', '0.258'],
     ['', '40', '0.0675', '0.01965', '88.4', '0.439']])

RETRACTIONS = [
    ('DGCNN collapses on hard data',
     'Bimodal across initialisations &mdash; 2 of 4 seeds collapse, 2 train normally (64.8%, 69.2%). '
     'What survives needs no collapse story: DGCNN is simply worse than model12 at noise 0.6.'),
    ('model12 is 3&times; cheaper than GNS',
     'That is dense <em>training</em> cost at N=49. Measured deployment cost at N=2500 is '
     '<span class="now">1.13&times; on CPU, 1.39&times; on CUDA</span> &mdash; real but modest.'),
    ('KG has a floor at &asymp;0.111',
     'An artifact of the k=5 operating point. |KG| falls monotonically to '
     '<span class="now">0.0675 by k=40</span> with no floor. The genuine trade-off is that '
     'illegal% bottoms at k=8 and rises after.'),
    ('Per-particle normalisation drives size transfer',
     'The <code>nonorm</code> rung removes exactly that and transfers <span class="now">better</span> '
     '(0.1204 vs 0.1269). Replaced by the kernel+additive conjunction, which two single-mechanism '
     'rungs pin down.'),
    ('The dense-only baselines cannot process N=2500',
     'True for DGCNN, <span class="now">false for PointNet</span>, which has no pairwise term and runs '
     '2500 points in 6&nbsp;ms. Inferred from the word &ldquo;dense&rdquo; rather than measured.'),
    ('Pure-symmetry loss gives real-data KG 0.357',
     'A July number carried forward uncritically. Re-scored with current periodic tooling it is '
     '<span class="now">3.009</span> &mdash; nine times worse than raw.'),
]

retr_html = ''.join(
    f'<div class="retraction"><div><span class="was">{w}</span></div><div class="muted">{n}</div></div>'
    for w, n in RETRACTIONS)

HTML = f"""<title>SPH corrector — final ablation report</title>
<style>{CSS}</style>
<div class="wrap">

<header>
  <div class="eyebrow">Physics-informed SPH corrector &middot; ablation suite &middot; 2026-08-10</div>
  <h1>What the corrector does, and which claims survived measuring it</h1>
  <p class="lede">73 scored arms across 5 architectures, 4 loss variants, 7 mechanism rungs and
  24 seeded replications. Every architecture trains under one identical recipe with parameter
  counts matched within 0.8&percnt;, so a difference in results is a difference in mechanism.
  Four claims were retracted and two numbers corrected along the way &mdash; those are reported
  here as prominently as the results.</p>
</header>

<section>
  <h2>01 &nbsp;Headline</h2>
  <h3>On the real SPH trajectory, the corrector cuts kernel-gradient asymmetry by 2.6&times;</h3>
  <div class="grid">
    {stat('raw &rarr; corrected', '0.33 &rarr; 0.13', 'mean |KG|, N=2500, disordered regime t&nbsp;&ge;&nbsp;300', 'good')}
    {stat('vs classical baseline', '2.2&times;', 'better than Transport Velocity (0.274)', 'acc')}
    {stat('best available', '0.0871', 'training at N=100 &mdash; 31&percnt; beyond the shipped checkpoint', 'good')}
    {stat('physics term ablated', '1.4712', '4.4&times; <em>worse</em> than no correction at all', 'warn')}
  </div>
  <p>An actual SPH re-simulation launched from corrected start states confirmed the clouds are
  usable restarts. That validation was performed with the production N=49 checkpoint at k=5, so
  the validated number and the best number are different runs &mdash; both are quoted throughout.</p>
  {deploy}
  <p class="muted">Every row scored with one tool (<code>score_arm.py</code>): periodic metrics,
  fixed evaluation clouds, <code>model_best.pt</code>, 15 timesteps at t&nbsp;&ge;&nbsp;300.</p>
</section>

<section>
  <h2>02 &nbsp;Qualitative</h2>
  <h3>Raw shows clumped filaments; the corrector produces a regular lattice</h3>
  <figure>
    <div class="scroll"><img src="{uri('side_by_side_sph_t1000.png')}"
      alt="Raw, Transport Velocity and model12 particle distributions at SPH timestep 1000"></div>
    <figcaption>Timestep 1000, N=2500. Colour is nearest-neighbour distance against the constraint
    rd&nbsp;=&nbsp;0.02; the lower row zooms a 0.35&times;0.35 corner. Spacing uniformity
    (nn&nbsp;CV): raw 18.7&percnt;, Transport Velocity 9.9&percnt;, model12
    <strong>2.8&percnt;</strong> &mdash; the corrector is 3.5&times; more uniform than the
    classical method.</figcaption>
  </figure>
  <figure>
    <div class="scroll"><img src="{uri('side_by_side_n49.png')}"
      alt="One 49-point cloud corrected by model12, GNS, DGCNN and PointNet"></div>
    <figcaption>One synthetic cloud through four architectures, K=5 passes. Grey arrows are the
    displacement field; red rings mark points still closer than rd. PointNet&rsquo;s parallel
    arrows are the tell &mdash; see section 06.</figcaption>
  </figure>
</section>

<section>
  <h2>03 &nbsp;Loss ablation</h2>
  <h3>The physics term is an optimisation enabler, not a regulariser</h3>
  <p>The strongest result in the suite. Ablating the SPH kernel-gradient term does not merely
  degrade quality &mdash; it <strong>reproduces the exact failure that motivated this work</strong>.
  model9, the predecessor whose corrected clouds were unusable as restarts, scored 1.278; model12
  with λ3&nbsp;=&nbsp;0 scores 1.471. The ablation puts the breakage back.</p>
  {loss}
  <div class="callout good">
    <p><strong>Three independent signals, not one.</strong> (1) Removing the term costs 65 points of
    <em>violation</em> reduction &mdash; although the ablated term is the <em>symmetry</em> term.
    (2) Training diverges: deterministic validation loss peaks at iteration 500 and never improves
    across the remaining 9,500, while the full loss improves monotonically to 10,000.
    (3) The final iterate degenerates entirely. Signal (2) does not depend on the collapse, which
    is why this result is robust where other collapse findings were not.</p>
  </div>
  <p><strong>Neither objective alone is usable.</strong> Without the physics term the corrector
  reproduces model9 (1.471). Without the constraint terms it is worse still (3.009, nine times
  raw) and it shreds the arrangement &mdash; <code>knn_keep</code> 0.132 means only 13&percnt; of
  each particle&rsquo;s neighbours survive. The combination is the contribution.</p>
  <p class="muted">Mechanism: the violation term is a mean over N&sup2; pairs of which only O(N)
  violate, so its gradient is diluted ~1/N and shrinks further as easy violations clear, while the
  displacement penalty is undiminished. The KG term supplies a dense per-particle gradient.</p>
</section>

<section>
  <h2>04 &nbsp;Architectures</h2>
  <h3>The small-N benchmark misranks architectures against deployment &mdash; twice</h3>
  <p>All five architectures are parameter-matched and trained with the identical loss, including
  the physics term. So the claim is not &ldquo;physics-informed beats vanilla&rdquo; &mdash; it is
  that given the same physics-informed objective, the architecture decides whether that objective
  is reachable.</p>
  {arch}
  <p class="muted">*two values = noise 0.6&middot;rd / noise 1.0&middot;rd. DGCNN cannot be
  deployed: its kNN graph is an N&times;N matrix rebuilt in feature space every round, so there is
  no fixed edge list to sparsify.</p>
  <figure>
    <div class="scroll"><img src="{uri('benchmark_vs_deployment.png')}"
      alt="Slope chart of architecture rank on the benchmark versus on the real trajectory"></div>
    <figcaption>model12 places 4th of 5 on the synthetic benchmark and 2nd on the real task. Two
    lines cross: GNS and the <code>maxagg</code> rung both beat model12 at N=49 and both lose at
    N=2500.</figcaption>
  </figure>
  <div class="callout">
    <p><strong>Selecting on the benchmark alone would have shipped the worse deployer, twice.</strong>
    This is the suite&rsquo;s most transferable finding, and it emerged from results that initially
    looked like losses.</p>
  </div>
</section>

<section>
  <h2>05 &nbsp;Mechanism</h2>
  <h3>What transfers is the fixed kernel <em>and</em> additive aggregation &mdash; neither alone</h3>
  <p>Isolated by an ablation model whose baseline rung is bit-identical to production model12
  (max&nbsp;|diff|&nbsp;=&nbsp;0.0), with one mechanism swapped at a time.</p>
  {mech}
  <p><strong>Two single-mechanism rungs pin it.</strong> <code>nokernel</code> changes only the
  weighting (fixed kernel &rarr; learned gate) and lands outside the good group. <code>wmax</code>
  changes only the aggregation (weighted mean &rarr; weighted max, kernel retained) and also lands
  outside. Changing either one, and nothing else, is enough.</p>
  <div class="callout good">
    <p>This matches the mathematics: <strong>|KG| is a weighted sum of kernel gradients</strong>, so
    an architecture that reproduces it needs both the distance weighting and the summation. Max
    destroys the sum; a learned gate destroys the weighting. The separation is clean across eleven
    deployment runs (0.1177&ndash;0.1269 versus 0.1319&ndash;0.1703, no overlap).</p>
  </div>
  <p class="muted">This claim was revised twice before reaching this form &mdash; first attributed
  to per-particle normalisation, then to the kernel alone. Each revision came from adding a rung
  that isolated one more thing, which is the argument for single-mechanism ablations over
  multi-way architecture comparisons.</p>
</section>

<section>
  <h2>06 &nbsp;A failure the standard metrics cannot see</h2>
  <h3>Degenerate runs emit a saturated uniform translation</h3>
  <p>Every loss term depends only on <em>relative</em> positions, so a uniform translation is
  invisible to all of them. |KG| and illegal&percnt; both report &ldquo;unchanged&rdquo; while the
  cloud slides a full box length over five passes.</p>
  <figure>
    <div class="scroll"><img src="{uri('displacement_decomposition.png')}"
      alt="Displacement fields decomposed into bulk translation and local restructuring"></div>
    <figcaption>Top: total displacement. Bottom: after removing the bulk component &mdash; what the
    model actually contributes. PointNet&rsquo;s lower panel is empty: its entire output is a
    translation of 0.168, exactly <code>max_displacement</code>, with a local component of 0.0001.
    Working models sit at 3&ndash;9&percnt; bulk.</figcaption>
  </figure>
  <p>It is a tanh saturation trap &mdash; once the output head saturates, the gradient vanishes and
  nothing pulls it back. The trainer now logs <code>bulk_drift</code> every evaluation and warns
  above 50&percnt;. <strong>Roughly half of some architecture variants fall into it depending only
  on initialisation</strong>, which is why the trainer is now seeded by default and why any
  collapse claim needs at least three seeds.</p>
</section>

<section>
  <h2>07 &nbsp;Operating point</h2>
  <h3>The shipped k=5 is not converged, and k=1 is worse than doing nothing</h3>
  {ksweep}
  <figure>
    <div class="scroll"><img src="{uri('k_curve.png')}"
      alt="KG and illegal percentage against number of correction passes"></div>
    <figcaption>A single pass makes the cloud a <em>worse</em> SPH restart than no correction at all
    (0.4661 vs raw 0.3331); the corrector only becomes a net win from k&nbsp;&ge;&nbsp;3.</figcaption>
  </figure>
  <div class="callout">
    <p><strong>The violation-versus-symmetry trade-off lives in k.</strong> |KG| falls monotonically
    with no floor, but illegal&percnt; reaches its minimum at k=8 and climbs thereafter &mdash;
    pushing symmetry harder actively costs constraint satisfaction. Recommended: <strong>k=5</strong>
    for validated claims, <strong>k=8</strong> as the best all-round point, k&nbsp;&ge;&nbsp;12 when
    symmetry is the priority.</p>
  </div>
</section>

<section>
  <h2>08 &nbsp;Retractions and corrections</h2>
  <h3>Four claims withdrawn, two numbers corrected</h3>
  <p>Three of these would have gone into the paper and been wrong. Withdrawing them is the largest
  single contribution to the work&rsquo;s truthfulness, and none of it shows up as a result.</p>
  {retr_html}
  <div class="callout warn">
    <p><strong>Common cause:</strong> a property measured in one regime and asserted in another
    without re-measuring &mdash; training cost quoted as deployment cost, an N=49 metric quoted at
    N=2500, a k=5 artifact quoted as a physical limit. Worth watching for in anything not
    re-derived here.</p>
  </div>
</section>

<section>
  <h2>09 &nbsp;Where the weights are</h2>
  <h3>Three deployable checkpoints, and the best one is not the validated one</h3>
  {weights}
  <div class="callout warn">
    <p>The two 2026-08-10 checkpoints were rescued out of <code>artifacts/training/</code>, which
    this repository documents as &ldquo;delete freely&rdquo;. They would otherwise have been lost.
    Both were trained <strong>unseeded</strong> &mdash; the <code>seed: 0</code> default landed
    later the same day &mdash; so re-running will not reproduce them byte-for-byte.</p>
  </div>
  <p>A checkpoint pairs with exactly one model config. <code>model12_sph_n100.pt</code> uses
  rd&nbsp;0.098, cutoff&nbsp;0.200 and λ3&nbsp;0.070; pairing it with the N=49 config will silently
  produce garbage rather than erroring.</p>
</section>

<section>
  <h2>10 &nbsp;Uncertainty</h2>
  <h3>Two thresholds, because the two regimes differ threefold</h3>
  <p>Three runs of the identical recipe give &sigma;&nbsp;&asymp;&nbsp;9&percnt; on synthetic N=49
  |KG| (0.0207&nbsp;/&nbsp;0.0216&nbsp;/&nbsp;0.0245) but only a <strong>3.5&percnt; range</strong>
  on the trajectory (0.1230&nbsp;/&nbsp;0.1237&nbsp;/&nbsp;0.1273). Deployment averages over 2500
  particles &times; 15 timesteps and washes out the initialisation noise that dominates a
  49-particle benchmark.</p>
  {tbl(['claim', 'n', 'margin', 'verdict'],
       [['best', 'λ3 ablation, synthetic and trajectory', '4 v 3', '6.5&times; / 11&times;', 'solid'],
        ['best', 'model12 vs PointNet', '1 v 3', '10&times;', 'solid'],
        ['best', 'model12 vs GNS @ matched production', '1 v 1', '90&percnt;', 'solid'],
        ['best', 'N=100 beats N=49 at deployment', '1 v 3', '31&percnt;', 'solid'],
        ['best', 'kernel + additive vs everything else', '4 v 7', 'no overlap', 'solid'],
        ['best', 'model12 vs GNS, each at its best', '3 v 3', 'no overlap', 'claimable'],
        ['', 'model12 vs DGCNN, N=49', '1 v 3', '26&percnt;', 'marginal'],
        ['bad', 'DGCNN / knngraph / noperiod collapse', '2-of-3, 2-of-4', '&mdash;', 'not claimable']])}
  <p class="muted">Applying the synthetic threshold to deployment comparisons &mdash; as an earlier
  version of this audit did &mdash; is roughly 3&times; too conservative, and wrongly rejected two
  claims that are in fact separable.</p>
</section>

<section>
  <h2>11 &nbsp;Limitations</h2>
  <h3>What this evidence does not cover</h3>
  <ul>
    <li><strong>One trajectory, one dataset, 2D only.</strong> Every deployment number comes from the
      same SPH run. No number of seeds fixes this, and it is the biggest remaining threat to
      generality.</li>
    <li><strong>λ3&nbsp;=&nbsp;0.27 was tuned on model12</strong> and applied unchanged to all
      baselines. Equal-budget comparison is the stated protocol, but the shared settings are
      model12&rsquo;s settings.</li>
    <li><strong>The best checkpoint is n=1 and unvalidated.</strong> N=100 scores 0.0871 but has
      never been through a solver; only the N=49 checkpoint at k=5 has.</li>
    <li><strong>Three bridge rungs are unreadable</strong> &mdash; they collapsed at n=1, and
      collapse here is initialisation-dependent. A complete bridge needs ~3 seeds per rung.</li>
    <li><strong>The <code>maxagg</code> rung changes two mechanisms</strong> (max, and weighting
      removed). <code>wmax</code> was added as the clean single-change counterpart.</li>
    <li><strong>Not attempted:</strong> 3D (needs a 3D quintic kernel), N=196, the full packing
      grid, per-architecture tuning, re-simulation at k&gt;5.</li>
  </ul>
</section>

<footer>
  Full run log and claim-by-claim history: <code>paper/JOURNAL.md</code> &middot;
  consolidated results: <code>paper/RESULTS.md</code> &middot;
  architecture reference: <code>paper/ARCHITECTURES.md</code> &middot;
  every number traces to a row in <code>paper/results.csv</code> (73 scored arms) &middot;
  figures: <code>vibecoding/visualizations/</code> &middot;
  guards: <code>tests/test_wholecloud.py</code> (bit-exact) and
  <code>tests/test_sparse_paths.py</code> (18 checks, 6 architectures)
</footer>
</div>
"""

OUT.write_text(HTML, encoding='utf-8')
print(f'-> {OUT}  ({OUT.stat().st_size / 1e6:.2f} MB)')
