"""Build the experiments-section draft: paper-ready prose, tables, figures,
with MAIN / APPENDIX / CUT-OK flags on every block. ASCII-only source; all
special characters are HTML entities."""
import base64
from pathlib import Path

ROOT = Path(r'e:\VSCode\CorrectorWorkshop')
FIGS = ROOT / 'vibecoding/visualizations/outputs'
OUT = Path(r'C:\Users\ioand\AppData\Local\Temp\claude\e--VSCode-CorrectorWorkshop'
           r'\d723bad1-170c-4fd5-a8d3-892d03187a81\scratchpad\experiments_section.html')

L1, L2, L3 = '&lambda;<sub>1</sub>', '&lambda;<sub>2</sub>', '&lambda;<sub>3</sub>'
BIB = ((ROOT / 'paper/references.bib').read_text(encoding='utf-8')
       .replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def uri(name):
    return 'data:image/png;base64,' + base64.b64encode((FIGS / name).read_bytes()).decode()


CSS = """
:root{
  --bg:#fdfdfb; --surface:#f4f4ef; --line:#ddddd2; --line2:#ebebe2;
  --ink:#1a1c1a; --muted:#5c635c;
  --accent:#31567c; --good:#2c6e4f; --warn:#9c4a42; --appx:#7a5e1e;
}
@media (prefers-color-scheme: dark){
  :root{ --bg:#101311; --surface:#181c19; --line:#2b302c; --line2:#20241f;
         --ink:#e8eae6; --muted:#909a90;
         --accent:#7aa7cc; --good:#5cba8c; --warn:#dd8378; --appx:#cfae5e; }
}
:root[data-theme="dark"]{ --bg:#101311; --surface:#181c19; --line:#2b302c; --line2:#20241f;
  --ink:#e8eae6; --muted:#909a90; --accent:#7aa7cc; --good:#5cba8c; --warn:#dd8378; --appx:#cfae5e; }
:root[data-theme="light"]{ --bg:#fdfdfb; --surface:#f4f4ef; --line:#ddddd2; --line2:#ebebe2;
  --ink:#1a1c1a; --muted:#5c635c; --accent:#31567c; --good:#2c6e4f; --warn:#9c4a42; --appx:#7a5e1e; }

*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:Charter,"Bitstream Charter","Sitka Text",Cambria,Georgia,serif;
  font-size:17px;line-height:1.68}
.mono,code{font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace}
code{font-size:.82em;background:var(--surface);padding:.08em .3em;border-radius:3px}
.wrap{max-width:960px;margin:0 auto;padding:56px 28px 120px}
header{border-bottom:2px solid var(--ink);padding-bottom:24px;margin-bottom:34px}
.eyebrow{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;
  letter-spacing:.15em;text-transform:uppercase;color:var(--accent);margin-bottom:10px}
h1{margin:0 0 10px;font-size:clamp(26px,3.6vw,36px);line-height:1.16;font-weight:700;
  letter-spacing:-.01em;text-wrap:balance}
.lede{margin:0;color:var(--muted);max-width:66ch;font-size:15.5px;
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
h2{margin:44px 0 4px;font-size:22px;font-weight:700;letter-spacing:-.008em}
h2 .num{color:var(--accent);margin-right:10px}
h3{margin:26px 0 6px;font-size:17.5px;font-weight:700;font-style:italic}
p{margin:0 0 15px;max-width:70ch;text-align:justify;hyphens:auto}
ul,ol{margin:0 0 15px;padding-left:22px;max-width:68ch}
li{margin-bottom:5px}
.muted{color:var(--muted)}
em.term{font-style:italic}
.chips{margin:2px 0 14px}
.chip{display:inline-block;font-family:ui-monospace,Menlo,Consolas,monospace;
  font-size:10px;letter-spacing:.1em;text-transform:uppercase;font-weight:600;
  padding:2px 9px;border-radius:99px;margin-right:6px;vertical-align:2px}
.chip.main{background:color-mix(in srgb,var(--good) 14%,transparent);color:var(--good);
  border:1px solid color-mix(in srgb,var(--good) 45%,transparent)}
.chip.appx{background:color-mix(in srgb,var(--appx) 14%,transparent);color:var(--appx);
  border:1px solid color-mix(in srgb,var(--appx) 45%,transparent)}
.chip.cut{background:color-mix(in srgb,var(--warn) 12%,transparent);color:var(--warn);
  border:1px solid color-mix(in srgb,var(--warn) 40%,transparent)}
.flagnote{font-size:13px;color:var(--muted);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:-8px 0 14px}
.eq{background:var(--surface);border:1px solid var(--line2);border-radius:4px;
  padding:12px 18px;margin:14px 0 18px;text-align:center;font-size:16.5px;overflow-x:auto}
.eq .where{display:block;font-size:13.5px;color:var(--muted);margin-top:6px;text-align:center}
.tablewrap{overflow-x:auto;margin:6px 0 6px}
table{border-collapse:collapse;width:100%;font-size:14px;min-width:540px;
  font-variant-numeric:tabular-nums;
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
th,td{padding:7px 12px;text-align:right;border-bottom:1px solid var(--line2);white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal}
thead th{font-size:12px;color:var(--muted);font-weight:600;border-top:2px solid var(--ink);
  border-bottom:1px solid var(--ink)}
tbody tr:last-child td{border-bottom:2px solid var(--ink)}
tbody td:not(:first-child){font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px}
tr.best td{color:var(--good);font-weight:600}
tr.bad td{color:var(--warn)}
tr.hl{background:color-mix(in srgb,var(--accent) 7%,transparent)}
tr.grp td{border-top:2px solid var(--line)}
.caption{font-size:14px;color:var(--muted);margin:8px 0 26px;max-width:74ch;
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.caption b{color:var(--ink)}
figure{margin:20px 0 4px}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:4px;background:#fff}
.scroll img{display:block;width:100%;min-width:700px;height:auto}
.note{border-left:3px solid var(--accent);background:var(--surface);
  padding:12px 17px;margin:16px 0 20px;border-radius:0 4px 4px 0;max-width:72ch;
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:15px}
.note.warn{border-left-color:var(--warn)}
.note p{text-align:left;hyphens:none;margin-bottom:8px}
.note p:last-child{margin-bottom:0}
.appendix-rule{margin:60px 0 8px;border:none;border-top:3px double var(--ink)}
footer{margin-top:56px;padding-top:20px;border-top:1px solid var(--line);
  color:var(--muted);font-size:13px;
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
"""


def tbl(headers, rows):
    h = ''.join(f'<th>{c}</th>' for c in headers)
    b = ''
    for r in rows:
        cls, cells = (r[0], r[1:]) if r and r[0] in ('best', 'bad', 'hl', 'grp', 'grp bad', '') else ('', r)
        b += f'<tr class="{cls}">' + ''.join(f'<td>{x}</td>' for x in cells) + '</tr>'
    return f'<div class="tablewrap"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def chips(*cs):
    m = {'main': 'KEEP &mdash; MAIN BODY', 'appx': 'MOVE TO APPENDIX', 'cut': 'CUT IF TIGHT'}
    return '<div class="chips">' + ''.join(f'<span class="chip {c}">{m[c]}</span>' for c in cs) + '</div>'


# ------------------------------------------------------------------ tables --
T_DEPLOY = tbl(
    ['method', 'mean |KG| &darr;', 'mean nn', 'illegal %', 'runs'],
    [['', 'uncorrected', '0.326', '0.0146', '99.5', '&mdash;'],
     ['bad', 'model9 (prior work)', '1.278', '0.0178', '96.9', '&mdash;'],
     ['', 'Transport Velocity', '0.274', '0.0168', '98.9', '&mdash;'],
     ['grp', 'PointNet', '0.334', '0.0146', '99.4', '1'],
     ['', 'GNS (matched production training)', '0.242', '0.0186', '86.4', '1'],
     ['', 'GNS (best training regime)', '0.132&ndash;0.143', '0.0197', '78.1', '3'],
     ['hl', 'model12 (production, <b>sim-validated</b>)', '0.127', '0.0195', '82.0', '1'],
     ['hl', 'model12 (best N=49 regime)', '0.123&ndash;0.127', '0.0195', '80.6', '3'],
     ['best', 'model12 trained at N=100', '<b>0.087</b>', '0.0196', '77.3', '1'],
     ['best', 'model12, k=12 passes', '0.085', '0.0197', '80.4', '1']])

T_OBST = tbl(
    ['', 'mean nn', 'nn CV', 'illegal %', 'knn-keep', 'bulk drift'],
    [['', 'initial', '0.0076', '35.6%', '96.1', '&mdash;', '&mdash;'],
     ['best', 'corrected (k=5)', '0.0117', '<b>3.0%</b>', '85.1', '0.64', '5.8%']])

T_LOSS = tbl(
    ['loss variant', 'viol. red.', '|KG| (N=49)', 'knn-keep', '|KG| (trajectory)', 'seeds'],
    [['best', f'full: {L1}+{L2}+{L3}', '82.9%', '0.0216', '0.661', '<b>0.127</b>', '3'],
     ['bad', f'{L3}=0 (physics term removed)', '17.6%', '0.1407', '0.701', '1.471', '4/4 degen.'],
     ['bad', f'{L1}={L2}=0 (symmetry only)', '&minus;19.5%', '0.3948', '0.132', '3.009', '1'],
     ['bad', f'{L2}=0 (no displacement reg.)', '0.0%', '0.2260', '1.000', '&mdash;', '1 (degen.)']])

T_ARCH = tbl(
    ['architecture', 'params', 'viol. red. (&sigma;=0.6/1.0&middot;r<sub>d</sub>)', '|KG| (0.6/1.0)', 'nn CV', 'deploys?'],
    [['best', 'model12 (ours)', '350,594', '82.9% / 82.0%', '0.0216 / 0.0308', '2.3%', 'yes'],
     ['', 'GNS-style', '347,966', '62.5% / <b>91.5%</b>', '0.0365 / <b>0.0205</b>', '3.9%', 'yes'],
     ['', 'DGCNN', '348,692', '77.1% / 66&ndash;73%*', '0.0272 / 0.036&ndash;0.041*', '3.1%', 'no'],
     ['', 'PointNet++', '350,240', '19.1% / 27.1%', '0.1065 / 0.1249', '23.8%', 'yes'],
     ['bad', 'PointNet', '351,914', '0.2% / 0.1%', '0.2258 / 0.3502', '34.9%', 'yes (inert)']])

T_MECH = tbl(
    ['variant', 'edge weighting', 'aggregation', '|KG| N=49', '|KG| N=2500'],
    [['best', 'nonorm', 'fixed kernel', 'sum', '0.0184&ndash;0.0196', '0.1177&ndash;0.1204'],
     ['best', 'model12', 'fixed kernel', 'weighted mean', '0.0216&ndash;0.0308', '0.1230&ndash;0.1273'],
     ['grp bad', 'GNS', 'learned (edge latent)', 'sum', '0.0198&ndash;0.0209', '0.1319&ndash;0.1427'],
     ['bad', 'nokernel', 'learned (scalar gate)', 'weighted mean', '0.0244&ndash;0.0252', '0.1486&ndash;0.1703'],
     ['bad', 'maxagg', '&mdash; (inert)', 'max', '0.0106&ndash;0.0114', '0.1434&ndash;0.1495'],
     ['bad', 'wmax', 'fixed kernel', 'weighted max', '0.0159', '0.1538']])

T_K = tbl(
    ['passes k', '|KG| &darr;', 'illegal % &darr;', 's / step (CPU)'],
    [['', '0 (raw)', '0.333', '99.4', '&mdash;'],
     ['bad', '1', '0.466', '97.1', '0.084'],
     ['', '3', '0.186', '87.9', '0.090'],
     ['hl', '5 (sim-validated)', '0.127', '82.0', '0.110'],
     ['best', '8', '0.098', '<b>79.2</b>', '0.143'],
     ['', '12', '0.085', '80.4', '0.177'],
     ['', '40', '<b>0.068</b>', '88.4', '0.439']])

TA1_AUDIT = tbl(
    ['claim', 'n', 'margin vs threshold', 'verdict'],
    [['best', f'{L3} ablation (synthetic / trajectory)', '4 v 3', '6.5&times; / 11&times;', 'solid'],
     ['best', 'model12 &gt; GNS at matched production training', '1 v 1', '90%', 'solid'],
     ['best', 'N=100 &gt; N=49 at deployment', '1 v 3', '31%', 'solid'],
     ['best', 'kernel+additive vs rest (deployment)', '4 v 7', 'no overlap', 'solid'],
     ['best', 'model12 &gt; GNS, each at its best (deployment)', '3 v 3', 'no overlap', 'claimable'],
     ['', 'model12 &gt; DGCNN at N=49', '1 v 3', '26% vs 18%', 'marginal'],
     ['bad', 'DGCNN / kNN-graph / no-PBC "collapse"', '2-of-3/4', '&mdash;', 'not claimable']])

TA2_SCALE = tbl(
    ['N', f'{L3} (&prop;1/N&sup2;)', 'KG min-image truncation error'],
    [['bad', '16', '2.53', '295%'],
     ['bad', '25', '1.04', '39%'],
     ['', '49 (training)', '0.27', '1.3%'],
     ['', '100', '0.070', '0.1%'],
     ['', '196', '0.019', '0% (support &lt; box/2)']])

TA3_COST = tbl(
    ['', 'model12', 'GNS', 'DGCNN', 'PointNet'],
    [['', 'training, N=49 dense (s/iter)', '0.073', '0.223', '0.047', '0.025'],
     ['', 'inference, N=2500 sparse, k=5 (s/step, CPU)', '0.189', '0.213', 'n/a', '~0.03'],
     ['', 'inference, CUDA', '0.049', '0.068', 'n/a', '&mdash;']])

# ------------------------------------------------------------------- build --
HTML = f"""<title>Experiments section &mdash; draft for transfer</title>
<style>{CSS}</style>
<div class="wrap">

<header>
  <div class="eyebrow">Complete single-file paper package &middot; &sect;1&ndash;&sect;5 + appendix A1&ndash;A8 &middot; numbers frozen 2026-08-10, commit 82f194a</div>
  <h1>Learned Correction of SPH Particle Distributions</h1>
  <p class="lede">Everything paper-related in one self-contained file: all five sections
  in paper voice, the full appendix, every figure (embedded), the verified BibTeX, and
  the asset/checkpoint locations. Every block carries a flag &mdash; KEEP, APPENDIX, or
  CUT&nbsp;IF&nbsp;TIGHT &mdash; and every number traces to
  <code>paper/results.csv</code> (73 scored arms), the 702-step KG sweep, or the scored
  obstruction run. Working title above is a placeholder. Page budget at the KEEP flags:
  &sect;1 0.6 + &sect;2 0.5 + &sect;3 0.9 + &sect;4 &asymp;2.5 + &sect;5 0.4
  &asymp; <b>4.9 pages</b>.</p>
</header>

<!-- INTRODUCTION -->
<h2><span class="num">1</span>Introduction</h2>
{chips('main')}
<p class="flagnote">Flag: draft prose, &asymp;0.6 page. The contribution list is the part
to keep verbatim; the opening can be reworded to taste.</p>

<p>Smoothed particle hydrodynamics discretises a fluid by particles whose interactions
are weighted by a smoothing kernel, and its accuracy rests on a property the equations
never enforce explicitly: that each particle's neighbourhood is spatially balanced. On a
disordered distribution the discrete kernel-gradient sum
&Sigma;<sub>j</sub>&nabla;W<sub>ij</sub>V<sub>j</sub> &mdash; identically zero on a
symmetric arrangement &mdash; acquires large residuals, and states that violate minimum
particle spacing cannot be used to start or restart a simulation. Classical remedies
(particle shifting, transport-velocity formulations, packing algorithms; &sect;2) repair
this inside the solver loop, iteratively and at every step.</p>

<p>We ask whether the repair can be <em>learned once</em> and applied as a standalone
operator. The question is sharper than it looks: a corrector of this kind trained only on
the spacing constraint &mdash; the natural first attempt, and our own predecessor model
&mdash; produces clouds that satisfy the constraint while <em>quadrupling</em> the
kernel-gradient residual (1.278 against 0.326 uncorrected), making its outputs useless as
restart states. Fixing the geometry is not the same as fixing the physics.</p>

<p>This paper shows that a compact message-passing network can do both, provided the
physics enters twice: as a kernel-gradient residual in the loss, and as a fixed
SPH-kernel-shaped weighting in the architecture. Our contributions:</p>

<ul>
  <li><b>A deployable physics-informed corrector.</b> Trained on 49-particle synthetic
  lattices, it corrects a real 2,500-particle SPH trajectory to mean |KG| 0.127 &mdash;
  2.2&times; better than the Transport Velocity baseline &mdash; at 0.19&thinsp;s per
  state on CPU, validated by an actual SPH re-simulation, and transfers unchanged to a
  bounded obstacle scene at 11.7&times; coordinate scale (&sect;4.2).</li>
  <li><b>The physics term is an optimisation enabler, not a regulariser.</b> Ablating it
  does not merely degrade quality: training diverges, and the deployed model reproduces
  the predecessor's failure (|KG| 1.471). Neither the constraint terms nor the physics
  term function alone (&sect;4.3).</li>
  <li><b>Small-N benchmarks misrank architectures against deployment.</b> Two
  parameter-matched competitors beat our model on the training-scale benchmark and lose
  on the real task (&sect;4.4); single-mechanism ablations locate what transfers &mdash;
  the fixed kernel weighting <em>with</em> additive aggregation (&sect;4.5).</li>
  <li><b>Deployment guidance with measured uncertainty.</b> An operating-point sweep, a
  degenerate mode invisible to the task metrics, and a claim audit against measured seed
  variance (&sect;4.6&ndash;4.8, appendix).</li>
</ul>

<!-- RELATED WORK -->
<h2><span class="num">2</span>Related work</h2>
{chips('main')}
<p class="flagnote">Flag: keep &mdash; sized for ~0.5 page. Citation keys refer to
<code>paper/references.bib</code> (15 entries, venues verified 2026-08-10); replace with
your \\cite{{}} commands. Full positioning notes: <code>paper/RELATED_WORK.md</code>.</p>

<p><em class="term">Particle regularity in SPH.</em> SPH accuracy degrades on disordered
particle distributions, and a family of classical remedies exists: particle shifting
moves particles down concentration gradients [xu2009accuracy, lind2012incompressible],
the transport-velocity formulation regularises positions inside the momentum equation
[adami2013transport], and packing algorithms iterate a damped dynamics to prepare initial
conditions [colagrossi2012particle, diehl2015generating]. All of these are
solver-coupled: they run as iterative physics loops inside, or ahead of, the simulation.
Our corrector addresses the same defect &mdash; a distribution violating minimum spacing
(equivalently, a blue-noise condition [bridson2007fast]) with asymmetric kernel-gradient
sums [morris1997modeling] &mdash; but as a learned, standalone operator applied to
arbitrary states with no solver in the loop.</p>

<p><em class="term">Learned particle simulation.</em> Graph-network simulators
[sanchezgonzalez2020learning, pfaff2021learning] learn dynamics rollouts over particle or
mesh states. Neural SPH [toshev2024neural] shows that such rollouts suffer
tensile-instability-like particle clustering and repairs them by inserting SPH relaxation
steps at inference; diffSPH [winchenbach2025diffsph] casts shifting itself as an
optimisation over differentiable SPH operators. Our work sits between these: rather than
adding physics relaxation to a learned simulator, we <em>learn the relaxation itself</em>
&mdash; and our architecture is precisely a GNS-class network with its learned edge
weighting replaced by the fixed SPH kernel it would otherwise have to approximate
(&sect;4.5).</p>

<p><em class="term">Physics-informed losses.</em> Embedding physical residuals in
training objectives is standard since PINNs [raissi2019physics]. Our kernel-gradient term
belongs to this family, but our ablation sharpens the usual claim: the term is not a soft
constraint that improves accuracy &mdash; without it, training diverges and the corrector
reproduces the failure mode of its predecessor (&sect;4.3). The physics is an
optimisation enabler.</p>

<p><em class="term">Point-cloud architectures.</em> PointNet and its hierarchical and
graph-based successors [qi2017pointnet, qi2017pointnetpp, wang2019dynamic] are the
standard learned operators on unordered point sets. We use them as parameter-matched
baselines and find their generic inductive biases &mdash; global pooling, feature-space
graphs, max aggregation &mdash; fail this task (&sect;4.4), and that small-N benchmark
rankings of them invert at deployment scale.</p>

<p>Finally, while we instantiate the recipe for SPH, nothing in it is SPH-specific: the
objective pairs a minimum-spacing constraint with the <em>target method's</em> own
consistency residual, and the architecture builds the corresponding interaction kernel
into its message weighting. Substituting the residual &mdash; e.g. an energy-based term
for molecular-dynamics initial conditions, in place of the kernel-gradient term &mdash;
yields a corrector for other particle methods; we return to this in the outlook.</p>

<div class="note"><p><b>Reviewer-facing positioning</b> (keep in mind, not in the paper):
Neural SPH keeps the learned simulator and adds classical relaxation; we learn the
relaxation. If asked &ldquo;why not just run classical relaxation?&rdquo; &mdash; the
deployment table answers it: Transport Velocity reaches 0.274, the learned corrector
0.127.</p></div>

<!-- METHODS -->
<h2><span class="num">3</span>Method</h2>
{chips('main')}
<p class="flagnote">Flag: keep, &asymp;0.9 page with one equation block. If your Methods
section also covers the objective, move the loss equation from &sect;4.1 here.</p>

<p><em class="term">Corrector network.</em> The corrector is an L-round message-passing
network over the radius graph of the cloud. Node states start at zero, so all signal
enters through pairwise geometry; one round computes, for every ordered pair within the
cutoff radius r<sub>c</sub>,</p>

<div class="eq">
  e<sub>ij</sub><sup>(l)</sup> = MLP<sub>e</sub><sup>(l)</sup> [h<sub>i</sub>, h<sub>j</sub>,
  x<sub>i</sub>&minus;x<sub>j</sub>, d<sub>ij</sub>, relu(r<sub>d</sub>&minus;d<sub>ij</sub>)],
  &emsp;
  h<sub>i</sub> &larr; h<sub>i</sub> + MLP<sub>n</sub><sup>(l)</sup> &Sigma;<sub>j</sub>
  w<sub>ij</sub>&thinsp;e<sub>ij</sub><sup>(l)</sup>,
  &emsp;
  w<sub>ij</sub> = <span style="white-space:nowrap">(1&minus;(d<sub>ij</sub>/r<sub>c</sub>)&sup2;)&sup2;
  / &Sigma;<sub>k</sub>(&middot;)</span>
  <span class="where">after L rounds, &Delta;x<sub>i</sub> =
  tanh(MLP<sub>o</sub>(h<sub>i</sub>))&thinsp;&middot;&thinsp;1.2&thinsp;r<sub>d</sub>.
  All relative positions use the minimum image on periodic domains.</span>
</div>

<p>Two design choices carry the results. First, the message weight w<sub>ij</sub> is a
<em>fixed, SPH-kernel-shaped function of distance</em>, not learned attention: it is
smooth, compactly supported (identically zero at d&thinsp;&ge;&thinsp;r<sub>c</sub>), and
scale-free once coordinates are normalised by r<sub>d</sub>. Every near pair contributes
&mdash; violating or not &mdash; which is what makes the kernel-gradient signal visible;
overlap depth remains explicit through the relu(r<sub>d</sub>&minus;d) feature.
&sect;4.5 shows this weighting, together with additive aggregation, is precisely the
component that survives the transfer from training to deployment scale. Second, the
receptive field is matched to the physics: with r<sub>c</sub>&thinsp;=&thinsp;2&Delta;x
(the SPH smoothing length h) and L=4 rounds, information propagates
&asymp;8 particle spacings &mdash; covering the quintic kernel's support of 6&Delta;x
&mdash; so kernel-gradient symmetry is learnable as a <em>local</em> property, which is
what permits size generalisation. The network is translation-invariant by construction
(it sees only relative positions) but deliberately not rotation-equivariant, matching a
fixed simulation frame.</p>

<p><em class="term">Whole-cloud deployment.</em> Training uses dense
N&times;N interactions at small N; deployment runs the identical weights over an
explicit edge list (exact, since w<sub>ij</sub> vanishes beyond r<sub>c</sub>). One
correction pass scales the cloud by r<sub>d</sub><sup>train</sup>/r<sub>d</sub><sup>test</sup>
into the training geometry, builds the periodic edge list with a cell-indexed
neighbour search, applies one forward pass, and unscales; the corrector is applied for k
such passes. Solid boundaries are represented by ghost particles filling the obstacle
interior at spacing r<sub>d</sub>, concatenated before each pass and re-pinned after it,
so the solid never drifts (&sect;4.2).</p>

<!-- SETUP -->
<h2><span class="num">4.1</span>Setup</h2>
{chips('main')}

<p><em class="term">Task.</em> Given a 2-D point cloud
<i>x</i>&thinsp;&isin;&thinsp;&#8477;<sup>N&times;2</sup> on a periodic domain that violates a
minimum pairwise distance <i>r<sub>d</sub></i>, predict per-point displacements
&Delta;<i>x</i> such that <i>x</i>+&Delta;<i>x</i> satisfies the constraint and is usable
as a restart state for an SPH simulation. Restart quality is measured by the discrete SPH
consistency condition</p>

<div class="eq">
  KG<sub>i</sub> = &Sigma;<sub>j</sub> &nabla;W(r<sub>ij</sub>, h)&thinsp;V<sub>j</sub> &nbsp;&rarr;&nbsp; 0
  <span class="where">W: quintic spline, h = 2&Delta;x; KG<sub>i</sub> vanishes on a symmetric
  neighbourhood, and mean&thinsp;|KG<sub>i</sub>| is our primary metric (lower is better).</span>
</div>

<p><em class="term">Training data.</em> Clouds are generated online per batch on the unit
torus: a randomly translated 7&times;7 lattice (N=49, spacing 1/7,
<i>r<sub>d</sub></i>&thinsp;=&thinsp;0.14&thinsp;&asymp;&thinsp;0.98&middot;spacing) with
bounded jitter, perturbed by Gaussian noise of amplitude drawn from
[0,&thinsp;&sigma;&middot;<i>r<sub>d</sub></i>] with &sigma;=0.6 (standard) or &sigma;=1.0
(hard). All geometry is minimum-image periodic.</p>

<p><em class="term">Loss.</em> All models are trained with the same physics-informed
objective</p>

<div class="eq">
  L = {L1}&thinsp;mean&thinsp;relu(r<sub>d</sub>&minus;d<sub>ij</sub>)
  &nbsp;+&nbsp; {L2}&thinsp;mean&thinsp;|&Delta;x|
  &nbsp;+&nbsp; {L3}&thinsp;mean<sub>i</sub>&thinsp;|KG<sub>i</sub>|&sup2;
  <span class="where">{L1}=1/r<sub>d</sub>=7.14,&ensp;
  {L2}=0.1&thinsp;{L1}/(N&minus;1)=0.0149,&ensp;
  {L3}=0.27. The KG term is soft and training-only &mdash; never enforced at inference.</span>
</div>

<p><em class="term">Protocol.</em> Every architecture shares one recipe: AdamW
(lr&thinsp;10<sup>&minus;3</sup>, wd&thinsp;10<sup>&minus;4</sup>), cosine schedule over
10<sup>4</sup> iterations, batch 32, K=5 unrolled correction passes with per-step
detachment, and an identical bounded output head
tanh(&middot;)&times;1.2&thinsp;r<sub>d</sub>. Parameter counts are matched to
350&thinsp;k&thinsp;&plusmn;0.8%. No per-architecture hyperparameter search is performed
(discussed in &sect;4.8). Because training can degenerate initialisation-dependently
(&sect;4.7), any degeneration claim requires &ge;3 seeds.</p>

<p><em class="term">Evaluation.</em> One scorer evaluates every arm: periodic metrics on
64 fixed held-out clouds (synthetic regime), and on a real N=2500 SPH trajectory
(1002 timesteps; disordered regime t&thinsp;&ge;&thinsp;300) through a whole-cloud sparse
corrector with coordinate scaling r<sub>d</sub><sup>train</sup>/r<sub>d</sub><sup>test</sup>.
Sparse and dense forward passes agree to 4&times;10<sup>&minus;7</sup> (18 checks across
all architectures). Corrected states from the production configuration were additionally
validated by an actual SPH re-simulation. Measurement spread over three identical-recipe
runs is &sigma;&asymp;9% on synthetic |KG| but only 3.5% on trajectory |KG| (deployment
averages over 2500 particles &times; 15 timesteps); differences below these thresholds
are not claimed. Deployment costs 0.189&thinsp;s/timestep on CPU and 0.049&thinsp;s on
GPU at N=2500, k=5.</p>

<div class="note"><p><b>Flag:</b> the two equations and the uncertainty sentence are the
parts reviewers will look for &mdash; keep verbatim. The cost sentence can move to the
appendix (Table&nbsp;A3) if space is tight.</p></div>

<!-- MAIN RESULT -->
<h2><span class="num">4.2</span>Correction of a real SPH trajectory</h2>
{chips('main')}

<p>Table&nbsp;1 reports deployment on the held-out trajectory. The corrector reduces mean
|KG| from 0.326 to <b>0.127</b> &mdash; 2.6&times; below the uncorrected state and
2.2&times; below Transport Velocity, the classical particle-shifting baseline &mdash;
while raising mean neighbour spacing to 0.0195 (r<sub>d</sub><sup>test</sup>=0.02) and
reducing the spread of that spacing (nn&nbsp;CV) from 18.7% to 2.8%, i.e. a visibly
regular near-lattice arrangement (Fig.&nbsp;1). Prior work without the symmetry term
(model9) <em>quadruples</em> |KG| instead; we return to this in &sect;4.3.</p>

{T_DEPLOY}
<p class="caption"><b>Table 1:</b> Deployment on the SPH trajectory (N=2500, disordered
regime t&thinsp;&ge;&thinsp;300; mean over timesteps). Uncorrected, Transport Velocity and
model9 rows are from the full 702-step sweep; corrector rows use the 15-step scorer
protocol, which reads the same checkpoint within 1% (0.128 vs 0.127). Ranges are min&ndash;max
over 3 training runs. <b>The sim-validated configuration and the best configuration are
different runs</b>: only the production N=49 checkpoint at k=5 has been through a solver;
the N=100 checkpoint (0.087, single run) and the k=12 operating point are better on the
metric but unvalidated.</p>

<figure>
  <div class="scroll"><img src="{uri('side_by_side_sph_t1000.png')}"
    alt="Raw, Transport Velocity, and corrected particle distributions at timestep 1000"></div>
</figure>
<p class="caption"><b>Figure 1:</b> Timestep 1000 of the SPH trajectory (full domain and
0.35&times;0.35 zoom; colour = nearest-neighbour distance against
r<sub>d</sub>=0.02). The uncorrected state exhibits clumped filaments; Transport Velocity
partially relaxes them; the corrector produces uniform spacing at the constraint distance
(nn&nbsp;CV 18.7% &rarr; 9.9% &rarr; 2.8%).</p>

<h3>Generality: a bounded scene with an obstacle</h3>
{chips('main')}
<p class="flagnote">Flag: keep &mdash; six lines plus a small table, and it is the answer
to the single-scenario objection.</p>

<p>To test transfer beyond the periodic trajectory we deploy the same production
checkpoint, without retraining or retuning, on a qualitatively different scenario:
initialising 6,100 particles around a gear-shaped obstacle in a bounded, non-periodic
domain at r<sub>d</sub>=0.012 (coordinate scale 11.7&times; the training
r<sub>d</sub>, versus 7.0&times; on the trajectory), with the obstacle interior filled by
832 fixed ghost particles re-pinned at every pass. The kernel-gradient metric is not
defined here (it presumes a periodic domain), so we report the structural metrics. The
corrector produces the same signature as on the trajectory &mdash; near-uniform spacing
(nn&nbsp;CV 3.0% versus 2.8%), local rather than bulk motion (drift 5.8%), and comparable
neighbourhood rewiring (knn-keep 0.64) &mdash; despite the different topology, boundary
conditions, density, and 2.4&times; larger cloud.</p>

{T_OBST}
<p class="caption"><b>Table 2:</b> Bounded obstacle scene (6,100 real + 832 ghost
particles, r<sub>d</sub>=0.012, k=5). Same checkpoint as Table&nbsp;1. Residual illegal%
remains high because the scene is initialised with grid spacing <em>at</em>
r<sub>d</sub> (96% illegal), but spacing regularity matches the periodic result. No
solver validation exists for this scenario.</p>

<!-- LOSS -->
<h2><span class="num">4.3</span>Loss ablation: the physics term is an optimisation enabler</h2>
{chips('main')}

<p>Table&nbsp;3 ablates each term of the objective. Removing the KG term
({L3}=0) does not merely degrade quality &mdash; <b>it reproduces the failure
mode of prior work</b>. Deployed on the trajectory, the ablated model yields |KG| 1.471,
4.4&times; <em>worse than applying no correction at all</em>, and adjacent to model9
(1.278), the predecessor whose corrected clouds were unusable as restarts and whose
failure motivated this work. The ablation puts the breakage back.</p>

{T_LOSS}
<p class="caption"><b>Table 3:</b> Loss ablation. All cells trained with the shared recipe
and scored identically; trajectory column as in Table&nbsp;1. <em>knn-keep</em> is the
mean Jaccard overlap of each particle's 6-NN set before/after correction. The
{L3}=0 arm degenerates in 4 of 4 initialisations; the symmetry-only arm
destroys the arrangement it is given (knn-keep 0.132). Neither objective functions alone.</p>

<p>Three independent signals support the {L3} result. First, removing the
<em>symmetry</em> term costs 65 points of <em>violation</em> reduction &mdash; the term it
was supposedly only regularising. Second, training diverges without it: deterministic
validation loss reaches its minimum at iteration 500 of 10,000 and never improves, whereas
the full objective improves monotonically to the final iterate. Third, the final iterate
degenerates into a saturated uniform translation (&sect;4.7). We attribute this to
gradient density: the violation term is a mean over N&sup2; pairs of which only O(N)
violate, so its gradient dilutes as easy violations clear, while the KG term supplies a
dense per-particle gradient throughout training.</p>

<div class="note"><p><b>Flag:</b> keep the whole subsection &mdash; this is the paper's
central claim. If a sentence must go, cut the third signal (the uniform-translation
detail) and let &sect;4.7 carry it.</p></div>

<!-- ARCHITECTURES -->
<h2><span class="num">4.4</span>Architecture comparison: the benchmark misranks</h2>
{chips('main')}

<p>We compare against four standard point-cloud architectures &mdash; PointNet,
PointNet++, DGCNN, and a GNS-style encoder&ndash;processor&ndash;decoder &mdash;
parameter-matched and trained with the identical physics-informed loss. The comparison
therefore isolates architecture: the question is not whether the physics term helps, but
whether a given architecture can exploit it.</p>

{T_ARCH}
<p class="caption"><b>Table 4:</b> Synthetic benchmark (N=49; K=5 passes; 64 fixed clouds).
*DGCNN at &sigma;=1.0 degenerates in 2 of 4 initialisations; the range shown is over the
runs that trained. &ldquo;deploys?&rdquo; = whether a sparse whole-cloud pass exists:
DGCNN's kNN graph is rebuilt in feature space each round and admits no fixed edge list.
PointNet deploys but is inert at N=2500 (Table&nbsp;1): with no pairwise term it cannot
see local geometry at any scale.</p>

<p>Two architectures beat ours on this benchmark: GNS at high disorder (0.0205 vs 0.0308)
and the max-aggregation variant of &sect;4.5 (0.0106, 2&times; better). <b>Both lose on
the real task</b> (Fig.&nbsp;2): at N=2500, GNS reaches only 0.132&ndash;0.143 (three
runs, no overlap with ours at 0.123&ndash;0.127), and under the matched production
training regime it reaches 0.242 &mdash; barely better than Transport Velocity. Model
selection on the small-N benchmark alone would therefore have shipped the worse deployer,
twice.</p>

<figure>
  <div class="scroll"><img src="{uri('benchmark_vs_deployment.png')}"
    alt="Slope chart: rank on the N=49 benchmark versus rank on the N=2500 trajectory"></div>
</figure>
<p class="caption"><b>Figure 2:</b> Benchmark rank versus deployment rank (|KG|, lower is
better). Crossing lines are architectures the benchmark misranks: GNS and the
max-aggregation variant both beat model12 at N=49 and lose at N=2500. model12 places 4th
of 5 on the benchmark and 1st among deployable architectures on the real task.</p>

<!-- MECHANISM -->
<h2><span class="num">4.5</span>What transfers: fixed kernel <em>and</em> additive aggregation</h2>
{chips('main')}
<p class="flagnote">Flag: keep the claim, the table, and the closing argument; the rung
descriptions can compress to one sentence each.</p>

<p>To locate the mechanism behind the benchmark&ndash;deployment reversal we ablate
model12 one component at a time, in a variant whose baseline configuration is
bit-identical to the production model. Two single-mechanism rungs are decisive:
<em>nokernel</em> replaces the fixed proximity kernel
(1&minus;(d/r<sub>c</sub>)&sup2;)&sup2; with a learned scalar gate, changing nothing else;
<em>wmax</em> keeps the kernel but replaces the weighted mean with a weighted max,
changing nothing else.</p>

{T_MECH}
<p class="caption"><b>Table 5:</b> Mechanism ablation, grouped by whether the fixed
geometric kernel is applied to messages <em>and</em> aggregation is additive. Ranges are
over available runs (1&ndash;3 per variant). The groups do not overlap at deployment:
0.118&ndash;0.127 versus 0.132&ndash;0.170 across eleven runs, four architectures, and
three aggregation schemes.</p>

<p>Changing either component alone &mdash; and nothing else &mdash; moves the model out of
the good group. This matches the structure of the objective: |KG| is a weighted
<em>sum</em> of kernel gradients, so an architecture that must drive it to zero at a
cardinality it never saw needs both the distance weighting (scale-free by construction)
and the summation. A learned weighting fits the training cardinality's statistics; a max
discards the balance information that summation carries. Per-particle normalisation, by
contrast, is <em>not</em> required &mdash; removing it (<em>nonorm</em>) transfers
marginally better than the production model.</p>

<!-- OPERATING POINT -->
<h2><span class="num">4.6</span>Operating point and training cardinality</h2>
{chips('main', 'cut')}
<p class="flagnote">Flag: keep the k=1 warning, the k=8 trade-off sentence, and the N=100
result in the main body (~half a page with Table&nbsp;6); the k-curve figure can move to
the appendix.</p>

<p>The corrector is applied iteratively; Table&nbsp;6 sweeps the number of passes k on the
trajectory. Three facts matter for deployment. <b>A single pass is harmful</b>: k=1 yields
|KG| 0.466 against 0.333 uncorrected, and the corrector only becomes a net win from
k&thinsp;&ge;&thinsp;3. |KG| then decreases monotonically with no floor through k=40
(0.068). The violation&ndash;symmetry trade-off instead appears in the constraint metric:
illegal pairs reach their minimum at k=8 and rise thereafter &mdash; beyond that point,
additional symmetry is bought at legality's expense. We use k=5 (sim-validated) for all
validated claims and recommend k=8 as the best all-round operating point.</p>

{T_K}
<p class="caption"><b>Table 6:</b> Correction passes on the trajectory (production
checkpoint). |KG| has no floor; illegal% turns at k=8. An operating-point sweep of this
kind is necessary to distinguish converged behaviour from artifacts: an earlier
&ldquo;|KG| floor&rdquo; hypothesis was an artifact of reading only k=5.</p>

<p>Training cardinality has a larger effect than any architectural choice we tested:
retraining the identical architecture at N=100 (with
{L3}&thinsp;&prop;&thinsp;1/N&sup2; rescaled accordingly, appendix
Table&nbsp;A2) improves deployment |KG| by 31% to <b>0.087</b> &mdash; the best result in
this work &mdash; consistent with reducing the gap between training and deployment scale
(receptive field 1.14 of the box at N=49 versus 0.80 at N=100, against 0.16 at
deployment). This checkpoint is a single run and has not been re-simulated.</p>

<!-- FAILURE MODE -->
<h2><span class="num">4.7</span>A failure mode invisible to the task metrics</h2>
{chips('appx')}
<p class="flagnote">Flag: move the figure and most of the prose to the appendix; keep the
two bolded sentences in the main body (they justify the seeding protocol and the
bulk-drift diagnostic).</p>

<p>Roughly half of training runs for several variants degenerate into a <b>saturated
uniform translation</b>: every particle moves by the same vector at exactly the output
bound. Because every term of the objective depends only on relative positions, this state
is invisible to |KG| and illegal% &mdash; both report the cloud unchanged while it
translates a full domain length over five passes (Fig.&nbsp;3). It is a tanh-saturation
trap: once the output head saturates, its gradient vanishes and no signal recovers it.
<b>Whether a run falls in depends only on initialisation</b> (e.g. DGCNN at &sigma;=1.0:
2 of 4 seeds), so we (i) log the bulk-drift fraction
|mean&thinsp;&Delta;x|/mean|&Delta;x| at every evaluation, and (ii) accept degeneration
claims only at 3/3 seeds &mdash; a bar only the {L3}=0 arm meets (4/4).</p>

<figure>
  <div class="scroll"><img src="{uri('displacement_decomposition.png')}"
    alt="Displacement fields decomposed into bulk translation and local restructuring"></div>
</figure>
<p class="caption"><b>Figure 3 (appendix):</b> Displacement decomposed into bulk
translation + local restructuring, one pass, identical cloud. Working models are
3&ndash;9% bulk; PointNet's local component is empty (10<sup>&minus;4</sup>) &mdash; its
entire output is the translation, which the task metrics cannot see.</p>

<!-- LIMITATIONS -->
<h2><span class="num">4.8</span>Limitations</h2>
{chips('main')}

<ul>
  <li><b>One trajectory, one solver validation.</b> The KG results derive from a single
  2-D SPH run. The bounded obstacle scene (&sect;4.2) shows the structural signature
  transfers to a second scenario, but it has no KG measure and no solver validation of its
  own.</li>
  <li><b>Shared hyperparameters favour our model.</b> The common recipe
  ({L3}, schedule, learning rate) was originally tuned on model12;
  equal-budget comparison without per-architecture search is the stated protocol, but the
  asymmetry is real.</li>
  <li><b>Best &ne; validated.</b> The re-simulation was performed with the production
  N=49 checkpoint at k=5; the stronger N=100 and k&thinsp;&ge;&thinsp;8 operating points
  are measured, not solver-validated, and the N=100 result is a single run.</li>
  <li><b>Scale limits of the loss.</b> The minimum-image KG term is exact only for
  N&thinsp;&gt;&thinsp;~50 on the unit torus (appendix Table&nbsp;A2); small-N ablations
  are not meaningful under this objective. 3-D requires a 3-D kernel and is future
  work.</li>
</ul>

<!-- CONCLUSION -->
<h2><span class="num">5</span>Conclusion and outlook</h2>
{chips('main')}
<p class="flagnote">Flag: draft prose, &asymp;0.4 page.</p>

<p>We presented a learned corrector that turns constraint-violating particle
distributions into usable SPH restart states, validated end-to-end by re-simulation. Its
two load-bearing ingredients are the same piece of physics entering twice: the
kernel-gradient residual in the objective &mdash; without which training itself diverges
and the corrector reproduces the failure of constraint-only prior work &mdash; and the
fixed, scale-free SPH kernel in the message weighting, which single-mechanism ablations
identify (together with additive aggregation) as the component that carries a model
trained on 49 particles to a 2,500-particle deployment. Along the way we found that the
small-scale synthetic benchmark inverts architecture rankings relative to the real task,
twice; we suggest treating deployment-scale evaluation as mandatory for learned particle
operators.</p>

<p><em class="term">Outlook.</em> Four directions follow directly. First, validation of
the stronger operating points found here &mdash; k=8 passes and training at N=100 (31%
better |KG| than the validated configuration) &mdash; by re-simulation. Second, three
dimensions: the corrector and its sparse deployment path are dimension-generic, and only
the quintic kernel-gradient primitive needs a 3-D normalisation. Third, disorder: the
noise level acts as a physical temperature, and the corrector's behaviour across it is
uncharted. Fourth, other particle methods: the recipe &mdash; a spacing constraint plus
the target method's own consistency residual, with the corresponding kernel built into
the architecture &mdash; is not SPH-specific; molecular-dynamics initial conditions with
an energy-based residual are the natural next instantiation.</p>

<!-- APPENDIX -->
<hr class="appendix-rule">
<h2><span class="num">A</span>Appendix material</h2>

<h3>A.1 Claim audit against measured seed variance</h3>
{chips('appx')}
{TA1_AUDIT}
<p class="caption"><b>Table A1:</b> Every comparative claim audited against the measured
run-to-run spread (&sigma;&asymp;9% synthetic, 3.5% range trajectory). Degeneration
claims for DGCNN and two ablation rungs are explicitly <em>not</em> made: outcomes are
initialisation-dependent (2 of 3&ndash;4 seeds).</p>

<h3>A.2 Scaling constraints on the objective</h3>
{chips('appx')}
{TA2_SCALE}
<p class="caption"><b>Table A2:</b> Two constraints tie the objective to cardinality. The
symmetry:violation gradient ratio grows as N&sup2; (|KG|&thinsp;&prop;&thinsp;&radic;N,
violation term &prop;&thinsp;1/N), so {L3} must scale as 1/N&sup2; for the
trade-off to be preserved; and the minimum-image evaluation of KG is exact only once the
kernel support 6&Delta;x fits within half the domain.</p>

<h3>A.3 Cost</h3>
{chips('appx')}
{TA3_COST}
<p class="caption"><b>Table A3:</b> Training vs deployment cost. The dense-training gap
(3&times;) does not survive to the sparse deployment path (1.1&ndash;1.4&times;): GNS's
persistent edge tensor is (B,N,N,H) dense but (E,H) sparse, and per-edge work is
near-identical.</p>

<h3>A.4 Additional qualitative comparison</h3>
{chips('appx', 'cut')}
<figure>
  <div class="scroll"><img src="{uri('side_by_side_n49.png')}"
    alt="One N=49 cloud through each architecture with displacement arrows"></div>
</figure>
<p class="caption"><b>Figure A1:</b> One synthetic cloud through each architecture (K=5;
arrows = displacement field; red rings = residual violations). PointNet's parallel arrows
show the uniform-translation failure of &sect;4.7 directly.</p>

<h3>A.5 Operating-point curve</h3>
{chips('appx')}
<figure>
  <div class="scroll"><img src="{uri('k_curve.png')}"
    alt="KG and illegal fraction versus number of correction passes"></div>
</figure>
<p class="caption"><b>Figure A2:</b> |KG| and illegal% versus correction passes k
(Table&nbsp;6 as a curve). The k=1 excursion above the uncorrected baseline and the
illegal% minimum at k=8 are the two features the deployment guidance rests on.</p>

<h3>A.6 Ablations run but not presented above</h3>
{chips('appx', 'cut')}
{tbl(['ablation', 'result', 'disposition'],
     [['', 'packing limit: r<sub>d</sub>/spacing = 1.00 (model12)', 'viol. red. 58.3%, |KG| 0.0214',
       'works at maximum rigidity (zero-jitter lattice is the only feasible state)'],
      ['', 'packing limit: comparative grid (3 architectures &times; 4 rungs)', 'dropped',
       'seed-fragility of degeneration made an honest version 36 runs'],
      ['', 'kernel width 1.5&times; spacing', '|KG| 0.0225', 'within seed spread of baseline: width uncritical, presence critical (&sect;4.5)'],
      ['bad', 'kernel width 3.0&times; spacing', 'degenerated', 'single run; not interpretable'],
      ['', 'all three mechanisms swapped at once (dgcnn-mech rung)', 'viol. red. 76.6%, |KG| 0.0260',
       'consistent with &sect;4.5; single run']])}
<p class="caption"><b>Table A4:</b> Completeness listing of ablations executed but not
carried in the main analysis, with the reason. Full per-run detail:
<code>paper/JOURNAL.md</code>.</p>

<h3>A.7 Assets, checkpoints and reproducibility</h3>
{chips('appx')}
{tbl(['checkpoint (src/models/weights/)', 'train N', 'deployment |KG|', 'status'],
     [['best', '<code>model12_sph_n100.pt</code>', '100', '<b>0.0871</b>',
       'best; n=1; NOT sim-validated; needs its own model config (r<sub>d</sub> 0.098, cutoff 0.200)'],
      ['', '<code>model12_sph_l4_noise1p0.pt</code>', '49', '0.1237', 'best N=49 regime, n=3'],
      ['hl', '<code>model12_sph_l4.pt</code>', '49', '0.1269', '<b>sim-validated</b> &mdash; quote for validated claims']])}
<p class="caption"><b>Table A5:</b> The three tracked checkpoints. The best and the
validated checkpoint are different runs; both 2026-08-10 checkpoints were trained
unseeded and will not reproduce byte-for-byte.</p>
<p class="muted" style="font-size:14px">Repository:
<code>github.com/ioandanielc/CorrectorWorkshop</code>, branch <code>sph-use-case</code>
(numbers frozen at commit <code>82f194a</code>, 2026-08-10). Every table above traces to
<code>paper/results.csv</code> (73 scored arms; scorer
<code>src/inference/experiments/ablations/score_arm.py</code>) or the 702-step sweep
(<code>kg_sweep_2026-07-21</code>). Run log with all retractions:
<code>paper/JOURNAL.md</code>. Figure scripts:
<code>vibecoding/visualizations/</code>. Guards: <code>tests/test_wholecloud.py</code>
(bit-exact vs the sim-validated artifact), <code>tests/test_sparse_paths.py</code>
(18 sparse/dense checks).</p>
<div class="note warn"><p><b>Figure status &mdash; content-final, not format-final.</b>
All five figures are PNG at 160&ndash;170&thinsp;dpi with titles baked into the image
(duplicating the captions here), and Fig.&nbsp;1 carries its colour scale only in the
caption. For camera-ready, regenerate as vector PDF at column width without in-image
titles, and add a colourbar to Fig.&nbsp;1 &mdash; the scripts in
<code>vibecoding/visualizations/</code> produce every figure from
<code>results.csv</code>; the choice of single- vs double-column width depends on the
venue template.</p></div>

<h3>A.8 References (BibTeX)</h3>
{chips('main')}
<p class="muted" style="font-size:14px">Verbatim copy of <code>paper/references.bib</code>
&mdash; 15 entries; <code>[web-verified]</code> = venue/authors checked against publisher
or arXiv on 2026-08-10, <code>[standard]</code> = canonical, re-check pages at
camera-ready.</p>
<div class="tablewrap"><pre style="font-size:11.5px;line-height:1.5;background:var(--surface);
border:1px solid var(--line);border-radius:4px;padding:14px 18px;overflow-x:auto">{BIB}</pre></div>

<footer>
  Sources: <code>paper/results.csv</code> (73 scored arms, one scoring tool) &middot;
  702-step sweep <code>kg_sweep_2026-07-21</code> &middot; obstruction run
  <code>obstruction/runs/scored_2026-08-10</code> &middot; run log with all retractions
  <code>paper/JOURNAL.md</code> &middot; consolidated results <code>paper/RESULTS.md</code>
  &middot; checkpoints <code>src/models/weights/</code> (the sim-validated and the best
  checkpoint are different runs &mdash; see Table&nbsp;1 caption) &middot; guards:
  <code>tests/test_wholecloud.py</code>, <code>tests/test_sparse_paths.py</code>.
</footer>
</div>
"""

OUT.write_text(HTML, encoding='utf-8')
print(f'-> {OUT}  ({OUT.stat().st_size / 1e6:.2f} MB)')
