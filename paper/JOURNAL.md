# Ablation run journal — physics-AI workshop 5-pager

Execution log for the ablation suite. Plan:
`C:\Users\ioand\.claude\plans\clever-meandering-koala.md`.

Tracked in git on purpose: `artifacts/` is gitignored and regenerable, so recovery after an
interrupt must not depend on it — or on conversation history.

---

## Observations log (newest first — read this for "what do we know")

### 2026-08-10 16:05 — ERROR: PointNet CAN run at N=2500; only DGCNN cannot
Measured: PointNet processes a 2500-point cloud in **6 ms**. It has no pairwise term at
all — per-point encoder then one global max-pool — so cost is O(N*H), not O(N^2). The
claim "the dense-only baselines cannot process 2500 points" appeared in
`ARCHITECTURES.md`, `RESULTS.md` and a figure caption, and was inferred from the word
"dense" rather than measured. Corrected in both documents.

**DGCNN** is the one that genuinely cannot scale: its kNN graph is an N x N distance
matrix rebuilt in feature space every round, so there is no fixed edge list to exploit.

This is the same failure mode as the 3x-cost claim and the KG floor: a number or property
carried over from one regime and asserted in another without measurement. Third instance
today. PointNet on the trajectory is now a cheap and worthwhile data point (expected: it
does nothing, making the no-locality floor visible on real data too).

### 2026-08-10 16:10 — tooling hardening (the day's rework, made cheaper next time)
- **Collapse detector** in `trainer.py`: logs `bulk_drift` every eval and warns when >50%
  of displacement is bulk translation. Gated to start after 25% of the schedule, because
  an untrained model is near-uniform too and an always-firing detector is useless. The
  warning compares against `max_disp * sqrt(D)`, the true saturation norm — `max_disp`
  alone bounds each COMPONENT, which is why measured displacements exceeded it all day.
- **Default `seed: 0`** in the production trainer config. This recipe has a ~50/50
  degenerate mode; leaving runs unseeded made every one a coin flip and irreproducible.
  Noted in the config that the shipped checkpoint predates this and was trained unseeded.
- **`tests/test_sparse_paths.py`** — 18 checks across 6 architectures x 3 cardinalities,
  all passing. `gns` and `model12_ablate` had gained sparse deployment paths with their
  equivalence verified only in throwaway scripts; now guarded in-repo alongside
  `test_wholecloud.py`. Confirms again that `ablate/baseline` is bit-identical to model12.
- **`watch_arms.py` DEAD detection** — a killed run leaves a log that simply stops and
  used to read as RUNNING forever. It misled me once today (the killed gns/rd1.00 rung).
  Now flags `DEAD(<minutes>)` when the log has not grown for 5 minutes.
- **`wmax` aggregation** added to `model12_ablate`: weighted max, so the kernel survives
  and only the aggregation operator changes. This is the clean single-change counterpart
  to `maxagg`, which silently also removed the weighting and confounded two mechanisms.
- New configs: `loss_config_rdsph_lam2_0.yaml` (4th loss cell),
  `ablations/cutoff/model_config_cutoff_{1p5,3p0}.yaml` (kernel width, never ablated —
  confound stated in the header: reach = L*cutoff moves with it),
  `bridge/model_config_bridge_wmax.yaml`.

### 2026-08-10 16:20 — seed spread is LARGER than assumed; claim audit performed
The bridge `baseline` rung is bit-identical to production model12, giving a third
independent sample of the same recipe:

| run | viol_red | \|KG\| |
|---|---|---|
| production (`train_run_..._10-40-18`) | 82.9% | 0.0216 |
| shipped `model12_sph_l4.pt` | 84.3% | 0.0207 |
| bridge baseline (`..._15-17-56`) | 80.5% | 0.0245 |

**sigma ~9% on |KG| (18% range), +-2 points on viol_red** — larger than the +-10% range
used all session. Re-audited every claim against it; the audit table now lives in
`paper/RESULTS.md`. Outcome:

- SOLID: the lambda3 ablation (6.5x synthetic, 11x trajectory), model12 vs PointNet
  (10x), model12 vs GNS at matched production training (90%), maxagg's benchmark win (2x).
- MARGINAL: model12 vs DGCNN at N=49 (26%, clears 18% only just); maxagg's deployment
  loss (18%, exactly at threshold).
- **NOT SEPARABLE, do not claim**: model12 vs GNS on the trajectory at each one's best
  (4-7%); nonorm vs model12 (5-10%).

Consequence: the "fixed kernel drives size transfer" grouping is **suggestive, not
established** — it rests on one solid gap (GNS production) and one borderline gap
(maxagg), with GNS-at-best inside the noise. RESULTS.md reworded accordingly. The
aggregation null result (nonorm ~ model12) is still useful: it rules aggregation OUT as
the mechanism even though it cannot positively confirm the kernel.

Lead with the deployment result — a 90% gap is the only architecture comparison that is
comfortably outside the noise.

### 2026-08-10 16:20 — bridge complete: 3 of 7 rungs collapsed and are unreadable
| rung | viol_red | \|KG\| | trajectory \|KG\| |
|---|---|---|---|
| maxagg | 98.8% | 0.0106 | 0.1495 |
| nonorm | 86.8% | 0.0196 | 0.1204 |
| baseline (= production model12) | 80.5% | 0.0245 | — |
| dgcnnmech (all three swapped) | 76.6% | 0.0260 | — |
| nokernel | collapsed | — | — |
| knngraph | collapsed | — | — |
| noperiod | collapsed | — | — |

`dgcnnmech` swaps all three mechanisms at once and trained fine (76.6%) while three
SINGLE-change rungs collapsed. That is incoherent as a mechanism signal and confirms the
collapses are stochastic. Combined with the DGCNN seed data (2 of 4 collapse), the
conclusion is that this recipe has a ~50/50 collapse mode at N=49 that is independent of
architecture. A usable bridge needs ~3 seeds per rung (21 runs, ~4 h) — not run.

### 2026-08-10 16:20 — seed chains complete
- **lambda3 = 0: 4 of 4 collapse** (unseeded + seeds 1/2/3). CLAIMABLE.
- **dgcnn @ noise 1.0: 2 of 4 collapse** (seeds 1 and 3 trained to 66.1% and 72.8%).
  Bimodal, stays retracted.

### 2026-08-10 16:00 — RETRACTED and REPLACED: it is the fixed KERNEL that transfers, not normalisation
`bridge_nonorm` is model12 with per-particle normalisation removed and nothing else
changed (fixed kernel weight kept, sum instead of weighted mean):

| | N=49 \|KG\| | N=2500 \|KG\| |
|---|---|---|
| model12 | 0.0216 | 0.1269 |
| nonorm | 0.0196 | **0.1204** |

Removing normalisation does **not** break size transfer — it transfers marginally better.
**The claim "model12 normalises per particle, which is what survives N=49 -> N=2500" is
WRONG and is retracted** from `ARCHITECTURES.md` and `RESULTS.md`. It had been presented
as a mechanism predicted a priori and then confirmed; the confirmation was a coincidence
of GNS differing from model12 in several ways at once.

**Replacement, and it is better supported.** Regrouping every deployed arm by whether the
FIXED GEOMETRIC KERNEL is actually applied to messages:

| arm | kernel weighting active | N=2500 \|KG\| |
|---|---|---|
| nonorm (kernel + sum) | yes | **0.1204** |
| model12 (kernel + weighted mean) | yes | **0.1269** |
| GNS (learned weights + sum) | no | 0.1319 best / 0.2417 production |
| maxagg (max — weight is inert) | no | 0.1495 |

Clean separation at 0.120-0.127 versus 0.132-0.242, across four architectures and three
different aggregation schemes. **What buys size transfer is the fixed, distance-shaped
kernel weighting — a scale-free geometric function — not normalisation and not the
aggregation operator.** Learned weightings fit N=49 statistics and do not generalise;
max discards the weighting entirely and also fails.

This is a stronger result for the paper than the retracted one: it says the
physics-shaped kernel is what makes the corrector deployable, which is the thesis.

Note nonorm also disposes of the `maxagg` confound worry in the right direction: nonorm
isolates aggregation (sum vs weighted mean, kernel kept) and shows aggregation barely
matters. So maxagg's deployment failure is attributable to losing the kernel, not to max.

### 2026-08-10 15:45 — `maxagg` reverses on deployment too: SECOND instance, key result
`model12_ablate` gained a `forward_sparse` for radius-graph rungs (exact vs dense to
4.4e-07 on all four), so the surprising rung could be deployed.

| | N=49 \|KG\| | N=49 viol_red | **N=2500 \|KG\|** | N=2500 illegal% |
|---|---|---|---|---|
| model12 (production) | 0.0216 | 82.9% | **0.1269** | **82.0%** |
| bridge `maxagg` | **0.0106** | **98.8%** | 0.1495 | 88.3% |

**maxagg wins the synthetic benchmark by 2x on |KG| and loses deployment by 18%.** This
is the SECOND architecture today to reverse between benchmark and deployment — GNS was
the first. Two independent instances make "the small-N synthetic benchmark misranks
architectures relative to the real task" the suite's most robust methodological finding,
and it is no longer a one-off anecdote.

Resolution of the max-aggregation argument: **partially retracted, and sharper for it.**
The claim "max cannot express cancellation" is plainly WRONG at N=49, where max beats the
kernel-weighted mean on every metric. It appears to hold where it matters — at N=2500,
where neighbourhoods are genuinely local, discarding all but the extremal neighbour loses
the balance information KG symmetry needs. At N=49 the receptive field spans the whole
cloud (reach/box 1.14), so the model can recover that information by other routes.

Caveat that must travel with this result: the `maxagg` rung changes TWO mechanisms (max
AND weighting removed, since max has nowhere to apply a per-edge scalar), so the effect
cannot be attributed to aggregation alone. A weighted-max rung is the clean follow-up.

### 2026-08-10 15:45 — PointNet++ completes the architecture table
`arch_pointnet2_n49_noise0.6`: viol_red 19.1%, |KG| 0.2260 -> 0.1065, nn_CV 34.9 -> 23.8%,
drift 33%. Poor but not inert — clearly better than PointNet (0.2%, |KG| unchanged) and
far behind every message-passing architecture. Consistent with the prediction that
feature-propagation interpolation smooths away the sub-rd precision the task needs.

### 2026-08-10 15:45 — lambda3=0 is 4/4 and DGCNN is 2/3
lambda3=0: unseeded + seeds 1, 2 collapsed; seed 3 collapsed by iteration 6100. Four
independent initialisations, all collapse. **Claimable.**
dgcnn @ noise 1.0: unseeded collapsed, seed 1 trained (64.8%), seed 2 collapsed.
Bimodal — **not claimable**, as already retracted.

### 2026-08-10 15:25 — `maxagg` is beating the baseline, and the rung is confounded
At iteration 4800 the max-aggregation rung reads 93.9% viol_red and |KG| 0.0157 against
model12's baseline final of 85.7% / 0.0228. That directly contradicts the claim in
`ARCHITECTURES.md` that "max aggregation cannot express cancellation ... the deepest
structural mismatch with the KG objective". If it survives scoring, that argument is
retracted.

**Found while checking: the rung changes TWO things, not one.** In
`model12_ablate.forward`, the `max` branch never applies the weight — a max over
neighbours has nowhere to put a per-edge scalar. So `aggregation=max` is really
"sum -> max AND weighting removed", unlike the other rungs which each change exactly one
mechanism. Code comment added. Consequence: `maxagg` cannot be used to attribute an
effect to aggregation alone, and its surprising strength may come from either change.

A clean single-change version would be `max_j (w_ij * e_ij)` — weighted max. Not
implemented; noted as the correct follow-up if this result matters to the paper.

### 2026-08-10 15:25 — DGCNN @ noise 1.0 is BIMODAL across seeds: 2 of 3 collapse
unseeded collapsed, seed 1 trained normally (64.8%), seed 2 collapsed by iteration 2700.
So the outcome is not "collapses" or "trains" but a coin flip on initialisation. This is
the sharpest possible demonstration of why the collapse claim was retracted, and it is
worth one sentence in the paper's limitations: on this task DGCNN's training outcome at
high disorder is initialisation-dependent, which is itself a robustness finding — just
not the deterministic mechanism claim originally made.

### 2026-08-10 15:00 — RETRACTED: the "KG floor ~0.111" is an artifact of k=5
Swept correction passes on the real trajectory, production model12 checkpoint,
15 timesteps t>=300:

| k | \|KG\| | vs k=5 | mean nn | ill% | s/step |
|---|---|---|---|---|---|
| 0 (raw) | 0.3331 | — | 0.01457 | 99.4% | — |
| 1 | **0.4661** | +267% | 0.01691 | 97.1% | 0.084 |
| 2 | 0.2710 | | 0.01837 | 92.0% | 0.079 |
| 3 | 0.1860 | | 0.01902 | 87.9% | 0.090 |
| 5 (shipped) | 0.1269 | 0.0% | 0.01947 | 82.0% | 0.110 |
| 8 | 0.0984 | -22.4% | 0.01963 | 79.2% | 0.143 |
| 12 | **0.0846** | **-33.3%** | 0.01966 | 80.4% | 0.184 |

Two findings:

1. **The floor is not a floor.** The 0.1109 min / 0.1135 p10 recorded from the 702-step
   sweep was measured at k=5 only. At k=12 the corrector reaches 0.0846 — well below it,
   still descending — for 0.07 s/step more. The "fundamental violation<->symmetry
   trade-off floor" framing in CLAUDE.md, RESULTS.md and private memory is WRONG as
   stated; it is an operating-point artifact. Whether a true floor exists is now open
   (extended sweep to k=40 running).
2. **k=1 is actively harmful**: |KG| 0.4661 against raw 0.3331. One pass makes the cloud
   a WORSE SPH restart than no correction. The corrector only becomes a net win from
   k>=3. Worth stating explicitly — it is a deployment trap.

Consequence for the paper: the headline improves. model12 at k=12 gives 0.0846 vs TV's
0.274 = **3.2x**, not the 2.2x quoted from k=5. The sim validation was performed at k=5,
so the k=5 number remains the *validated* one — quote both, and be explicit about which
was simulated.

### 2026-08-10 15:10 — extended sweep to k=40: no floor, and the real trade-off located
| k | \|KG\| | ill% | s/step |
|---|---|---|---|
| 5 | 0.1269 | 82.0% | 0.121 |
| 8 | 0.0984 | **79.2%** | 0.143 |
| 12 | 0.0846 | 80.4% | 0.177 |
| 16 | 0.0796 | 82.3% | 0.218 |
| 20 | 0.0769 | 83.6% | 0.258 |
| 30 | 0.0715 | 86.6% | 0.346 |
| 40 | 0.0675 | 88.4% | 0.439 |

**|KG| never floors** — monotone down to 0.0675 at k=40, still descending. So the floor
claim is dead in its entirety, not merely mis-located.

**But the violation<->symmetry trade-off is real; it just lives in k.** Illegal% bottoms
at k=8 (79.2%) then climbs to 88.4% by k=40 while |KG| keeps improving, and nn saturates
at 0.01966 from k=12. Pushing symmetry harder actively costs legality. That is the genuine
physical trade-off the "floor" was a garbled version of — and it is a better paper result,
because it is a curve the reader can act on rather than an asserted limit.

Recommended operating points: k=5 (sim-validated), k=8 (best all-round: lowest illegal%,
|KG| 22% better than k=5), k>=12 (symmetry-first).

### 2026-08-10 14:40 — BEST RESULT: lambda3=0 on the real trajectory reproduces model9
Deployed the physics-off checkpoint on N=2500, t>=300: **|KG| 0.3331 -> 1.4712**, i.e.
4.4x WORSE than leaving the cloud alone, and adjacent to model9's 1.2775.

| series | \|KG\| |
|---|---|
| raw | 0.326 |
| model9 (prior work, no symmetry term) | 1.278 |
| **model12 with lambda3=0** | **1.471** |
| model12 with lambda3=0.27 | 0.127 |

model9 is the checkpoint whose corrected clouds were unusable as SPH restarts — the
failure that motivated model12. Ablating the KG term **recreates that failure**. This
converts the loss claim from "our term improves a metric" to "this term is what fixed the
thing that was broken", demonstrated by putting the breakage back. Cost: 5 minutes of
scoring on a checkpoint we already had.

Combined with lambda3=0 collapsing in 3 independent runs and the monotonic val-loss
divergence, the loss ablation is now the most robust result in the suite.

### 2026-08-10 14:40 — WATCH: bridge `maxagg` contradicts the max-aggregation argument
At iteration 3300 the max-aggregation rung reads 92.3% violation reduction and |KG| 0.0157
— better than model12's baseline final (85.7% / 0.0228). `paper/ARCHITECTURES.md` argues
"max aggregation cannot express cancellation ... the deepest structural mismatch with the
KG objective". If this holds to 10000 that argument is WRONG and must be retracted.
Mid-run readings have misled twice today, so no conclusion until it finishes.

Note `nokernel` (rung 1) finished collapsed at 0.0%. n=1, and collapses here are
seed-fragile, so it is NOT interpreted.

### 2026-08-10 14:20 — CORRECTED: the "3x cheaper" claim does not hold at deployment
That figure is TRAINING cost at N=49 dense (model12 0.073 vs GNS 0.223 s/iter). Measured
deployment cost through the whole-cloud sparse path, N=2500, k=5, 5 timed repetitions:

| device | model12 | GNS | ratio |
|---|---|---|---|
| CPU | 0.189 s/timestep | 0.213 s | **1.13x** |
| CUDA | 0.049 s/timestep | 0.068 s | **1.39x** |

The dense gap comes from GNS materialising a persistent `(B,N,N,H)` edge tensor; on the
sparse path that becomes `(E,H)` and per-edge work is near-identical (GNS 321x107 vs
model12 260x128 multiply-accumulates). **At deployment model12 is 1.1-1.4x cheaper, not
3x.** `paper/RESULTS.md` corrected; quote the deployment number, never the training one.

Method note: this is why "we already measured that" deserves checking — the quantity that
had been measured (training, N=49, dense) was not the quantity being claimed (inference,
N=2500, sparse).

### 2026-08-10 14:00 — RETRACTED: "DGCNN collapses on hard data" is an init artifact
`dgcnn @ noise 1.0, seed 1` finished at **64.8% violation reduction, |KG| 0.0481** — it
trained normally. The unseeded run of the identical config collapsed to 0.0% for all
10k iterations.

Per the rule fixed before results were seen (3/3 collapse required; any seed training
normally = artifact), **this claim is dead and must not appear in the paper.** Every
earlier journal entry asserting DGCNN collapses at high noise is superseded by this one.

What survives: DGCNN is *worse* than model12 at noise 0.6 on scored metrics (77.1% vs
82.9% viol_red, |KG| 0.0272 vs 0.0216) — a quantitative gap above the measured seed
spread, which needs no collapse story.

**General lesson, and it now governs how every remaining result is read: collapses in
this setup are SEED-FRAGILE.** A single collapsed run means nothing. This applies to
the bridge `nokernel` rung (currently collapsed at 7800, n=1 — do not interpret) and to
`pack_dgcnn_rd1.00` (collapsed, n=1 — the packing-limit claim now also needs seeds
before it can be stated).

### 2026-08-10 14:00 — lambda3=0 collapse is holding: 2/2, seed 2 in progress
seed 1 finished collapsed (0.0% through 10000); seed 2 collapsed by iteration 4100.
Unlike DGCNN's, this one is replicating. Still needs seed 3 for 3/3. Note it also rests
on a second, independent signal that does not depend on the collapse at all: the
deterministic validation loss peaked at iteration 500 and never improved again, while
lambda3=0.27 improved monotonically to 10000.

### 2026-08-10 14:00 — trajectory gap model12 vs GNS is stable under finer sampling
Re-scored at stride 10 (71 timesteps) vs the original stride 50 (15 timesteps):

| arm | 15 steps | 71 steps |
|---|---|---|
| model12 noise1.0 | 0.1237 | **0.1230** |
| gns noise1.0 | 0.1346 | **0.1319** |
| gap | 8.1% | **7.2%** |

Quadrupling the sample barely moved either number, so the gap is a stable MEASUREMENT,
not a sampling artifact. It is still n=1 on seeds against a measured +-10% seed spread on
|KG|, so the best-vs-best claim remains unclaimable without 3 seeds per arm. The
production-settings comparison (0.1269 vs 0.2417, 90%) is unaffected and stands.

### 2026-08-10 13:45 — DECISIVE: GNS's synthetic win does NOT transfer to deployment
GNS gained a `forward_sparse` (exact vs dense to 1.6e-07 at N=40/120/400), so it could be
run through `WholeCloudCorrector2D` on the real trajectory. Scored, N=2500, 15 timesteps
t>=300, raw |KG| = 0.3331:

| arm | \|KG\| out | mean nn | illegal% |
|---|---|---|---|
| **model12 noise1.0** | **0.1237** | 0.01948 | 80.9% |
| **model12 noise0.6 (production)** | **0.1269** | 0.01947 | 82.0% |
| gns noise1.0 | 0.1346 | 0.01965 | 78.1% |
| gns noise0.6 | **0.2417** | 0.01864 | 86.4% |

At N=49 noise 1.0, GNS beat model12 decisively (91.5% vs 82.0% viol_red, |KG| 0.0205 vs
0.0308). On the real N=2500 cloud **model12 wins in both training conditions**, and under
the production condition GNS is nearly 2x worse — barely better than TV's 0.2735.

**The mechanism was predicted before the run.** `paper/ARCHITECTURES.md` already listed as
a GNS con: "unnormalised sum means message magnitude scales with neighbour count ... works
against size transfer". model12's per-particle normalisation makes node states independent
of degree, so it survives N=49 -> N=2500; GNS's plain sum does not. A mechanism predicted
a priori and then confirmed on real data is far stronger evidence than a benchmark win.

**Methodological consequence worth its own paragraph in the paper: the synthetic small-N
benchmark MISRANKS the architectures relative to real deployment.** Anyone selecting an
architecture on the N=49 high-disorder benchmark alone would have picked GNS and shipped
the worse deployer.

Pipeline sanity check: model12 noise0.6 scores 0.1269 here vs the independently recorded
0.128 from the full 702-step `kg_sweep` — the scorer agrees with the sim-validated artifact.

### 2026-08-10 13:15 — CORRECTION: the "collapse" is a saturated UNIFORM TRANSLATION
Earlier entries describing collapsed arms as "exactly zero displacement" were WRONG.
Measured displacement-field decomposition (one pass, 32 clouds):

| arm | mean \|d\| | \|mean d\| | bulk share | local spread |
|---|---|---|---|---|
| model12 noise0.6 | 0.0372 | 0.0015 | **4.0%** | 0.0372 |
| gns noise0.6 | 0.0357 | 0.0029 | 8.1% | 0.0356 |
| dgcnn noise0.6 | 0.0370 | 0.0039 | 10.5% | 0.0368 |
| dgcnn noise1.0 | 0.2375 | 0.2375 | **100.0%** | 0.0000 |
| pointnet noise0.6 | 0.1680 | 0.1680 | **100.0%** | 0.0001 |
| lambda3=0 | 0.1729 | 0.1689 | 97.7% | 0.0283 |

The degenerate arms emit a **saturated uniform translation**: every point moves by the
same vector, at exactly `max_displacement` (0.168 per component; 0.2376 norm per pass;
1.188 over K=5 — which is precisely the `disp_k5` recorded in results.csv).

Why it hides: every loss term depends only on RELATIVE positions, so a uniform
translation is invisible to the violation and KG terms. |KG|, illegal% and viol_red all
read "unchanged" while the cloud slides a full box length. **|KG| and illegal% cannot
detect this failure — only the new `uniform_frac` diagnostic can.**

Mechanism: a tanh saturation trap. Once the output head saturates, d(tanh)/dz ~ 0 and no
gradient pulls it back. Caveat for the paper: the bounded `tanh * max_displacement` head
is OUR fairness choice applied to all five architectures, so the trap is partly induced
by it. model12 avoids it; the others fall in.

### 2026-08-10 13:15 — CONFIRMED: GNS beats model12 at high disorder
Scored, N=49, noise 1.0*rd:

| arm | viol_red | \|KG\| | knn_keep | drift |
|---|---|---|---|---|
| **gns noise1.0** | **91.5%** | **0.0205** | 0.602 | 2% |
| model12 noise1.0 | 82.0% | 0.0308 | 0.633 | 4% |

At noise 0.6 model12 wins decisively (82.9%/0.0216 vs 62.5%/0.0365); at noise 1.0 GNS
wins by a similar margin in the other direction. **The fixed kernel is not
unconditionally better — it is better when disorder is moderate, and at 3x less
compute.** Coherent reading: a strong prior wins when signal is scarce; learned
flexibility wins once the task supplies enough signal to fit the weighting itself.

Crucially GNS achieves this **using our KG loss** — every arm shares
`loss_config_rdsph_lam3_0p27.yaml`. So the paper's contribution should be framed as the
LOSS (which transfers across architectures) plus the efficient kernel-weighted
architecture, not as an architecture bake-off.

### 2026-08-10 13:15 — topology preservation does NOT favour model12 (raw)
New scorer metrics `knn_keep_k5` (Jaccard of each particle's 6-NN set before/after) and
`uniform_frac_k1`. Among working models: model12 0.661, dgcnn 0.669, gns 0.680 — model12
is slightly LOWEST, because it corrects the most and therefore rewires the most.

**The metric is confounded**: models that do nothing score ~0.99 (pointnet 0.984, dgcnn
noise1.0 0.995). It must be read jointly with violation reduction. Correction per unit of
rewiring: model12 244, dgcnn 233, gns 195 — model12 leads, but present it as a ratio or a
reviewer will point straight at the raw column.

### 2026-08-10 13:15 — qualitative figures produced
`vibecoding/visualizations/corrector_side_by_side.py` -> `outputs/`:
- `side_by_side_n49.png` — one N=49 cloud through all four architectures with the
  displacement field as arrows; PointNet's parallel arrows show the failure by eye.
- `side_by_side_sph_t1000.png` — SPH t=1000, N=2500, raw / TV / model12, full box plus a
  0.35 zoom. raw mean nn 0.01435 |KG| 0.3097; TV 0.01684 / 0.1407; model12 0.01958 /
  0.1111. Only model12 appears: the dense-only baselines cannot process 2500 points.
- `displacement_decomposition.png` — total motion vs motion after removing the bulk
  translation. PointNet's local panel is EMPTY (mean |local| 0.0001) against a bulk of
  0.168. This is the figure that carries the drift claim.

### 2026-08-10 12:50 — bridge PROMOTED ahead of the packing tail (user decision)
Reframing that motivated it: model12 is best positioned not as "beats the baselines" but
as **a substitution on GNS — the learned edge machinery replaced by the fixed SPH kernel
it was approximating** — at matched parameters, 3x less compute, and better results in
the target regime (82.9% vs 62.5% viol_red at noise 0.6). Not a literal ablation: model12
also carries the `relu(rd-d)` feature, per-particle normalisation, and geometry
re-injected every round, which GNS lacks. Say "substitution", not "subset".

The bridge measures exactly this axis at three points — model12 (fixed kernel) ->
`nokernel` (learned scalar gate, everything else identical) -> GNS (full learned edge
latent) — so it was promoted over the remaining packing rungs.

Packing chain STOPPED after the rd/s=1.00 rung was 2/3 complete (model12 done, dgcnn
done, gns had just started so ~1 min was lost). **Still owed: gns @ rd1.00, and all of
rungs 0.95 / 0.90 / 0.80 (10 runs).**

Bridge order (most informative first): nokernel, maxagg, nonorm, knngraph, noperiod,
dgcnnmech, baseline.

### 2026-08-10 12:50 — DGCNN collapses at the packing limit too
`pack_dgcnn_rd1.00` scored: **viol_red -0.5%, |KG| 0.2303 -> 0.2303** (unchanged), versus
`pack_model12_rd1.00` at **58.3% / 0.2303 -> 0.0214**. So DGCNN collapses under BOTH
stressors — high noise and maximum constraint rigidity — while model12 handles both.
Consistent with the max-aggregation mechanism rather than anything data-specific.

### 2026-08-10 12:30 — lambda3=0 FINISHED: the physics term is an optimisation enabler
Three findings, from the deterministic validation loss (fixed 256-cloud set — the
trustworthy signal, not the noisy per-iteration block):

**1. Best-checkpoint comparison (both scored from `model_best.pt`, the standard protocol):**

| arm | viol_red | illegal% | \|KG\| |
|---|---|---|---|
| model12 lambda3=0.27 | **82.9%** | 4.0 -> 0.7 | **0.0216** |
| model12 lambda3=0 | 17.6% | 4.0 -> 3.3 | 0.1407 |

Removing the physics term costs 65 points of violation reduction and 6.5x the |KG| —
and note the ablated term is the KG term, yet the *violation* objective is what
degrades most. The physics term is not polishing symmetry; it is making the whole
optimisation work.

**2. Training diverges without it.** Best val loss by iteration:
- lambda3=0: best at **iteration 500** (0.037355), never improved again across the
  remaining 9500 iterations.
- lambda3=0.27: improved **monotonically to iteration 10000** (0.007827), best == final.

**3. The final iterate collapses completely** to exactly zero displacement (|KG| out ==
|KG| in). `model_best.pt` preserves the pre-collapse iteration-500 state, which is why
the scored row above is 17.6% rather than 0%. Both numbers are real and describe
different checkpoints of the same run — always say which.

Mechanism, consistent with all three: the violation term is a mean over N^2 pairs of
which only O(N) violate, so its gradient is diluted ~1/N and shrinks further as the easy
violations clear; the displacement penalty is undiminished. Retreating to zero
displacement becomes locally profitable. The KG term supplies a dense per-particle
gradient that prevents the retreat. Seeds 1 (collapsed) and the unseeded run agree; 2 and
3 pending.

### 2026-08-10 12:30 — WATCH: GNS may beat model12 at noise 1.0
GNS noise-1.0 at iteration 3400 reads 83.9% viol_red and |KG| 0.0327, against model12's
FINAL noise-1.0 scores of 82.0% / 0.0308. Too early to call (mid-run readings already
misled once today — see the GNS noise-0.6 entry), but if it holds, the fixed kernel's
advantage may be regime-dependent rather than universal. Do not write the architecture
section until this arm finishes.

### 2026-08-10 12:20 — GNS finished: model12 still clearly ahead, but GNS is a real baseline
Scored, N=49, noise 0.6: **GNS viol_red 62.5%, |KG| 0.0365** against **model12 82.9% /
0.0216**. So the fixed SPH kernel is worth ~20 points of violation reduction and ~69%
of |KG| against an otherwise identical architecture (same radius graph, same cutoff,
same rounds, same parameters, same loss). This is the suite's cleanest single number for
"writing the kernel into the wiring beats learning it".

Caution on mid-run readings: GNS logged 29.8% at iteration 9400 and finished at 62.5%.
The per-iteration block is a fresh random batch and is extremely noisy — **never judge an
arm from it before completion.** Its mid-run |KG| readings (~0.11) were similarly
pessimistic against a final 0.0365.

### 2026-08-10 12:20 — GNS does NOT collapse at noise 1.0, and that matters
At iteration 800 of the noise-1.0 arm, GNS is already at 59.2% violation reduction with
|KG| 0.3400 -> 0.0519. DGCNN and PointNet both sat at exactly zero displacement for the
whole 10k on that same data, same loss, same budget.

So the collapse is **not** a generic "baselines fail on hard data" effect — it is
specific to DGCNN's mechanism set (kNN graph + max aggregation). GNS keeps a radius graph
and a summed aggregation and copes fine. That considerably strengthens the mechanistic
story and removes the "you just under-trained the baselines" objection.

### 2026-08-10 12:20 — lambda3=0 seed 1 REPLICATED the collapse (2/2 so far)
Seed 1 climbed to 17.6% by iteration 1300, then fell to exactly 0.0% by 3100 with |KG|
unchanged — the same learn-then-collapse trajectory as the unseeded run. Two of two.
Seeds 2 and 3 still to run; the pre-agreed rule (3/3 required) stands.

### 2026-08-10 12:20 — model12 at the packing limit, scored
`pack_model12_rd1.00`: viol_red 58.3%, illegal 4.6 -> 1.9%, **|KG| 0.2303 -> 0.0214**.
The |KG| result at maximum rigidity slightly BEATS its own rd/s=0.98 arm (0.0216).
Plausible mechanism: at zero jitter the feasible configuration IS the perfectly symmetric
lattice, so the violation and symmetry objectives stop competing. Violation reduction is
lower (58.3% vs 82.9%) because the constraint is far harder to satisfy exactly.
DGCNN's rd/s=1.00 rung is now running — the head-to-head.

### 2026-08-10 12:15 — the collapse is LEARN-THEN-COLLAPSE, not failure-to-start
The unseeded `lambda3=0` arm reached 13.8% violation reduction at iteration 300, then
fell to exactly 0.0% by 3200 and stayed there through 4600. Seed 1 of the replication is
at 13.4% and |KG| 0.2514 -> 0.1635 at iteration 900 — i.e. currently learning, exactly
where the unseeded run also still looked healthy.

**Consequence for reading these runs: early-iteration readings are not predictive, and a
seeded arm cannot be scored "trained normally" until it is near 10k.** The model first
learns to correct, then abandons it. That is consistent with the dense-vs-sparse gradient
hypothesis — once the easy violations are cleared, the remaining violation gradient is
diluted to ~1/N and the displacement penalty dominates, so retreating to zero
displacement becomes locally profitable. With lambda3 > 0 the KG term keeps supplying a
dense gradient and there is no such retreat.

Do not conclude anything from the replication until all six runs are near 10k.

### 2026-08-10 12:15 — N=49 architecture block COMPLETE (6 arms)
At the harder noise level, **three of the four architectures do nothing at all**:

| arm | viol_red (scorer) | \|KG\| in -> out |
|---|---|---|
| model12 noise1.0 | **82.0%** | 0.3501 -> **0.0308** |
| dgcnn noise1.0 | 0.2% | 0.3501 -> 0.3504 |
| pointnet noise1.0 | 0.1% | 0.3501 -> 0.3502 |

model12 is the only architecture that functions at noise = 1.0*rd. DGCNN and PointNet
both sit at the zero-displacement solution, leaving |KG| fractionally WORSE than the
input. Scored numbers, periodic metrics, fixed evaluation clouds.

### 2026-08-10 12:05 — seed replication launched for the two collapse arms
`trainer.py` gained an optional `seed` (train config key or `--seed`). It seeds **torch
only**, so model initialisation varies while the data sequence stays pinned by the
dataset config's own seed — which is exactly what separates "initialisation artifact"
from "property of the recipe". Absent = unseeded = the historical behaviour of every run
before today. Seed is logged (`Torch seed: N`) rather than written into the config
snapshot, since it arrives by CLI; `watch_arms.py` parses it from the log.

Running: seeds 1/2/3 x {`lambda3=0`, `dgcnn @ noise 1.0`}, 6 runs.
Decision rule agreed in advance: **3/3 collapse = claimable; any seed that trains
normally = the collapse is an initialisation artifact and must not be claimed.**

### 2026-08-10 11:55 — model12 is unaffected at the packing limit
`rd/spacing = 1.00` (zero lattice jitter; the exact lattice is the ONLY feasible
configuration, KG exactly 0) — model12 reached |KG| 0.0230 at iteration 6700, matching
its rd/s=0.98 result (0.0228), still improving. No sign of the stall that hit DGCNN on
hard data. DGCNN's rd/s=1.00 rung is next in that chain and is the direct head-to-head.

### 2026-08-10 11:50 — GNS plateaus roughly 5x behind model12
|KG| descending 0.129 -> 0.110 -> 0.097 but flattening near ~0.11 against model12's
final 0.0228, with violation reduction ~33% against 85.7%. Same radius graph, same
cutoff, same rounds, same parameters, same loss — the difference is the fixed SPH kernel
weighting versus a learned one. This is the suite's sharpest single comparison and the
number most able to change the story; it still has iterations left.

### 2026-08-10 11:45 — fairness protocol fixed, per-architecture tuning ruled out
Equal budget for every architecture, model12 included; no per-model hyperparameter
search. Two disclosures instead: (1) the shared recipe was tuned on model12 in July, the
honest asymmetry, belongs in limitations; (2) DGCNN's collapse is a task effect, not a
tuning artifact — 79.4% at noise 0.6 and 0.0% at noise 1.0 on identical hyperparameters,
which also rules out an implementation fault since the same code path works at low noise.

## Next action

**Two training chains are RUNNING (launched ~16:00, ~2.5 h of GPU between them).** This
is the agreed final block of work for the day; nothing new should be added after it.

Chain 1 `bkv29tadp` — seeds, to move claims off n=1:
  gns @ noise1.0 seeds 1,2 · nonorm seeds 1,2 · maxagg seeds 1,2
Chain 2 `b5i3hyw0c` — new arms:
  lambda2=0 · cutoff_rd 1.5x and 3.0x spacing · wmax seeds 0,1 · model12 @ N=100

When they land, in order:
1. Score every new arm (`score_arm.py`), including **trajectory** scoring for nonorm,
   maxagg, wmax and N=100 — the deployment number is the one that matters.
2. Re-check the claim audit in `RESULTS.md` against the new n=3 evidence; the fixed-kernel
   mechanism should move from "suggestive" to established or be dropped.
3. Rebuild exhibits, commit, push. **Then stop for the day.**

Still owed and NOT scheduled: PointNet on the trajectory (it can run — see the correction
below), pure-sph arm re-scored with current tooling, obstruction experiment scored with
the new metrics, reproducibility script, k-curve and benchmark-vs-deployment figures,
CLAUDE.md de-staling.

Note on concurrency: measured contention makes model12 2.3x slower (0.073 -> 0.167
s/iter), so two chains cost ~13% total throughput. Accepted here to fit the time budget.

---

## Status board

| Phase | Step | Status |
|---|---|---|
| 0a | Create journal + `paper/` | done |
| 0 | Trainer: backward inside unroll loop | done |
| 0 | Trainer: `micro_batch_size` gradient accumulation | done |
| 0 | Verify gradient equivalence (old vs new, N=49) | done |
| 0 | Verify N=49 recipe reproduces recorded metrics | running |
| 0 | Confirm val-loss chunking holds at N=196 | done |
| 1 | `score_arm.py` — synthetic scorer | todo |
| 1 | `score_arm.py` — real-trajectory scorer | todo |
| 1 | `results.csv` aggregation | todo |
| 2 | `pointnet.py` | todo |
| 2 | `dgcnn.py` | todo |
| 2 | Architecture configs | todo |
| 2 | 12 baseline runs (3 arch x N{25,49} x noise{0.6,1.0}) | todo |
| 3 | `loss_config_rdsph_lam3_0.yaml` (lambda3=0 control) | todo |
| 3 | Loss ablation runs @ N=49 | todo |
| 4 | Measure lambda3 for N=16, N=25 | todo |
| 4 | Configs for N=16, N=25 | todo |
| 4 | Train ladder 16/25/49/100/196 | todo |
| 4 | train-N x deploy-N scoring matrix | todo |
| 4 | Ladder scored on real trajectory (KG floor readout) | todo |
| 5 | `model12_ablate.py` with component flags | todo |
| 5 | 5 bridge-rung runs @ N=49 | todo |
| 6 | Exhibit scripts | done |
| 6 | Main-body exhibits + appendix | done (4 exhibits, 40 rows) |
| + | Seeds: lambda3=0 (4/4), dgcnn (2/4 bimodal) | done |
| + | Bridge 7 rungs (4 usable, 3 collapsed at n=1) | done |
| + | Deployment: GNS + maxagg + nonorm + lambda3=0 on trajectory | done |
| + | k-sweep k=1..40; deployment cost; structural metrics | done |
| + | Tooling: collapse detector, default seed, sparse tests, DEAD detection | done |
| + | Chain 1: gns/nonorm/maxagg seeds | running |
| + | Chain 2: lambda2=0, cutoff sweep, wmax, N=100 | running |
| + | PointNet on trajectory; pure-sph re-score; obstruction scoring | todo |
| + | Repro script; k-curve + benchmark-vs-deployment figures | todo |
| + | CLAUDE.md de-staling (holds 2 retracted claims) | todo |

Legend: `todo` | `running` | `done` | `failed` | `blocked`

---

## Log

Append-only, newest last. One entry per step started and finished. Timestamps are local
(the machine runs UTC+00:00 per the trainer logs).

### 2026-08-10 — Phase 0a — journal created — done
Created `paper/` and this file. `paper/` also holds `results.csv` and figure outputs.

### 2026-08-10 10:30 — Phase 0 — trainer edits — done
`src/training/trainer.py`: `backward()` moved inside the unroll loop; new optional
`micro_batch_size` in the trainer config (defaults to `batch_size`, so every existing
config is unaffected). `total_loss` became a python float, so the two `total_loss.item()`
log/CSV sites were updated.

### 2026-08-10 10:33 — Phase 0 — gradient equivalence — done PASS
Same seed, same batch, N=49 production recipe. OLD = sum-then-backward, NEW = per-step
backward with micro-batching.

| variant | loss | d(loss) | max abs d(grad) | relative | peak VRAM |
|---|---|---|---|---|---|
| OLD | 0.42122406 | — | — | — | 4.26 GB |
| NEW micro=32 | 0.42122405 | 7.5e-09 | 4.77e-07 | 1.11e-07 | 0.97 GB |
| NEW micro=16 | 0.42122403 | 2.6e-08 | 1.91e-06 | 4.44e-07 | 0.51 GB |
| NEW micro=8 | 0.42122403 | 2.9e-08 | 1.55e-06 | 3.61e-07 | 0.27 GB |

Agreement at float32 round-off, as predicted (summation order differs, so bit-identity was
never expected). Per-step backward alone cuts peak memory 4.4x.

### 2026-08-10 10:35 — Phase 0 — smoke test — done PASS
`src/configs/training/smoke_test/*` runs clean on CPU, mean|KG| 0.186 -> 0.110.

### 2026-08-10 10:36-10:38 — Phase 0 — throughput after the fix — done
40-iteration probes, production recipe, RTX 4080 SUPER 16 GB. **The N=100 and N=196 walls
are gone.**

| N | micro_batch | s/iter before | s/iter after | 10k iters |
|---|---|---|---|---|
| 100 | 32 (none needed) | 4.575 (VRAM spill) | **0.248** | ~41 min |
| 196 | 8 | OOM | 0.978 | ~2.7 h |
| 196 | 16 | OOM | **0.925** | ~2.6 h |

Decisions: N=100 needs no accumulation (`micro_batch_size` unset); N=196 uses
`micro_batch_size: 16`. Val-loss chunking holds at N=196 (it runs under `no_grad`, and the
256-cloud validation set divides evenly by 32). Probe run dirs deleted.

### 2026-08-10 10:39 — Phase 0 — N=49 production re-run — running
Confirms the edited trainer reproduces the shipped checkpoint's metrics.
```
.venv\Scripts\python.exe src/training/trainer.py ^
  --train-config   src/configs/training/trainer/train_config_sph_adamw.yaml ^
  --dataset-config src/configs/training/dataset/dataset_config_sph.yaml ^
  --loss-config    src/configs/training/loss/loss_config_rdsph_lam3_0p27.yaml ^
  --model-config   src/configs/training/model/model_config_12_sph_L4.yaml
```
Target: viol_reduction 89.6%, illegal 0.39%, mean|KG| 0.0501 (K=1) / 0.0220 (K=5).

### 2026-08-10 10:54 — Phase 0 — N=49 production re-run — done
`train_run_2026-08-10_10-40-18`, 350,594 params. Final (iter 10000, K=5):
viol_red 85.7%, illegal 0.53%, mean|KG| 0.0228. |KG| matches the recorded 0.0220; the
viol_red gap vs 89.6% is within this metric's own noise (the SAME run logs 88.4% at iter
9900 and 85.7% at 10000, because the block is computed on a fresh random training batch).
NOT yet settled deterministically — pending `score_arm.py` vs the shipped checkpoint.
`tests/test_wholecloud.py` PASSES bit-exact after the trainer change.

### 2026-08-10 11:01-11:18 — Phase 2 — N=49 architecture arms
All arms share `loss_config_rdsph_lam3_0p27.yaml` — the baselines are trained WITH the
KG term, so the claim is "same physics-informed objective, architecture decides whether
it is reachable", not "physics-informed vs vanilla".

| arm | run dir | viol_red K5 | illegal% | \|KG\| K5 | isolated s/iter |
|---|---|---|---|---|---|
| model12 noise0.6 | `train_run_2026-08-10_10-40-18` | 85.7% | 0.53 | 0.0228 | 0.073 |
| dgcnn noise0.6 | `train_run_2026-08-10_10-51-22` | 79.4% | 0.77 | 0.0310 | 0.047 |
| pointnet noise0.6 | `train_run_2026-08-10_11-01-18` | -0.1% | 3.73 | 0.2548 | 0.025 |
| model12 noise1.0 | `train_run_2026-08-10_11-05-02` | 80.5% | 0.84 | 0.0376 | (concurrent) |
| gns noise0.6 | `train_run_2026-08-10_11-16-59` | running | | | 0.223 |
| dgcnn noise1.0 | `train_run_2026-08-10_11-18-32` | running | | | |

s/iter is only valid from the ISOLATED probes; runs launched concurrently log inflated
values (model12 0.073 -> 0.167 alongside GNS).

### 2026-08-10 11:10-11:25 — Phase 2 — GNS and PointNet++ added (user request)
Two more baselines, both parameter-matched and smoke-tested through the trainer:
- **GNS-style** `models/architectures/gns/gns.py`, hidden_dim 107 -> 347,966 (-0.75%).
  Radius graph at the same cutoff, persistent edge latents, unweighted sum. Differs from
  model12 in exactly two mechanisms, so model12 = "GNS + SPH kernel weighting".
- **PointNet++** `models/architectures/pointnet2/pointnet2.py`, hidden_dim 186 ->
  350,240 (-0.10%). FPS + ball query + max-pool, twice, then 3-NN interpolation back up.
  The only hierarchical arm. Hierarchy is 49->12->3 at N=49, 196->49->12 at N=196.
Measured radius-graph degree at cutoff 0.286 on a 7x7 lattice: mean 11.6, min 7, max 19 —
which is why DGCNN uses k=12 and PointNet++ nsample=12.
Report written: `paper/ARCHITECTURES.md`.

### 2026-08-10 11:35 — Phase 0 — scorer verification vs shipped checkpoint — done PASS
`score_arm.py` on identical fixed clouds (seed 1234, 64 clouds, periodic metrics), both
using the same run's config snapshot:

| checkpoint | viol_red | illegal% | \|KG\| |
|---|---|---|---|
| new trainer (`train_run_2026-08-10_10-40-18/model_best.pt`) | 82.9% | 4.0 -> 0.7 | 0.0216 |
| shipped (`src/models/weights/model12_sph_l4.pt`) | 84.3% | 4.0 -> 0.6 | 0.0207 |

Phase 0 SETTLED: the gradient test already proved mathematical equivalence (rel 1e-7);
this confirms the end-to-end result is within run-to-run variation. Raw output kept at
`paper/phase0_check.csv`.

### 2026-08-10 11:35 — METHODOLOGY: seed variance is unquantified and the trainer sets no seed
`trainer.py` never seeds torch, so model init (and thus the final checkpoint) differs run
to run. The two rows above are the SAME recipe trained twice, giving a first estimate of
that spread: **viol_red +-1.4 points, |KG| +-4% relative**.

Consequences for the architecture exhibit, with n=1 per arm:
- model12 vs DGCNN viol_red gap = 6.3 points ~ 4.5x the observed seed spread — likely real.
- model12 vs DGCNN |KG| gap = 36% ~ 9x the spread — solid.
- Any future gap under ~2 points of viol_red or ~10% of |KG| is NOT separable from seed
  noise and must not be claimed.

Options: (a) report the measured spread as an uncertainty note and only claim gaps that
clear it — free; (b) 3 seeds per arm — triples the cost. Currently doing (a); flagged to
the user.

### 2026-08-10 11:26 — Phase 2b — packing sweep configs generated
`vibecoding/sweeps/make_packing_configs.py` -> `src/configs/training/ablations/packing/`.
Four rungs, rd/spacing 0.80/0.90/0.95/1.00 at N=49, all lengths pinned to rd so only
constraint rigidity varies. `max_displacement` deliberately NOT scaled, so model configs
are reused unchanged and the architectures are byte-identical across rungs.

| rd/s | rd | lattice jitter | input illegal% | \|KG\| clean | lambda3 |
|---|---|---|---|---|---|
| 0.80 | 0.114286 | 0.01429 | 1.8% | 0.0444 | 0.1795 |
| 0.90 | 0.128571 | 0.00714 | 2.7% | 0.0222 | 0.2274 |
| 0.95 | 0.135714 | 0.00357 | 3.4% | 0.0111 | 0.2532 |
| 1.00 | 0.142857 | **0.00000** | 4.6% | **0.00000** | 0.2814 |

At rd/s = 1.00 the jitter is exactly zero: the only feasible configuration is the exact
lattice, whose KG is exactly 0. Maximum coordination demand, minimum geometric freedom.

---

## Live finding — 2026-08-10 11:55: TWO arms collapse to zero displacement

Both `dgcnn @ noise 1.0` (finished, 10000 iters) and `model12 @ lambda3=0` (3200 iters)
emit **exactly zero displacement**: |KG| out equals |KG| in to four decimals, violation
reduction 0.0%, illegal% unchanged or drifting up.

For lambda3=0 this is NOT the global optimum. With
`L = lambda1*mean(relu(rd-d)) + lambda2*mean|disp|`, doing nothing costs ~0.0096
(the input illegality) while correcting costs ~0.0005. Correcting is ~20x better, so
zero displacement is a local minimum the optimiser fails to escape.

Working hypothesis, and it is a strong result if it survives: **the KG term is a DENSE
training signal and the violation term is a SPARSE one.** Every particle has a nonzero
KG residual whenever its neighbourhood is asymmetric, whereas the violation term is a
mean over all N^2 pairs of which only O(N) violate — diluting its gradient by ~1/N.
So the physics term may not merely improve final quality; it may be what makes the
optimisation work at all. That reframes "physics-informed" from a regulariser to an
optimisation enabler.

**NOT YET CLAIMABLE — needs seed replication.** A collapse is a binary, catastrophic
outcome, which is far more likely to be an initialisation artifact than a small
quantitative gap is. The trainer sets no torch seed and both arms are n=1. Distinguish:
- quantitative gaps (model12 vs DGCNN etc.) — the measured spread (+-1.4 pts, +-10% KG)
  is a sufficient uncertainty statement;
- **collapse/no-collapse outcomes — require >=3 seeds** before any claim.

Proposed: 3 seeds each for `lambda3=0` and `dgcnn @ noise 1.0` (~1.5 h total).

## Decisions & deviations

### 2026-08-10 — scorer uses periodic distances; the trainer's inline eval does not
`trainer.py`'s eval block measures with `torch.cdist` (non-periodic) while the loss uses
`_pbc_rel` (minimum image). On a torus `cdist` inflates nn distances and misses wrap-seam
violations. `score_arm.py` is periodic throughout, so its numbers are **not** directly
comparable to trainer log lines. Paper numbers come from the scorer. The trainer's block
was left alone — changing it would invalidate comparison with every historical run.

### 2026-08-10 — BLOCKER for the small-N ladder rungs: KG minimum-image truncation
The quintic kernel's compact support is `3h = 6*dx`; minimum image is exact only when that
fits in `box/2`. Measured truncation error against an explicit all-periodic-images sum, on
disordered clouds:

| N | dx | support | support <= box/2 | \|KG\| min-image | \|KG\| all-images | error |
|---|---|---|---|---|---|---|
| 16 | 0.2500 | 1.500 | no | 0.26763 | 0.06782 | **295%** |
| 25 | 0.2000 | 1.200 | no | 0.21183 | 0.15204 | **39%** |
| 49 | 0.1429 | 0.857 | no | 0.25150 | 0.24816 | 1.3% |
| 100 | 0.1000 | 0.600 | no | 0.29069 | 0.29030 | 0.1% |
| 196 | 0.0714 | 0.429 | yes | 0.51987 | 0.51987 | 0.0% |
| 400 | 0.0500 | 0.300 | yes | — | — | 0.0% |
| 2500 | 0.0200 | 0.120 | yes | — | — | 0.0% |

The formal `support <= box/2` rule is conservative — the kernel derivative decays, so N=49
is off by only 1.3% and N=100 by 0.1%. **The shipped N=49 checkpoint and the N=2500
deployment are both fine.** But at N=16 and N=25 the loss term is not the SPH kernel
gradient, which breaks the cardinality ablation's premise that only cardinality varies.

**RESOLVED by the user (2026-08-10):** ladder = **49 / 100 / 196**; Phase 2 baselines at
**N=49 and N=100**. N=16 and N=25 are dropped entirely. The min-image KG loss is kept as
shipped — changing it would move the objective away from the one the sim-validated
checkpoint was trained on.

### 2026-08-10 — Phase 2 cost is higher than the plan's estimate
The plan quoted ~1.5 h for the baselines assuming N in {25, 49}. With N in {49, 100} and
measured per-architecture speeds (model12 0.073, DGCNN 0.108, PointNet 0.047 s/iter at
N=49; x4.16 at N=100) the full 3 arch x 2 N x 2 noise grid is ~6.4 h, not 1.5 h. The N=49
block is 64 min; the N=100 block is ~5.3 h. Flagged to the user; N=49 block launched first
since it is unambiguous.

---

## Numbers ledger

Every value destined for the paper, with the artifact it came from. A figure must not quote
a number absent from this table.

| Value | Number | Source artifact | Established |
|---|---|---|---|
| Real-trajectory KG, raw (t>=300 mean) | 0.3257 | `artifacts/inference/experiments/sph_tv/runs/kg_sweep_2026-07-21_17-32-40/report.txt` | pre-existing |
| Real-trajectory KG, TV baseline | 0.2735 | same | pre-existing |
| Real-trajectory KG, model9 K=5 | 1.2775 | same | pre-existing |
| Real-trajectory KG, model12 wholecloud k=5 | 0.1279 | same | pre-existing |
| KG floor (model12_wc min / p10, t>=300) | 0.1109 / 0.1135 | same | pre-existing |
| mean_nn after correction (model12_wc) | 0.01950 | same | pre-existing |
| illegal% after correction (model12_wc) | 82.7% | same | pre-existing |
| model12 L4 hd128 parameter count | 350,594 | measured from checkpoint architecture | 2026-08-10 |
| Baseline train cost, N=49 B=32 unroll5 | 73 ms/iter, 4.26 GB peak | measured, RTX 4080 SUPER 16 GB | 2026-08-10 |
| Train cost N=100 (pre-fix) | 4575 ms/iter, 17.24 GB peak (spills VRAM) | same | 2026-08-10 |
| Train cost N=196 (pre-fix) | OOM, ~68 GB projected | same | 2026-08-10 |
| symmetry:illegality ratio @ N=49/100/196/400 | 2.67 / 10.37 / 38.81 / 167.4 | measured, 64 clouds @ noise 0.6*rd | 2026-08-10 |
| lambda3 for matched trade-off @ N=49/100/196 | 0.27 / 0.070 / 0.019 | derived from the ratio above | 2026-08-10 |
| N values that break datagen | 50, 200 (not perfect squares -> dart-throwing jams) | measured, `RuntimeError` | 2026-08-10 |
