# Results — what the ablation suite established

Consolidated from `paper/results.csv` (scored, periodic metrics, fixed evaluation
clouds), `paper/JOURNAL.md` (run log), and the pre-existing sim-validated trajectory
artifacts. Every number here traces to an artifact; nothing is quoted from a training
log's per-iteration block, which is computed on a fresh random batch and is far noisier
than it looks.

**Protocol.** Every architecture trains under one identical recipe — same loss
(`rdsph`, lambda1=7.14, lambda2=0.0149, lambda3=0.27), same data, same optimiser and
schedule, same 10k iterations, same 5-step unrolling, same bounded
`tanh * max_displacement` head, parameter counts matched within 0.8%. No per-architecture
tuning, model12 included. The baselines are trained **with** the physics-informed KG term.

**Uncertainty.** Two runs of the identical recipe differ by ±1.4 points of violation
reduction and ±10% relative on |KG|. Every arm is n=1 unless stated. Gaps below that are
not claimed.

---

## 1. The deployment result (the paper's spine)

Real SPH trajectory, N=2500, disordered regime t >= 300. An actual SPH re-simulation from
model12-corrected start states was validated end-to-end (2026-07-15).

| series | mean \|KG\| | mean nn | nn CV |
|---|---|---|---|
| raw (no correction) | 0.326 | 0.0146 | 18.7% |
| model9 (prior work, no symmetry term) | 1.278 | 0.0178 | — |
| Transport Velocity (classical baseline) | 0.274 | 0.0168 | 9.9% |
| GNS, production training | 0.242 | 0.0186 | — |
| GNS, best training | 0.132 | 0.0197 | — |
| **model12, production training** | **0.127** | 0.0195 | **2.8%** |
| **model12, best training** | **0.123** | 0.0195 | — |

model12 is best or tied-best in every comparison, ~90% better than GNS at matched
production settings, and 2.2x better than the classical TV baseline. DGCNN and PointNet
cannot run here at all — no sparse path.

**Structural regularity** is the measure that matches what the figures show: nn CV, the
spread of neighbour spacing relative to its mean. model12 is 3.5x more uniform than TV
and 6.7x more uniform than raw. For an SPH restart this is what "well conditioned" means.

Caveat: model12's 7.2% edge over GNS at *best* settings sits inside the ±10% seed spread,
so "model12 >= GNS" is solid while "model12 > GNS at best settings" would need 3 seeds.
The production-settings gap (90%) is far outside it and stands.

## 2. The physics term is an optimisation enabler, not a regulariser

The strongest ablation result. Both scored from `model_best.pt`:

| arm | viol_red | \|KG\| | nn CV |
|---|---|---|---|
| lambda3 = 0.27 (full) | **82.9%** | **0.0216** | **2.3%** |
| lambda3 = 0 (physics off) | 17.6% | 0.1407 | 13.5% |
| lambda1 = lambda2 = 0 (pure symmetry, 2026-07-22) | — | real-data KG 0.357, *worse than raw 0.326* | — |

Three independent signals:

1. **Removing the KG term costs 65 points of violation reduction** — and the ablated term
   is the *symmetry* term, yet the *violation* objective is what collapses.
2. **Training diverges without it.** Deterministic validation loss (fixed 256-cloud set):
   lambda3=0 peaked at **iteration 500** and never improved across the remaining 9500;
   lambda3=0.27 improved **monotonically to 10000**.
3. **The final iterate degenerates** into a saturated uniform translation.

Replicated 2/2 across seeds (third in progress). Signal 2 does not depend on the collapse
at all, which is why this result is robust where others are not.

Mechanism: the violation term is a mean over N^2 pairs of which only O(N) violate, so its
gradient is diluted ~1/N and shrinks further as easy violations clear, while the
displacement penalty is undiminished. The KG term supplies a dense per-particle gradient.

## 3. Architecture comparison, and where it reverses

N=49 synthetic, matched parameters, identical loss:

| noise 0.6*rd | viol_red | \|KG\| | nn CV | | noise 1.0*rd | viol_red | \|KG\| | nn CV |
|---|---|---|---|---|---|---|---|---|
| **model12** | **82.9%** | **0.0216** | **2.3%** | | **GNS** | **91.5%** | **0.0205** | **1.6%** |
| DGCNN | 77.1% | 0.0272 | 3.1% | | model12 | 82.0% | 0.0308 | 2.4% |
| GNS | 62.5% | 0.0365 | 3.9% | | | | | |
| PointNet | 0.2% | 0.2258 | 34.9% | | | | | |

**The synthetic benchmark misranks the architectures relative to deployment.** GNS wins
at N=49 high disorder; on the real N=2500 cloud model12 wins (§1). Anyone selecting on the
benchmark alone would have shipped the worse deployer.

**The mechanism was predicted before the trajectory run.** model12 normalises messages
per particle, so node states are independent of neighbour count and transfer from N=49 to
N=2500; GNS sums messages unnormalised, so magnitude scales with degree and does not
transfer. A mechanism predicted a priori then confirmed on real data is stronger evidence
than the benchmark win itself.

**Cost — quote the deployment number, not the training one.** The two differ a lot and
conflating them overstates the case:

| | model12 | GNS | ratio |
|---|---|---|---|
| training, N=49, dense | 0.073 s/iter | 0.223 s/iter | 3.0x |
| **inference, N=2500, sparse, k=5, CPU** | **0.189 s/step** | 0.213 s/step | **1.13x** |
| **inference, N=2500, sparse, k=5, CUDA** | **0.049 s/step** | 0.068 s/step | **1.39x** |

The 3x training gap comes from GNS materialising a persistent `(B,N,N,H)` edge tensor; on
the sparse path that becomes `(E,H)` and per-edge work is near-identical (GNS 321x107,
model12 260x128). **At deployment model12 is 1.1-1.4x cheaper, not 3x.** The efficiency
half of the substitution claim is real but modest — the accuracy and size-transfer
results (§1) are what carry it.

## 4. A failure mode the standard metrics cannot see

Degenerate arms emit a **saturated uniform translation** — every point moves by the same
vector at exactly `max_displacement`. Because every loss term depends only on relative
positions, |KG| and illegal% both read "unchanged" while the cloud slides a full box
length over five passes.

| arm | mean \|d\| | bulk share | local spread |
|---|---|---|---|
| model12 | 0.0372 | **4.0%** | 0.0372 |
| GNS | 0.0357 | 8.1% | 0.0356 |
| DGCNN | 0.0370 | 10.5% | 0.0368 |
| PointNet | 0.1680 | **100.0%** | 0.0001 |

Found only by decomposing the displacement field. It is a tanh saturation trap — once the
output head saturates, d(tanh)/dz ~ 0 and no gradient recovers it. Honest caveat: the
bounded head is our fairness choice applied to all five architectures, so the trap is
partly induced by it.

## 5. Methodological findings worth reporting

- **Collapse results are seed-fragile.** "DGCNN collapses on hard data" was retracted:
  the unseeded run sat at 0.0% for 10k iterations, seed 1 of the identical config reached
  64.8%. Any single collapsed run means nothing here.
- **lambda3 does not carry across cardinality.** |KG| ~ sqrt(N) while the illegality term
  ~ 1/N, so their ratio grows as N^2 and lambda3 must fall as 1/N^2 to hold the trade-off.
- **The KG term's minimum-image convention breaks at small N.** Support is 6*dx; measured
  truncation error is 295% at N=16, 39% at N=25, 1.3% at N=49, 0.1% at N=100. The shipped
  checkpoint and the N=2500 deployment are both fine; small-N ablations are not.
- **Structure has two opposite meanings.** Preserving the input arrangement (`knn_keep`)
  and producing a well-ordered output (nn CV) point different ways: model12 is *lowest* on
  the first among working models (it corrects most, so rewires most) and best on the
  second. For a restart the second is what matters. `knn_keep` is also confounded — models
  that do nothing score 0.98.

---

## Figures

`vibecoding/visualizations/corrector_side_by_side.py` -> `outputs/`
- `side_by_side_sph_t1000.png` — raw / TV / model12 at N=2500, full box plus zoom
- `side_by_side_n49.png` — one cloud through all four architectures, displacement arrows
- `displacement_decomposition.png` — bulk translation vs local restructuring

## Not done, and why

- **KG floor vs training cardinality** — the ladder (49/100/196) was cut for time. The
  floor is observed at 0.111 (min) / 0.1135 (p10) over 702 timesteps but its dependence on
  training N is untested. This is the honest limitations paragraph.
- **Packing sweep beyond rd/s = 1.00** — model12 works at maximum rigidity (58.3%
  viol_red, |KG| 0.0214). The comparative claim was dropped: after the seed-fragility
  finding it would need 3 seeds x 3 architectures x 4 rungs.
- **Seeds for the best-vs-best trajectory gap** (7.2%, inside the seed spread).
- **N=100 architecture block** — the N=49 comparison plus the trajectory result already
  cover the architecture question.
