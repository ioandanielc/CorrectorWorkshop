# CorrectorWorkshop — Research Plan

## What this is

A PyTorch framework for a **Poisson disk corrector**: given a 2D point cloud that violates the minimum-distance constraint `rd`, the model predicts per-point displacements to produce a valid cloud. Architecture: violation-weighted edge network — each point aggregates messages from its violating neighbours only, weighted by violation depth. Trained with iterative unrolling (K=3 passes per batch), deployed single-pass (K=1).

---

## Ablation map

This project is a **forward ablation** (reverse of the usual): we start from the simplest possible baseline and add one component at a time, each motivated by a specific failure of the previous version. Every architectural decision has a controlled comparison.

### Completed

| Component | Baseline (without) | Why added | Outcome |
|---|---|---|---|
| Violation-weighted aggregation | model6: uniform sum over all N−1 neighbours | Uniform sum dilutes the single violating-neighbour signal across 49 legal ones; gradient is 50× too small on the pair that actually matters | model7: surgical corrections, median displacement = 0.000×rd, +48% efficiency; but single-pass clearance lower — exposes the multi-point cluster ceiling |
| K=3 iterative unrolling | K=1 single pass (always compared side-by-side) | One pass cannot resolve 3+ mutually-violating points without moving legal neighbours; K passes compound precision | Violation reduction ×1→×3: 70.7% → 94.6%; K=3 advantage grows with packedness — at p=0.5 it halves residual violations vs K=1 |
| Output clamping (tanh × max_disp) | model7/8: unbounded linear output | Packed training showed runaway displacements up to 3.4×rd, cascading new violations in early iterations | model9: max displacement bounded to 0.96×rd; packed training stable from the start |
| 3-layer edge MLP | 2-layer (model7/8) | 2-layer MLP undersells the geometry of multi-point conflict clusters | Improved efficiency and packed-regime performance; marginal over model8 GELU but consistent |
| Packed training data (p=0.5) | Sparse N=50 (p≈0.11) | Sparse clouds have slack — model never sees the failure mode it needs to solve | K=3 advantage 2× larger in packed regime; confirms that training distribution must match deployment density |

### Planned

| Component | Baseline (without) | Why testing | What we'll measure |
|---|---|---|---|
| hidden_dim ∈ {128, 256, 512} | 128 (current) | High packedness → more violating neighbours per point → richer aggregated signal that may need more capacity to interpret | Violation reduction and efficiency at hardest packedness; plateau identifies the capacity ceiling |
| edge_depth ∈ {3, 4} | 3 (current) | Deeper MLP = higher-order reasoning about local cluster geometry; diminishing returns expected but one data point needed | Marginal gain vs compute; determines whether width or depth is the right scaling axis |
| Self-feature (embed x_i) | No positional signal | Near-boundary points have legal neighbours on one side only — asymmetric geometry the model cannot currently distinguish from interior points | Improvement concentrated at boundary; if negligible after PBC (toroidal domain) then the feature is a boundary workaround, not a genuine gain |
| Periodic boundary conditions | Hard domain boundary [0,1]² | Boundary artefacts conflate "near-boundary" with "genuinely hard geometry"; toroidal domain makes every point equivalent | Reduction in boundary-region errors; cleaner test of model capacity independent of domain edge effects |

---

## Completed

### Step 1 — Baseline (model6, sparse)
Uniform-push edge network on sparse clouds (N=50, rd=0.05, noise σ ∈ [0, 0.03]). Converges to 0.23% illegal pairs but applies large displacement uniformly to all points, including legal ones. NN drift: +31% above natural at ×5 passes. Works because sparse clouds have slack to push into.

### Step 2 — Violation-weighted aggregation (model7)
**Motivation:** model6 dilutes the signal from violating neighbours across all 49 legal ones; in a packed cloud that shoving creates new violations.
Replaced uniform sum with violation-depth-weighted aggregation — legal neighbours contribute zero. Result: surgical corrections (median displacement = 0.000×rd), 48% better efficiency than model6, but single-pass clearance is lower because one pass cannot resolve 3+ point clusters without disturbing legal neighbours.

### Step 3 — Iterative unrolling (K=3)
**Motivation:** model7 cannot resolve multi-point clusters in one pass; giving it K passes lets precision compound — each step sees updated geometry and tightens further.
K=3 training with `.detach()` between steps. Violation reduction ×3: 94.6%. NN drift at ×3: 11.5% above natural vs model6's 31.1% at ×5. Eval always reports K=1 (deploy intent) and K=3 (train signal) side by side.

### Step 4 — Deeper MLP + output clamping (model9)
**Motivation:** at high density, early training showed runaway displacements (up to 3.4×rd) cascading into new violations; a 2-layer edge MLP undersells complex multi-point cluster geometry.
Added a third hidden layer and hard-bounded output via tanh (max_displacement = 1.2×rd). Max displacement dropped from 3.4×rd to 0.96×rd; efficiency improved further.

### Step 5 — Packed regime (packedness=0.5, N≈231)
**Motivation:** real application is densely packed — N=50 has too much slack to expose the hard failure modes.
Introduced `PackedPoissonDiskDataset` parameterised by packedness (fraction of triangular-lattice max density). At packedness=0.5, N≈231 with ~77 violations/cloud at input. Model9 K=1: **78.7% violation reduction**; K=3: **96.0%**. The K=3 advantage is far larger than in sparse — confirming that multi-pass matters most precisely when the geometry is hardest.

### Step 6 — Model comparison study
Benchmarked all models (6, 7, 8, 9) on held-out sets, sparse and packed. Key finding: model6 achieves the best single-pass raw clearance (90.9%) but over-spreads aggressively and would cascade new violations in truly packed clouds. Model9 has the best surgical efficiency and bounded displacement — the right properties for the packed regime.

---

## Today

### Step 7 — Packedness sweep *(running)*
**Motivation:** model9 is confirmed at packedness=0.5; unknown where it breaks down.
Training model9 at packedness ∈ {0.75, 0.90, 1.00} (N≈347, 416, 462 — approaching the physical limit of ~462 points at rd=0.05). Goal: identify the packedness level where violation reduction plateaus.

### Step 8 — Width/depth capacity grid *(tonight)*
**Motivation:** at high packedness each point has more violating neighbours and more complex conflict clusters; the current model (hidden_dim=128, 3-layer MLP) may lack capacity.
Grid over hidden_dim ∈ {128, 256, 512} × edge_depth ∈ {3, 4} at the breaking-point packedness from Step 7.

### Step 9 — Self-feature
**Motivation:** the model currently has no "where am I in the domain?" signal. Near-boundary points have legal neighbours on one side only, which changes the correction geometry.
Concatenate a small learned embedding of `x_i` to the violation-aggregated representation before the output MLP.

### Step 10 — Final training run
Train the winning configuration (best width/depth × self-feature decision) at the hardest feasible packedness. This is the production model.

### Step 11 — Periodic boundary conditions
Remove domain-boundary artefacts by switching to a toroidal domain: minimum-image-convention (MIC) wrapping in the model forward pass, post-displacement modulo, custom toroidal Poisson disk sampler. The physically correct setting for materials / MD applications where every point should be geometrically equivalent.

---

## Next — with colleague

### Step 12 — Real data benchmark
**The decisive test.** Synthetic experiments show the trade-off clearly — model6 wins on raw single-pass clearance, model9 wins on surgical precision and displacement conservation. On real packed data with no slack, model6's global push is expected to cascade new violations; model9's violation-weighted approach is the hypothesis to confirm.

Evaluation protocol (no ground truth clean cloud available):
- **viol/cloud** before vs after — primary metric
- **NN drift** — mean nearest-neighbour distance must stay close to rd, not balloon
- **Cascade check** — new violations introduced vs resolved (net reduction)
- **Zero-violation stability** — feed clean real clouds, model must output near-zero displacement

Implementation: normalise real clouds to [0,1]², infer rd from 5th-percentile NN distance on clean examples, run all model checkpoints, produce comparison table and visualisations.

