# CorrectorWorkshop — Research Plan

## What this is

A PyTorch framework for a **Poisson disk corrector**: given a 2D point cloud that violates the minimum-distance constraint `rd`, the model predicts per-point displacement vectors to restore validity. Architecture: violation-weighted edge network — each point aggregates messages from its violating neighbours only, weighted by violation depth. Trained with iterative unrolling (K=3 passes per batch), deployed single-pass (K=1).

**Deployment target:** N≈2500 points on a periodic (toroidal) domain, noise comparable to training distribution.

---

## Ablation map

This project is a **forward ablation**: start from the simplest baseline and add one component at a time, each motivated by a specific failure of the previous version.

### Completed

| Component | Baseline (without) | Why added | Outcome |
|---|---|---|---|
| Violation-weighted aggregation | model6: uniform sum over all N−1 neighbours | Uniform sum dilutes the violating-neighbour signal 50×; gradient too small on the pair that matters | model7: surgical corrections, median displacement = 0.000×rd, +48% efficiency; exposes single-pass cluster ceiling |
| K=3 iterative unrolling | K=1 single pass | One pass cannot resolve 3+ mutually-violating clusters without moving legal neighbours | Violation reduction ×1→×3: 70.7%→94.6%; advantage scales with packedness |
| Output clamping (tanh × max_disp) | model7/8: unbounded output | Packed training showed runaway displacements up to 3.4×rd cascading new violations | model9: max displacement 0.96×rd; stable packed training |
| 3-layer edge MLP | 2-layer (model7/8) | 2-layer undersells multi-point cluster geometry | Improved efficiency and packed performance |
| Packed training data (p=0.5) | Sparse N=50 (p≈0.11) | Sparse clouds have slack — model never sees the hard failure mode | K=3 advantage 2× larger in packed regime |

### Planned

| Component | Baseline (without) | Why testing | What we'll measure |
|---|---|---|---|
| hidden_dim ∈ {128, 256, 512} | 128 (current) | High packedness → more violating neighbours → richer signal may need more capacity | Violation reduction at hardest packedness; plateau = capacity ceiling |
| edge_depth ∈ {3, 4} | 3 (current) | Deeper = higher-order cluster geometry reasoning | Marginal gain vs compute; width vs depth trade-off |
| Self-feature (embed x_i) | No positional signal | Near-boundary points have asymmetric neighbourhoods the model cannot distinguish from interior | Boundary-region improvement; tells us whether position context is a genuine gain or just a boundary workaround |
| Sparse radius graph + MIC (inference) | Dense O(N²) | Real data has N=2500 — dense computation is infeasible (3.2GB per cloud); model logic is already local so no retraining required | Scale to N=2500 with same weights; validate that inference-time MIC correctly handles cross-boundary violations |

---

## Completed

### Step 1 — Baseline (model6, sparse)
Uniform-push edge network, N=50, rd=0.05, noise σ ∈ [0, 0.03]. Converges to 0.23% illegal pairs but applies large displacement uniformly to all points including legal ones. NN drift +31% at ×5 passes. Works because sparse clouds have slack; fails when slack is gone.

### Step 2 — Violation-weighted aggregation (model7)
**Motivation:** model6 dilutes the violating signal across 49 legal neighbours. In packed geometry that shoving cascades new violations.
Replaced uniform sum with violation-depth-weighted aggregation — legal neighbours contribute zero. Surgical corrections (median displacement = 0.000×rd), +48% efficiency. Single-pass clearance lower: exposes that 3+ point clusters need multiple passes.

### Step 3 — Iterative unrolling (K=3)
**Motivation:** one pass cannot resolve multi-point clusters; K passes let precision compound — each step sees updated geometry.
K=3 training with `.detach()` between steps. Violation reduction ×3: 94.6%. NN drift ×3: +11.5% above natural vs model6's +31.1% at ×5. Eval reports K=1 (deploy intent) and K=3 (train signal) side by side throughout.

### Step 4 — Deeper MLP + output clamping (model9)
**Motivation:** packed training showed runaway displacements (3.4×rd) cascading new violations; 2-layer MLP undersells cluster geometry.
Third hidden layer + tanh-clamped output (max = 1.2×rd). Max displacement 3.4→0.96×rd; stable packed training.

### Step 5 — Packed regime (packedness=0.5, N≈231)
**Motivation:** real application is packed — N=50 never exposes the hard failure mode.
`PackedPoissonDiskDataset` parameterised by packedness fraction of triangular-lattice max. At p=0.5, N≈231, ~77 violations/cloud at input. Model9 K=1: **78.7%** violation reduction; K=3: **96.0%**. K=3 advantage is 2× larger than in sparse — multi-pass matters most when geometry is hardest.

### Step 6 — Model comparison study
Benchmarked all models (6, 7, 8, 9) on held-out sets, sparse and packed. Model6: best single-pass raw clearance (90.9%) but over-spreads (+31% NN drift) and would cascade violations in packed geometry. Model9: best surgical efficiency (0.0288) and bounded displacement — the right properties for the target regime.

---

## Today

### Step 7 — Packedness sweep *(running)*
**Motivation:** model9 confirmed at p=0.5; unknown where it breaks.
Training at packedness ∈ {0.75, 0.90, 1.00} (N≈347, 416, 462 — approaching the physical limit of ~462 at rd=0.05). Identifies the breaking point that targets the capacity test.

### Step 8 — Width/depth capacity grid *(tonight)*
**Motivation:** more violating neighbours per point at high packedness may need more capacity to interpret.
Grid over hidden_dim ∈ {128, 256, 512} × edge_depth ∈ {3, 4} at the hardest packedness from Step 7.

### Step 9 — Self-feature
**Motivation:** no positional signal — near-boundary points have asymmetric neighbourhoods the model cannot distinguish from interior points.
Concatenate a learned embedding of `x_i` to the violation-aggregated representation before the output MLP.

### Step 10 — Final training run
Train the winning configuration (best width/depth × self-feature decision) at the hardest feasible packedness. Production model.

### Step 11 — Sparse radius graph + MIC inference wrapper
**Motivation:** real data has N≈2500 on a toroidal domain. Dense O(N²) computation is infeasible (3.2 GB per cloud at hidden_dim=128). Noise level in real data is comparable to training — the corrector is not expected to get stuck.

**Key insight:** no retraining required. The model is purely local — violation-weighted aggregation already ignores pairs beyond rd. At N=2500 with cutoff 2×rd, each point has ~10–20 edges instead of 2,499; memory drops from 3.2 GB to ~13 MB per cloud. For the toroidal boundary: minimum image convention (MIC) at inference time (`rel_pos -= round(rel_pos)`) makes any cross-boundary violation look geometrically identical to an interior violation. The model, trained without PBC, handles it correctly without modification — it never sees the boundary. Post-correction modulo wraps positions back to [0,1]².

Implementation: one new file — an inference wrapper that builds a radius graph with MIC distances and runs the existing model via `scatter_add`. No architectural change, no new training.

---

## Next — with colleague

### Step 12 — Real data benchmark
**The decisive test.** Synthetic results establish the trade-off: model6 wins on raw single-pass clearance; model9 wins on surgical precision. On real packed data (N≈2500, toroidal, comparable noise), model6's global push will cascade new violations; model9's violation-weighted approach is the hypothesis.

Apply via the Step 11 sparse+MIC wrapper. Since noise is comparable to training, stuck/cascade failures are not expected — the corrector operates in the regime where it has been validated (shallow, isolated violations, sufficient geometric slack for local fixes).

Evaluation protocol (no ground truth clean cloud):
- **viol/cloud** before vs after — primary metric
- **NN drift** — mean nearest-neighbour / rd must stay close to 1.0
- **Cascade check** — new violations introduced vs resolved; net = resolved − introduced
- **Zero-violation stability** — clean real clouds must produce near-zero displacement

Implementation: infer rd from 5th-percentile NN distance on clean examples, normalise to [0,1]², run all model checkpoints through the sparse+MIC wrapper, produce comparison table and visualisations in `analysis/eval_real.py`.
