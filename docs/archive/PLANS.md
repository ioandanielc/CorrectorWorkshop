# CorrectorWorkshop — Research Plan

## What this is

A PyTorch framework for a **Poisson disk corrector**: given a 2D point cloud that violates the minimum-distance constraint `rd`, the model predicts per-point displacement vectors to restore validity. Architecture: violation-weighted edge network — each point aggregates messages from its violating neighbours only, weighted by violation depth. Trained with iterative unrolling (K=3 passes per batch), deployed single-pass (K=1).

**Deployment target:** N≈2500 points on a periodic (toroidal) domain, noise comparable to training distribution.

---

## Ablation map

Forward ablation: start from the simplest baseline, add one component at a time, each motivated by a specific failure of the previous version.

### Completed

| Component | Baseline (without) | Why added | Outcome |
|---|---|---|---|
| Violation-weighted aggregation | model6: uniform sum over all N−1 neighbours | Uniform sum dilutes the violating signal 50×; gradient too small on the pair that matters | Surgical corrections, median displacement = 0.000×rd, +48% efficiency; exposes single-pass cluster ceiling |
| K=3 iterative unrolling | K=1 single pass | One pass cannot resolve 3+ mutually-violating clusters without moving legal neighbours | Violation reduction ×1→×3: 70.7%→94.6%; K=3 advantage grows with packedness |
| Output clamping (tanh × max_disp) | model7/8: unbounded output | Packed training showed runaway displacements up to 3.4×rd cascading new violations | Max displacement 3.4→0.96×rd; stable packed training |
| 3-layer edge MLP | 2-layer (model7/8) | 2-layer undersells multi-point cluster geometry | Improved efficiency and packed performance |
| Packed training data (p=0.5) | Sparse N=50 (p≈0.11) | Sparse clouds have slack — model never sees the hard failure mode | K=3 advantage 2× larger in packed regime; validates training-deployment density match |
| Packedness sweep (p=0.75, 0.90) | Only tested p=0.5 | Unknown where model breaks above p=0.5 | Performance drops sharply p=0.5→0.75 (K=1: 78.7%→54.8%), then plateaus at p=0.90 (55.2%). K=3 advantage keeps growing. Capacity likely the bottleneck, not the architecture. |

### Planned / running

| Component | Baseline | Why testing | What we'll measure |
|---|---|---|---|
| hidden_dim ∈ {128, 256, 512} × edge_depth ∈ {3, 4} *(running)* | hd=128, d=3 | Plateau at p=0.75–0.90 may reflect capacity limit: more violating neighbours → richer signal → needs more width to interpret | Violation reduction and efficiency at p=0.90; which axis (width vs depth) gives the most gain per parameter |
| Self-feature (embed x_i) | No positional signal | Near-boundary points have asymmetric neighbourhoods the model cannot distinguish from interior | Boundary-region improvement; if negligible it's a boundary workaround, not a genuine gain |
| Sparse radius graph + MIC (inference, 11a) | Dense O(N²) | Real data has N=2500 — dense is infeasible (3.2 GB/cloud); model logic is already local so no retraining required | Scale to N=2500 with same weights; MIC handles toroidal boundary transparently |
| 5×5 domain tiling + N≈100 model (inference, 11b) | Single large cloud | Domain decomposition: partition into 25 cells of ~100 pts each, apply small model in parallel; overlap halo of size r_d handles cross-cell violations | Compare with 11a on real data; tiling advantage is parallelism, 11a advantage is exactness |

---

## Completed

### Step 1 — Baseline (model6, sparse)
Uniform-push edge network, N=50, rd=0.05, noise σ ∈ [0, 0.03]. Converges to 0.23% illegal pairs but applies large displacement uniformly to all points including legal ones. NN drift +31% at ×5 passes. Works because sparse clouds have slack; fails when slack is gone.

### Step 2 — Violation-weighted aggregation (model7)
**Motivation:** model6 dilutes the violating signal across 49 legal neighbours; in packed geometry that shoving cascades new violations.
Replaced uniform sum with violation-depth-weighted aggregation — legal neighbours contribute zero. Surgical corrections (median displacement = 0.000×rd), +48% efficiency. Single-pass clearance lower: exposes that 3+ point clusters need multiple passes.

### Step 3 — Iterative unrolling (K=3)
**Motivation:** one pass cannot resolve multi-point clusters; K passes let precision compound — each step sees updated geometry.
K=3 training with `.detach()` between steps. Violation reduction ×3: 94.6%. NN drift ×3: +11.5% above natural vs model6's +31.1% at ×5. Eval reports K=1 (deploy intent) and K=3 (train signal) side by side throughout.

### Step 4 — Deeper MLP + output clamping (model9)
**Motivation:** packed training showed runaway displacements (3.4×rd); 2-layer MLP undersells cluster geometry.
Third hidden layer + tanh-clamped output (max = 1.2×rd). Max displacement 3.4→0.96×rd; stable packed training.

### Step 5 — Packed regime (packedness=0.5, N≈231)
**Motivation:** real application is packed — N=50 never exposes the hard failure mode.
`PackedPoissonDiskDataset` parameterised by packedness fraction of triangular-lattice max. At p=0.5, N≈231, ~77 violations/cloud at input. Model9 K=1: **78.7%** violation reduction; K=3: **96.0%**. K=3 advantage 2× larger than in sparse.

### Step 6 — Model comparison study
Benchmarked all models (6, 7, 8, 9) on held-out sets, sparse and packed. Model6: best single-pass raw clearance (90.9%) but over-spreads (+31% NN drift) and would cascade violations in packed geometry. Model9: best surgical efficiency (0.0288) and bounded displacement.

### Step 7 — Packedness sweep (p=0.75, 0.90)
**Motivation:** model9 confirmed at p=0.5; unknown where it breaks.
Trained model9 at packedness ∈ {0.75, 0.90}, N≈{346, 416}.

| packedness | N | K=1 viol reduction | K=3 viol reduction | corr. eff K=1 |
|---|---|---|---|---|
| 0.50 *(prev)* | 231 | 78.7% | 96.0% | — |
| 0.75 | 346 | 54.8% | 82.2% | 84.1% of ceil |
| 0.90 | 416 | 55.2% | 79.1% | 83.4% of ceil |

Key findings: sharp drop p=0.5→0.75, then plateau. Correction efficiency stays high (~84%) — the model is still operating precisely, but the problem is harder. K=3 advantage keeps growing with density. p=0.90 is the target for the capacity grid.

---

## Running now

### Step 8 — Width/depth capacity grid at p=0.90
**Motivation:** the plateau at p=0.75–0.90 may reflect that hidden_dim=128 lacks the capacity to interpret the richer aggregated signal from more violating neighbours per point. Edge depth may also matter for complex cluster geometry.

4 variants, all at packedness=0.90 (N≈416), K=3:

| Variant | hidden_dim | edge_depth | params |
|---|---|---|---|
| hd128_d3 | 128 | 3 | 51K (baseline) |
| hd256_d3 | 256 | 3 | ~200K |
| hd512_d3 | 512 | 3 | ~780K |
| hd256_d4 | 256 | 4 | ~265K |

Running sequentially. `edge_depth` is now a config param in model9 (default=3, backward compatible).

---

## Next

### Step 9 — Self-feature
**Motivation:** no positional signal — near-boundary points have asymmetric neighbourhoods the model cannot distinguish from interior points.
Concatenate a learned embedding of `x_i` to the violation-aggregated representation before the output MLP.

### Step 10 — Final training run
Train the winning configuration (best width/depth × self-feature decision) at the hardest feasible packedness. Production model.

### Step 11 — Inference at N=2500: two approaches

Real data has N≈2500 on a toroidal domain. Dense O(N²) is infeasible (3.2 GB/cloud). Two complementary approaches:

#### 11a — Sparse radius graph + MIC wrapper
**No retraining required.** The model is purely local — violation-weighted aggregation already ignores pairs beyond rd. With cutoff 2×rd and N=2500, each point has ~10–20 edges; memory drops from 3.2 GB to ~13 MB per cloud. MIC at inference (`rel_pos -= round(rel_pos)`) makes any cross-boundary violation look geometrically identical to an interior violation — the model handles it correctly without modification. Post-correction modulo wraps positions back to [0,1]².

One new file: inference wrapper using `scatter_add`. No architectural change, no new training.

#### 11b — 5×5 domain tiling + N≈100 model
**Domain decomposition.** Partition [0,1]² into a 5×5 grid → ~100 points per cell → apply a small model trained at N≈100 (packedness ≈ 0.22, well within validated regime). 25 cells can run in parallel.

**Cross-boundary violations** (pairs that straddle a cell edge) need one of:
- **Overlap padding (preferred):** extend each cell outward by r_d; include ~10–20 neighbour points from adjacent cells in the model input; apply updates only to core points, discard halo updates. Every violation is seen by at least one cell.
- **Random tiling:** shift the grid randomly each pass. Over K passes, boundaries fall at different places — every point is interior in at least one pass.

**Normalization:** cells must be fed to the model in absolute coordinate space (center-only, no scale change), so that r_d stays at the trained value. Re-scaling a 0.2×0.2 cell to [0,1]² would inflate r_d to 0.25 — physically impossible at N≈100.

Training: one new config at N≈100, rd=0.05, packedness≈0.22 — straightforward.

| | **11a sparse+MIC** | **11b tiling** |
|---|---|---|
| Retraining | none | new N=100 config |
| Boundary violations | exact (MIC lossless) | need overlap padding |
| Memory | ~13 MB/cloud | tiny per cell |
| Parallelism | single pass | 25 cells parallel |

Both produce one new file. Run both on the real data benchmark and compare.

---

## Next — with colleague

### Step 12 — Real data benchmark
**The decisive test.** Model6 wins on raw single-pass clearance; model9 wins on surgical precision. On real packed data (N≈2500, toroidal, comparable noise), model6's global push will cascade new violations; model9's violation-weighted approach is the hypothesis. Apply via both Step 11 wrappers.

Evaluation protocol (no ground truth clean cloud):
- **viol/cloud** before vs after — primary metric
- **NN drift** — mean nearest-neighbour / rd must stay close to 1.0
- **Cascade check** — new violations introduced vs resolved; net = resolved − introduced
- **Zero-violation stability** — clean real clouds must produce near-zero displacement

Implementation: infer rd from 5th-percentile NN distance on clean examples, normalise to [0,1]², run all model checkpoints through both wrappers, produce comparison table and visualisations in `analysis/eval_real.py`.
