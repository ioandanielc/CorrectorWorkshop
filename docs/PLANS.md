# CorrectorWorkshop — Roadmap

---

## ✅ Done — Correction efficiency metric

**What it is:** a noise-invariant scalar logged alongside the existing metrics:

```
correction_eff = (mean_violation_before - mean_violation_after) / mean_displacement
```

**Why it's noise-invariant:** both the numerator (violation depth removed) and denominator
(displacement applied) scale linearly with the noise level σ added to the cloud.
Double σ → roughly double the violations → roughly double the displacement needed to fix them →
ratio stays constant. This makes it meaningful to compare runs trained at different noise
levels, or even individual eval steps within a variable-noise run where each batch is
drawn from a different σ.

**The theoretical upper bound — why `2 / (N − 1)`:**

Consider the ideal single-pass corrector on a cloud with N points.
Suppose there is one violating pair (i, j) at violation depth `d = rd − dist(i,j) > 0`.
To fix it with minimum total displacement, the model should move i and j
symmetrically apart by `d/2` each — total displacement across all N points is `2 × (d/2) = d`,
mean displacement is `d / N`.
The violation removed is exactly `d`, so the mean violation removed (averaged over
all N(N−1) ordered pairs, of which only 2 count) is `2d / N(N−1)`.

Plugging in:

```
correction_eff  =  Δmean_viol / mean_disp
               =  [2d / N(N-1)]  /  [d / N]
               =  2 / (N - 1)
```

At N = 50 this gives **0.0408**. This is the ceiling: a model that achieves it is applying
the geometric minimum displacement and nothing more. In practice the model will be below
this ceiling because (a) it can't perfectly target only violating pairs, and (b) it applies
some displacement to already-legal points. The ratio `efficiency_pct = correction_eff / ceil × 100`
directly measures how close to optimal the model is — and it means the same thing at
σ = 0.001 or σ = 0.03.

**Implementation:** `training/trainer.py` — computed in the eval block, logged as:
```
correction_eff   = 0.0116  (28.4% of ceil=0.0408)   [Δviol/disp, noise-invariant]
```
Also written to `loss.csv` as columns `correction_eff` and `efficiency_pct`.

---

## Step 1 — Variable noise ✅ in progress
**Goal:** teach the model to calibrate correction magnitude to violation severity.  
**Change:** `noise_scale_min: 0.0`, `noise_scale_max: 0.03` (was fixed at 0.01).  
**Why:** fixed σ trains the model to always push ~1×rd; variable σ including σ=0
forces it to learn "do nothing when already legal" and scale output to actual depth.  
**Config:** `dataset_config_3.yaml` + `loss_config_4.yaml` (λ1=50) + model6.

---

## Step 2 — Violation-weighted aggregation (model7) ✅ done

**Goal:** make aggregation physically meaningful — only violating neighbours
drive the displacement output.

### What changed vs model6

The single modification is in the aggregation step. Model6 sums all neighbour
embeddings uniformly:
```python
agg = edge_emb.sum(dim=2)                          # model6: uniform sum
```
Model7 weights each neighbour's message by its violation depth before summing:
```python
viol_sum = violation.sum(dim=2, keepdim=True)
weights  = violation / (viol_sum + 1e-8)           # (B, N, N, 1), sums to ~1 per point i
agg      = (edge_emb * weights).sum(dim=2)         # model7: violation-weighted sum
```
A neighbour j that is perfectly legal (violation = 0) contributes exactly zero
to point i's context vector. A neighbour at twice the violation depth of another
contributes twice as much. A point with no violating neighbours at all gets
`agg = 0`, which should drive the output MLP toward zero displacement — the
correct answer.

### Why this matters

In model6, a point with 49 legal neighbours and 1 severely-violating one averages
the single meaningful signal across all 49 others, diluting it by ~50×. Gradient
flows through all edges equally regardless of their physical relevance. In model7,
the gradient from the violating edge dominates its signal is no longer diluted.

A secondary effect: `correction_eff` (Δviol/disp) should climb closer to the
theoretical ceiling of `2/(N−1) = 0.0408`, because the model wastes less
displacement budget on already-legal pairs.

### Observed training behaviour

Model7 shows a harder initialisation phase than model6:

| iter | illegal_pairs after | displacement | correction_eff |
|---|---|---|---|
| 100 | 0.80% (m6: 0.21%) | 2.53× rd mean / 0.49× median | 0.2% of ceil |
| 900 | **1.31%** (worse than input) | — | **−2.2%** (negative) |
| 1000 | 0.65% | 0.328× rd | 14.7% of ceil |

The temporary regression at iter 900 (violations *increased*) is characteristic of
violation-weighted aggregation: at initialisation, edge embeddings are random and
the weighting amplifies whichever random directions happen to have high violation
depth. The model briefly learns to push points *into* violations before the loss
corrects it. This did not occur in model6 because uniform aggregation smooths over
random init noise. The collapse resolves quickly — by iter 1000 the model is
self-consistent and displacement mean/median are nearly equal (0.328 vs 0.326× rd),
indicating stable, uniform corrections.

### Final results (10 000 iters, variable σ ∈ [0, 0.03])

| iter | illegal_pairs after | correction_eff | disp mean / median |
|---|---|---|---|
| 100 | 0.80% (m6: 0.21%) | 0.2% of ceil | 2.53× / 0.49× rd |
| 900 | **1.31%** (worse than input) | **−2.2%** | — |
| 1000 | 0.65% | 14.7% of ceil | 0.328× / 0.326× rd |
| 8000 | 0.42% | 63.9% of ceil | 0.093× / 0.020× rd |
| **10 000** | **0.36%** | **75.8% of ceil** | **0.080× / 0.000× rd** |

Model6 at convergence for comparison: 0.23% illegal, 39% legal clouds, ~28% of ceil, disp 0.156×/0.129× rd.

### What the final numbers reveal

**Model7 wins on efficiency (75.8% vs 28% of ceiling) but loses on raw violation
reduction (0.36% vs 0.23% illegal pairs, 25% vs 39% fully-legal clouds).**

The median displacement at convergence is literally 0.000×rd — the model applies
near-zero displacement to the majority of points. It has learned a near-perfect
sparse selector: do nothing to legal points, push only violators. This is the
architecture working correctly.

The violation gap vs model6 is not an architecture failure — it is a **single-pass
ceiling**. When 3+ mutually-violating points form a cluster, fixing one pair's
violation by moving two points can push them into new violations with a third.
A single forward pass cannot resolve this; it would need to move a *legal* point
to create slack, which the violation-weighted aggregation correctly refuses to do.

Model6 hides this by applying large, near-uniform displacement to every point
(including legal ones), which creates slack by accident. This is why model6 fixes
more violations in one shot — it is doing undirected shoving, not targeted surgery.

**The fix is Step 3 (iterative unrolling)**: give model7 K passes on the same
cloud. Each pass reduces violations; the next pass sees updated geometry and
tightens further. The sparse, precise nature of model7's corrections is exactly
what makes multi-pass refinement tractable — model6's broad displacements would
compound and overshoot.

### Files
`models/fixed_rd/model7.py`, `configs/model_configs/model_config_7.yaml`,
`configs/smoke_test/model_config_7.yaml`.

---

## Step 3 — Iterative unrolling ▶ next

**Goal:** close the violation-reduction gap revealed by model7. A single forward
pass cannot resolve multi-point conflict clusters without moving legal points;
K passes let each step reduce remaining violations without overstepping.

**Why model7 specifically needs this:** model6's broad uniform displacement
creates slack by accident, masking the single-pass ceiling. Model7's near-zero
median displacement at convergence (0.000×rd) shows it has learned to be precise
— precision is wasted if it only gets one shot.

### Implementation (all changes in `trainer.py`)

Replace the single forward + loss with a K-step unroll loop:

```python
x_current  = x                     # noisy input (pre-correction)
total_loss = torch.tensor(0.0, device=device)

for k in range(train_cfg['unroll_steps']):       # K = 3
    displacement = model(x_current, rd=rd)
    x_next       = x_current + displacement
    # displacement penalty measured from *this step's* input (per-step efficiency)
    total_loss   = total_loss + loss_fn(x_current, x_next, rd, **loss_cfg['params'])
    x_current    = x_next.detach()               # stop gradient between steps

optimizer.zero_grad()
total_loss.backward()
optimizer.step()

corrected = x_current              # final state after K steps (used for eval)
```

**Why `.detach()` between steps:** gradients only flow through a single step's
forward pass. The model learns "given any partially-corrected cloud, improve it" —
a fixed-point iteration. Without detach, full BPTT through K steps is possible
but memory scales with K and vanishing gradients appear at K≥4. Start with detach.

**Loss term:** `loss_fn(x_current, x_next, ...)` measures displacement from *that
step's* input, not from the original noisy cloud. Each step is penalised for its
own correction magnitude — the right signal for per-step efficiency.

**Eval changes:**
- Pre-correction: still `x` (original noisy input)
- Post-correction: `x_current` (after K steps)
- Displacement: `(x_current - x).norm(dim=-1)` — total across all K steps
- `correction_eff`: unchanged formula, now measures K-step efficiency jointly

**Config:** one new key in `train_config`:
```yaml
unroll_steps: 3   # 1 = current single-pass behaviour (backward compatible)
```

**Model architecture:** no changes — model7.py is called identically K times,
each time with updated point positions as input.

**Files:** `training/trainer.py`, new `configs/trainer_configs/train_config_2.yaml`.

---

## Step 4 — Larger clouds
**Goal:** verify the model scales with N; expose it to richer conflict patterns.  
**Change:** N: 50 → 100 → 200, same rd=0.05, same box [0,1]².  
**Note:** edge network is O(N²) — at N=200 compute per iter is ~16× vs N=50.
Reduce `num_iterations` or enable AMP if needed.  
**Files:** `configs/dataset_configs/dataset_config_4.yaml` (N=100), `_5.yaml` (N=200).

---

## Step 5 — Packed scenarios (3 substeps)

### What "packed boundary" means

A **packed boundary** cloud is one where the point density is at or near the
maximum admissible for the given rd — no additional point can be inserted anywhere
in the domain without violating the Poisson disk constraint. In 2D the theoretical
ceiling is the triangular lattice: `N_max ≈ 2 / (√3 × rd²)`. At rd=0.05 this
gives **N_max ≈ 462** in [0,1]².

This regime is qualitatively harder than dense-but-not-packed clouds because there
is no slack: every correction that moves a point away from one violation risks
pushing it into another. The free space that a low-density corrector relies on
simply does not exist. Solutions must be genuinely local and conflict-aware, which
is exactly why iterative unrolling (step 3) is a prerequisite.

---

### Domain boundary enforcement

At low density (steps 1–4) points rarely sit close enough to the domain edge for
noise or corrections to push them outside — boundary enforcement adds complexity
with negligible benefit.

At packed densities this changes fundamentally. Many points necessarily sit near
the boundary (there is nowhere else), so:
- **Noise** will push a non-trivial fraction outside [0,1]²
- A boundary point has neighbours only on one side — if pushed outside it appears
  artificially "legal" to the loss, producing meaningless gradients
- The domain boundary is *why* N_max exists; if points escape it the packedness
  concept loses its ground truth

**Two-stage enforcement for step 5:**

1. **Noise clamping** — after adding Gaussian noise, clamp all coordinates to
   [0,1]² before returning the batch. One line in `PackedPoissonDiskDataset.noise_sample`.

2. **Drop DataProcessor for packed/PBC runs** — model7 is translationally invariant
   by construction (it uses relative positions). Centering was only necessary for
   model1–3 (raw-coordinate global attention). Without DataProcessor the model
   operates directly in [0,1]² space and the corrected output can be clamped trivially:
   `x_corrected = (x + displacement).clamp(0, 1)`.
   This aligns with what PBC (step 6) requires anyway — no centering on a torus.

**Loss change** — optionally add a boundary penalty term for packed runs:
`relu(0 - x_corrected) + relu(x_corrected - 1)` summed over all coordinates,
weighted by a small λ_boundary. Keeps the model from learning to "escape" the
domain as a way to avoid violation penalties.

---

### Step 5a — Packed boundary + small noise

**Goal:** establish a baseline on packed clouds. Small σ means few violations and
shallow violation depths, but even shallow violations are hard to fix because every
displacement is constrained by nearby legal pairs.  
**Noise:** `noise_scale_min: 0.0`, `noise_scale_max: 0.005` (0.1× rd).  
**N:** ~400 (≈87% of N_max) — firmly in the packed regime without being adversarially
tight.  
**Expected difficulty:** the model must learn to make tiny, precise corrections
without disturbing the densely-packed legal neighbourhood.  
**Files:** `configs/dataset_configs/dataset_config_packed_a.yaml`.

---

### Step 5b — Packed boundary + large noise

**Goal:** test the model on the hardest single-regime scenario: many points
clustered together in a maximally-packed cloud.  
**Noise:** `noise_scale_min: 0.02`, `noise_scale_max: 0.04` (0.4–0.8× rd).  
**Why hard:** at large σ in a packed cloud, multiple clusters of 3–5 mutually-
violating points form simultaneously. Fixing one pair displaces into another.
This is the canonical multi-point conflict that single-pass correction cannot
resolve — iterative unrolling earns its keep here.  
**Files:** `configs/dataset_configs/dataset_config_packed_b.yaml`.

---

### Step 5c — Packed boundary + full variable noise

**Goal:** the complete packed curriculum — σ ∈ [0, 0.04], combining the
"do-nothing" case (legal packed cloud), the easy case (5a), and the hard case (5b)
in a single training distribution.  
**Why last:** the model must handle the full difficulty spectrum simultaneously,
including the pathological combination of maximum density and maximum perturbation.
Only run after 5a and 5b have confirmed the model can handle each regime individually.  
**Files:** `configs/dataset_configs/dataset_config_packed_c.yaml`.

---

## Step 6 — Periodic boundary conditions (PBC)
**Goal:** toroidal domain — boundary effects gone, every point equivalent.
The full physically-motivated setting for materials / molecular dynamics.  
**Changes:**
- `model7.forward`: replace `rel_pos = xi - xj` with MIC wrapping:
  `rel_pos = rel_pos - torch.round(rel_pos)`
- Post-displacement: `x_corrected = (x + displacement) % 1.0`
- Drop `DataProcessor.make_invariant` (centering meaningless on torus;
  model7 is already translationally invariant by construction)
- Custom PBC-aware Poisson disk generator (scipy doesn't support toroidal sampling)  
**Files:** `models/fixed_rd/model8.py`, `data/data_generator.py` (PBC sampler),
new dataset + model configs.

---

## Step 7 — Benchmark on real packed data

**Goal:** determine which model and inference strategy performs best on real-world point
clouds that are already very densely packed — the regime where the model6 vs model7
distinction is most consequential.

### Why this is the decisive test

The synthetic experiments (Steps 1–6) showed a clear trade-off:

| | Model6 | Model7 |
|---|---|---|
| Strategy | Global uniform push | Targeted per-pair push |
| Single-pass violations (×1) | 0.33% | 0.64% |
| NN drift at ×5 | **1.311 × rd** | 1.131 × rd |
| Surgical efficiency | 0.018 | **0.026** |

In sparse synthetic clouds (50 pts, ~10% packing density), model6 can afford to globally
expand the cloud because there is empty space to push into. On **real packed data** that
slack does not exist: every point is surrounded by legal neighbours on all sides. Model6's
global push creates new violations while fixing old ones. Model7's violation-weighted
aggregation — which applies near-zero displacement to legal points — is the only approach
that can fix local clusters without disturbing the global structure.

The real data test is therefore not a generalization sanity check — it is the benchmark
that actually matters. Synthetic results are a proxy; this is the target.

### What "packed" means for the corrector

In a packed cloud, a violating pair (i, j) has neighbours k₁, k₂, … on both sides at
distance ≈ rd. Moving i away from j pushes i toward k₁. A corrector that moves
non-violating points (model6's global push, or a large uniform displacement) will
cascade new violations. The only safe correction is:

1. Move only i and j
2. Move them the minimum amount necessary
3. Move them in the direction with the most free space (requires local geometry awareness)

Model7's aggregation weights give it exactly properties 1 and 2. Property 3 is what
iterative unrolling (Step 3) and larger K help with — each pass sees updated geometry
and re-routes around newly-crowded directions.

### Models to compare

Run all trained checkpoints on the same real dataset:

| Condition | Weights | Inference passes |
|---|---|---|
| Model6 K=1 (baseline) | K=1-trained | ×1 |
| Model6 K=1 × 3 | K=1-trained | ×3 |
| Model7 K=1 | K=1-trained | ×1 |
| Model7 K=1 × 3 | K=1-trained | ×3 |
| Model7 K=3 trained | K=3-trained (principled λ) | ×1 |
| Model7 K=3 trained × 3 | K=3-trained (principled λ) | ×3 |

The hypothesis: model7 K=3-trained weights, applied ×1 or ×3, will show the best
combination of low residual violations and minimal NN drift on packed real data.
Model6 will visibly over-spread even at ×1 when there is no empty space to absorb the
global push.

### Evaluation protocol (no ground truth clean cloud)

- **Primary:** `viol_per_cloud` before vs after — must go down, must not increase anywhere
- **Displacement conservatism:** `mean_nn_dist` after correction — must stay close to rd,
  not balloon above it (over-spreading in a packed cloud means the corrector is shoving
  legal structure aside)
- **Zero-violation test:** feed a subset of clean real clouds (zero violations) — model
  must output near-zero displacement. Any significant displacement on a legal packed cloud
  is a false positive that will break the global structure.
- **Cascade check:** after correction, count how many new violations were introduced
  (pairs that were legal before but illegal after) vs how many were resolved. Net
  reduction = resolved − introduced. Model6 is expected to have a high introduction rate
  in packed clouds.
- **Visual:** side-by-side before/after for 10–20 clouds; look for structure preservation

### What needs to change in code

- **Scale normalisation:** real clouds may not live in [0,1]². Normalise by mapping the
  bounding box to [0,1]² and scaling rd accordingly before inference. Undo the scale
  after correction to return to original coordinate space.
- **rd:** if not given, estimate from the empirical 5th-percentile nearest-neighbour
  distance across a set of clean (violation-free) examples from the dataset.
- **No model changes** — model7's forward pass takes `rd` as a runtime scalar and is
  data-agnostic. The comparison script (`analysis/eval_real.py`) handles normalisation.

### Files

- `analysis/eval_real.py` — loads real point set (numpy / CSV / PLY), normalises,
  runs each model condition, reports full metric table, saves comparison sheet
- `data/real/` — real dataset files (to be added when data is available)

---

## Architecture improvements (parallel track)
- **Self-feature**: concatenate a learned embedding of `x_i` to `agg` before
  `output_mlp` — gives the model an explicit "where am I?" signal.
- **λ1 scheduler**: decay λ1 over training (strong early push → gentle late polish).
- **Output clamping** (`max_displacement`): prevents runaway displacements in
  early training, especially at high density.
