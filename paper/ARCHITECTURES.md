# Five architectures on one task: what each one can see

Reference for the architecture ablation. What each model perceives when handed the
same point cloud, what that lets it do and stops it doing, and what the numbers show.

All five train under identical conditions — same loss (`rdsph`, lambda1=7.14,
lambda2=0.0149, **lambda3=0.27**), same data, same optimizer and schedule, same 10k
iterations, same 5-step unrolling, same bounded `tanh * max_displacement` head, and
parameter counts matched within 0.8%. **The baselines get the physics-informed KG
term too.** The claim is not "physics-informed beats vanilla" — it is that given the
same physics-informed objective, the architecture decides whether that objective is
reachable.

### Fairness protocol (fixed, and deliberately not per-architecture tuned)

Every architecture receives an *identical budget*; none is tuned individually, model12
included. Equal-budget comparison is the protocol, not best-effort-per-model. Two
things follow that must be stated rather than hidden:

- **The shared recipe was originally tuned on model12** (the 2026-07-15 sweep chose
  L=4, lambda3=0.27, AdamW, lr 1e-3). So the shared settings are model12's settings.
  This is the honest asymmetry in the comparison and belongs in the limitations.
- **The one catastrophic result is not a tuning artifact.** DGCNN reaches 79.4%
  violation reduction at noise 0.6 and *exactly* 0.0% at noise 1.0 — same code, same
  hyperparameters, same everything but the disorder level. A wrong learning rate would
  be wrong at both. What changed is the task, which is the mechanism claim; it also
  rules out an implementation fault, since the identical code path works at low noise.

Per-architecture hyperparameter searches were deliberately NOT run. They would improve
the baselines by an unknown amount, cost a multiple of the whole suite, and replace a
clean stated protocol with an unfalsifiable "was it tuned enough" argument.

| model | params | vs model12 | isolated s/iter | file |
|---|---|---|---|---|
| model12 | 350,594 | — | 0.073 | `models/architectures/model12/model12.py` |
| GNS-style | 347,966 | −0.75% | 0.223 | `models/architectures/gns/gns.py` |
| DGCNN | 348,692 | −0.54% | 0.047 | `models/architectures/dgcnn/dgcnn.py` |
| PointNet++ | 350,240 | −0.10% | tbd | `models/architectures/pointnet2/pointnet2.py` |
| PointNet | 351,914 | +0.38% | 0.025 | `models/architectures/pointnet/pointnet.py` |

---

## The task, and the two length scales that decide everything

Given a cloud violating a minimum distance `rd`, predict per-point displacements
giving a valid cloud that is also a usable SPH restart. Two requirements, at two
different ranges:

1. **Constraint satisfaction** — pairwise and local. Point `i` must end up `>= rd`
   from each neighbour. Range: **1 spacing**.
2. **Kernel-gradient symmetry** — `KG_i = sum_j dW/dr * e_ij * V_j -> 0`. Not
   pairwise: a statement that the neighbourhood is *balanced* around `i`, involving
   every particle inside the kernel support, violating or not. Range: **6 spacings**.

Requirement 1 is easy — push overlapping pairs apart. Requirement 2 is the one that
separates architectures, because it needs *non-violating* neighbours to influence the
answer, weighted by distance. An architecture that only reacts to violations, or that
keeps just the most salient neighbour, is structurally unable to express "these
directions cancel".

Intuitively: **requirement 1 asks "who is too close to me?", requirement 2 asks "am I
being pulled evenly from all sides?"** The second question can only be answered by
something that adds up all the directions with the right weights.

---

## model12 — a physical kernel written into the wiring

```
w_ij = (1 - (d_ij/rc)^2)^2 / sum_j (...)     fixed, not learned; exactly 0 at d >= rc
e_ij = edge_mlp_l([h_i, h_j, rel_ij, d_ij, relu(rd - d_ij)])
h_i += node_mlp_l( sum_j w_ij e_ij )          L=4 rounds, residual
```

**What it sees:** every neighbour inside a physical radius, each one's contribution
pre-scaled by a smooth function of distance that mimics the SPH kernel, normalised so
the weights sum to one. Overlap depth arrives separately as `relu(rd − d)`, so
"too close" and "how far" are distinct signals rather than entangled.

**Pros**
- The distance weighting is *imposed*, not learned — it cannot be discarded, and no
  training data is spent rediscovering it.
- Kernel-weighted **sum** is the same mathematical form as `KG` itself, so symmetry is
  directly expressible rather than approximable.
- Smooth: `w` reaches zero continuously at the cutoff, so the output is continuous in
  particle positions — no jump when a pair crosses the boundary.
- Per-particle normalisation makes node states independent of neighbour count, which
  is why it transfers from N=49 to N=2500.

**Cons**
- The kernel is a commitment. On a task where the right weighting is *not* kernel-like,
  it is a bias you cannot train away.
- Dense `(B,N,N,·)` in `forward`; needs `forward_sparse` for large clouds.
- Not rotation-equivariant (deliberate for a fixed simulation frame).

## GNS-style — the same graph, but the weighting is learned

```
e_ij  = edge_encoder([rel_ij, d_ij])              persistent edge latent
e_ij += edge_mlp_l([e_ij, h_i, h_j])              residual edge update
h_i  += node_mlp_l([h_i, sum_{j in ball} e_ij])   unweighted sum
```

Identical radius graph, identical cutoff, identical round count, identical
minimum-image geometry. It differs in exactly two mechanisms, which makes it the
sharpest baseline in the suite:

| | model12 | GNS |
|---|---|---|
| message weight | fixed `(1−q²)²` SPH kernel | learned, in a persistent edge latent |
| aggregation | kernel-weighted mean | unweighted sum |

**So model12 = GNS + SPH-kernel message weighting.** The gap between them is the price
of *hoping the network learns the kernel* versus *building it in*.

**What it sees:** the same neighbours, but with a hard on/off membership and a learned
notion of how much each matters, carried in an edge state that persists across rounds.

**Pros**
- Strictly more expressive in principle — it *can* represent the kernel weighting,
  and can represent weightings the kernel cannot.
- Persistent edge latents give edges memory, useful when a relationship needs to be
  reasoned about over several rounds.
- The published, recognised architecture for learned particle simulation.

**Cons**
- Must spend capacity and data learning distance weighting that model12 gets free.
- The hard radius mask makes the output **discontinuous** as pairs enter and leave —
  a jump exactly where SPH physics is smooth.
- Unnormalised sum means message magnitude scales with neighbour count (measured
  degree on a 7×7 lattice: mean 11.6, min 7, max 19), so density variation leaks into
  latent magnitudes and works against size transfer.
- ~3× the compute per iteration (0.223 vs 0.073 s/iter) for the edge state.

## DGCNN — neighbours in feature space, and only the loudest one survives

```
idx  = k nearest in the CURRENT feature space     rebuilt every round
e_ij = edge_mlp_l([h_i, h_j − h_i])
h_i  = max_j e_ij
```

**What it sees:** after round 0, *not the spatial neighbourhood at all*. The graph is
rebuilt in feature space, so two points adjacent in the graph may be far apart in the
box — that is the entire point of DGCNN in its home domain (it groups semantically
similar structure). Here, where the objective is defined by physical distance, that
freedom is mostly a liability.

Then `max` keeps, per feature channel, the single most extreme neighbour and throws
the rest away.

**Pros**
- Fixed degree `k` gives predictable cost and density-adaptive neighbourhoods.
- Feature-space graphs excel at semantic grouping — genuinely the right tool for
  segmentation/classification.
- Cheapest of the graph models here (0.047 s/iter at k=12).

**Cons**
- **Max aggregation cannot express cancellation.** Symmetry is "all these directions
  sum to zero"; a max reports "the biggest direction was this". This is the deepest
  structural mismatch with the KG objective in the whole suite.
- Feature-space neighbours drift away from physical neighbours, so the graph stops
  corresponding to the physics after the first round.
- No physical cutoff: `k` is a count, not a length, so the model has no representation
  of the kernel support.

## PointNet++ — a hierarchy instead of depth

```
SA:  FPS centroids -> ball query -> mini-PointNet -> max-pool     (N -> N/4 -> N/16)
FP:  inverse-distance 3-NN interpolation back up, with skips
```

The only model here that grows its receptive field by **pooling** rather than by
iterating local message passing.

**What it sees:** a coarse summary. Points are grouped into balls, each ball is
crushed to one max-pooled vector at a centroid, and information travels by moving up
and down that hierarchy. A point's final feature is an *interpolation* of nearby
centroid features plus its own skip connection.

**Pros**
- Reaches long range in two levels rather than L hops — cheap receptive-field growth.
- Multi-scale by construction; strong where density varies a lot.
- Ball query is a genuine physical radius (unlike DGCNN's `k`).

**Cons**
- **Interpolation smooths.** The final per-point feature is a distance-weighted blend
  of a few coarse vectors, but this task needs precision well below `rd`. Neighbouring
  points get near-identical features and therefore near-identical displacements —
  which is close to a uniform translation, and a uniform translation changes nothing.
- Max-pooling inside each ball discards the same balance information DGCNN's max does.
- The hierarchy is cramped at small N (49 → 12 → 3); only at N=196 (196 → 49 → 12) is
  it well proportioned. The N=49 column should be read with that in mind.
- FPS is a sequential loop — awkward and slow to batch.

## PointNet — no pairwise term at all

```
h_i = encoder(x_i);   g = max_i h_i;   disp_i = head([h_i, g])
```

**What it sees:** its own coordinates, and one global summary vector shared by every
point. It never learns *which* points are near it. Two points in completely different
neighbourhoods, at the same position modulo the global feature, get the same
displacement.

**Pros**: trivially permutation-equivariant, cheapest here (0.025 s/iter), and the
correct *control* — it isolates "capacity" from "geometry".

**Cons**: structurally incapable of the task. With no pairwise term there is also
nowhere for the minimum image to apply, so it sees raw coordinates on the torus.

---

## Why the same data looks different to each of them

Hand all five the identical 49-point cloud. What reaches the decision for point `i`:

| model | what point `i` actually perceives |
|---|---|
| model12 | "12 neighbours, at these directions, weighted 0.31/0.22/0.18/… by distance, and 2 of them are overlapping me by 0.03" |
| GNS | "12 neighbours, at these directions, each with a learned importance; here is their unweighted total" |
| DGCNN | "my 12 most feature-similar points; the strongest response among them was this" |
| PointNet++ | "the coarse summary of my region, blended from 3 nearby centroids" |
| PointNet | "my position, and the average vibe of the whole cloud" |

Descending that table, the *balance* information — the thing `KG` is made of — is
progressively destroyed: model12 preserves it exactly (weighted sum), GNS preserves it
approximately (unweighted sum, learnable weights), DGCNN discards it (max), PointNet++
blurs it (interpolation) and then discards it (max), PointNet never had it.

Meanwhile the *violation* information survives much further down: "someone is too
close" is a strong local signal that even a max can carry. Which yields the concrete
prediction the measurements test — **architectures should separate much more on |KG|
than on violation reduction.**

---

## Measured so far

N=49, noise 0.6·rd, 10k iterations, K=5, from the trainer's eval block. These are
non-periodic and computed on a fresh random batch, so treat them as indicative;
`score_arm.py` numbers are the ones bound for the paper.

| arm | viol_red K5 | illegal% | mean nn | \|KG\| K5 | vs model12 KG |
|---|---|---|---|---|---|
| **model12** | **85.7%** | **0.53** | 0.1420 | **0.0228** | — |
| DGCNN | 79.4% | 0.77 | 0.1409 | 0.0310 | +36% worse |
| PointNet | −0.1% | 3.73 | 0.0981 | 0.2548 | +1018% worse |
| GNS | running | | | | |
| PointNet++ | queued | | | | |

The prediction is holding so far:

- **DGCNN loses 6.3 points of violation reduction but 36% of KG** — separating far more
  on symmetry than on constraint satisfaction, exactly as the mechanism table implies.
- **PointNet does nothing at all**: |KG| 0.2548 against an input of 0.2544, and
  violation reduction is *negative*. Ten thousand iterations with the KG term in its
  loss cannot buy symmetry without relative geometry. This is the cleanest possible
  demonstration that the physics has to be in the architecture, not only the objective.

Open questions the remaining runs answer: how much of model12's margin over DGCNN is
the kernel specifically (GNS isolates it), whether hierarchy substitutes for depth
(PointNet++), and whether the margin widens as the task approaches the packing limit
`rd/spacing -> 1`, where the only feasible configuration is the exact lattice and its
KG is exactly zero.
