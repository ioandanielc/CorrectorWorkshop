# Cardinality ablation

Does the KG floor (~0.111) depend on the cardinality the corrector was trained
at, or is it a property of the violation<->symmetry trade-off?

The shipped checkpoint trains at N=49, where the model's reach
(`L * cutoff_rd` = 8 spacings) exceeds the box (7 spacings) — it sees every
particle at once. At deployment (N=2500, 50 spacings) it cannot. Every arm here
re-trains the same architecture at a larger N, i.e. a smaller reach/box ratio,
and is evaluated on the same real trajectory.

| Arm | N | side | spacing | rd | cutoff_rd | reach/box | deploy scale |
|---|---|---|---|---|---|---|---|
| baseline (`src/configs/training/{dataset,model,loss}/`) | 49 | 7 | 0.1429 | 0.140 | 0.286 | 1.14 | 7.0 |
| n100 | 100 | 10 | 0.1000 | 0.098 | 0.200 | 0.80 | 4.9 |
| n196 | 196 | 14 | 0.0714 | 0.070 | 0.1429 | 0.57 | 3.5 |

## Holding the recipe fixed across arms

Every length is a fixed multiple of the lattice spacing `1/sqrt(N)`, so the arms
are the same recipe at different scales:

    rd               = 0.98 * spacing
    cutoff_rd        = 2    * spacing        (= h, the SPH smoothing length)
    max_displacement = 1.2  * rd
    noise_scale_max  = 0.6  * rd
    lambda1          = 1 / rd
    lambda2          = 0.1 * lambda1 / (N-1)

`h_factor: 2.0` and `box: 1.0` are unchanged — `kernel_gradient` derives
`dx = box / sqrt(N)` itself, so the KG kernel rescales automatically.

**lambda3 does NOT carry over.** Measured at matched relative disorder
(64 clouds, noise 0.6*rd):

| N | mean\|KG\| | illegality term | symmetry : illegality | lambda3 |
|---|---|---|---|---|
| 49 | 0.2322 | 0.0096 | 2.67 | 0.27 |
| 100 | 0.3288 | 0.0048 | 10.37 | 0.070 |
| 196 | 0.4516 | 0.0024 | 38.81 | 0.019 |

`|KG| ~ sqrt(N)` (the `dx**2` factor in `utils/metrics.py`) while the illegality
term ~ `1/N` (a mean over all N^2 pairs, of which only O(N) violate), so their
ratio grows as N^2 and lambda3 must fall as 1/N^2 to keep the same trade-off.
Leaving lambda3 at 0.27 would make the n196 arm a ~14.5x-overweighted symmetry
run rather than a cardinality arm. `lambda2`'s existing `1/(N-1)` already holds
the displacement:illegality ratio constant (0.217 -> 0.214), so it needs no
correction.

## Cost — measured, and the blocker

Dense `forward` materialises `(B, N, N, ·)`, so activation memory grows as
`B * N^2`. Measured on an RTX 4080 SUPER (16 GB), B=32, unroll_steps=5, the real
recipe:

| N | ms/iter | peak VRAM | 10k iters |
|---|---|---|---|
| 49 | 73 | 4.26 GB | ~12 min |
| 100 | 4575 | 17.24 GB — spills past VRAM, thrashes | (~13 h) |
| 196 | — | ~68 GB projected, hard OOM | — |

Data generation is not the bottleneck at any N (1.2 ms/iter, <0.1%).

The 5x multiplier is avoidable. `trainer.py` sums `total_loss` over the unroll
loop and calls `backward()` once afterwards, so all 5 steps' graphs are alive at
the same time. The steps are already `detach()`ed from each other, so each step's
loss is an independent function of the parameters and
`sum_k backward(L_k) == backward(sum_k L_k)` exactly — backward-per-step would
free each graph immediately and cut peak memory ~5x, changing no result.

Even so, N=196 at B=32 needs ~13.6 GB after that fix. Running all three arms at
an identical effective batch of 32 needs gradient accumulation over micro-batches
as well. **Both are trainer changes and are not implemented** — until they are,
the arms can only be run at matched `B * N^2` (B = 32 / 8 / 2), which changes
gradient noise between arms and confounds the comparison.

## Running

    .venv\Scripts\python.exe src/training/trainer.py ^
      --train-config   src/configs/training/trainer/train_config_sph_adamw.yaml ^
      --dataset-config src/configs/training/ablations/cardinality/dataset_config_sph_n196.yaml ^
      --loss-config    src/configs/training/ablations/cardinality/loss_config_rdsph_n196.yaml ^
      --model-config   src/configs/training/ablations/cardinality/model_config_12_sph_L4_n196.yaml

## Evaluating

All arms must be scored on the **same** data at the same deployment settings —
whole-cloud, k=5, `box: 1.0`, on the N=2500 trajectory. KG is only comparable at
equal `dx`, so cross-arm comparison of training-time KG is meaningless; the
deployment measurement is at identical dx for every arm and is the one that
counts. Each arm needs an experiment config with its own `rd_train`
(0.140 / 0.098 / 0.070) so `compute_scale` maps the trajectory correctly.

Primary readout: does `model12_wc` KG over t >= 300 still bottom out at ~0.111?
Secondary: illegal% (82.7% for the N=49 checkpoint) should fall with training N
if the violation objective is what fails to size-generalise.
