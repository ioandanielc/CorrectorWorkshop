# Related work — draft (§2 of the paper)

Prose in paper voice, ~330 words, four threads. BibTeX in `references.bib`; citation
keys inline. Verification: entries marked [web-verified] in the .bib had authors/venue
checked against publisher or arXiv pages on 2026-08-10; [standard] entries are canonical
but page numbers should be re-checked at camera-ready. One correction found while
verifying: the 2009 shifting paper is Xu, Stansby & **Laurence** (`xu2009accuracy`),
not Rogers.

---

**Particle regularity in SPH.** SPH accuracy degrades on disordered particle
distributions, and a family of classical remedies exists: particle shifting moves
particles down concentration gradients [xu2009accuracy, lind2012incompressible], the
transport-velocity formulation regularises positions inside the momentum equation
[adami2013transport], and packing algorithms iterate a damped dynamics to prepare initial
conditions [colagrossi2012particle, diehl2015generating]. All of these are
solver-coupled: they run as iterative physics loops inside, or ahead of, the simulation.
Our corrector addresses the same defect — a distribution violating minimum spacing
(equivalently, a blue-noise/Poisson-disk condition [bridson2007fast]) with asymmetric
kernel-gradient sums [morris1997modeling] — but as a learned, standalone operator applied
to arbitrary states with no solver in the loop.

**Learned particle simulation.** Graph-network simulators [sanchezgonzalez2020learning,
pfaff2021learning] learn dynamics rollouts over particle or mesh states. Neural SPH
[toshev2024neural] shows that such rollouts suffer tensile-instability-like particle
clustering and repairs them by inserting SPH relaxation steps at inference; diffSPH
[winchenbach2025diffsph] casts shifting itself as an optimisation over differentiable SPH
operators. Our work sits between these: rather than adding physics relaxation to a
learned simulator, we *learn the relaxation itself* — and our architecture is precisely a
GNS-class network with its learned edge weighting replaced by the fixed SPH kernel it
would otherwise have to approximate (§4.5).

**Physics-informed losses.** Embedding physical residuals in training objectives is
standard since PINNs [raissi2019physics]. Our kernel-gradient term belongs to this
family, but our ablation sharpens the usual claim: the term is not a soft constraint that
improves accuracy — without it, training diverges and the corrector reproduces the
failure mode of its predecessor (§4.3). The physics is an optimisation enabler.

**Point-cloud architectures.** PointNet and its hierarchical and graph-based successors
[qi2017pointnet, qi2017pointnetpp, wang2019dynamic] are the standard learned operators on
unordered point sets. We use them as parameter-matched baselines and find their generic
inductive biases — global pooling, feature-space graphs, max aggregation — fail this task
(§4.4), and that small-N benchmark rankings of them invert at deployment scale.

---

## Positioning notes (not for the paper)

- **Neural SPH is the closest work and the sharpest contrast.** They keep the learned
  simulator and add classical relaxation at inference; we learn the relaxation. If a
  reviewer asks "why not just run SPH relaxation?", the answer is in the deployment
  numbers: TV (the classical regulariser) reaches 0.274; the learned corrector reaches
  0.127, at 0.19 s/step.
- **diffSPH's shifting-as-optimisation** is per-state optimisation at inference; ours is
  amortised into one network — same relation as PINNs vs learned operators.
- If space allows one more citation thread: JAX-SPH (Toshev et al. 2024, arXiv
  2403.04750) as dataset/framework context.
