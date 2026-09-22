# Width-32 transport diagnosis

Evaluation only; training recipe was not changed. Each probe uses the first 128 cached validation examples per pair-count cell (640 examples total). These are checkpoint interventions, not retrained architecture comparisons.

## Epoch 13: memory interventions

| Intervention | 16-pair accuracy | 64-pair accuracy |
|---|---:|---:|
| normal | 55.47% | 3.12% |
| memory_off | 0.05% | 0.01% |
| no_decay | 55.22% | 3.05% |
| no_delta_transport | 42.43% | 2.43% |
| no_transport | 41.46% | 1.81% |
| oracle_reads | 51.37% | 6.09% |

`no_decay` removes scalar forgetting from the transport only. `no_delta_transport` retains scalar decay but removes rank-one transforms. `no_transport` removes both transforms while retaining beta-weighted writes. `oracle_reads` uses labels to select four distinct hyperplanes and weight only those containing the actual value-position write bucket; this is an artificial diagnostic and not a deployable model. None changes local GDN recurrence.

## Epoch 16: profile selectivity

| Pairs | Layer | Occupied value buckets / available | Values covered by four reads | Target share of geometric routing weight |
|---|---:|---:|---:|---:|
| 16 | 1 | 3.32 / 25 | 15.21 | 6.27% |
| 16 | 2 | 5.21 / 25 | 15.77 | 6.23% |
| 64 | 1 | 4.72 / 49 | 60.30 | 1.56% |
| 64 | 2 | 7.76 / 49 | 37.82 | 1.57% |

Numbers average over examples/queries and heads. Target share is the normalized incidence/read-selector weight across actual value positions, before the signed transported Q/K content coefficient. It is not the final attention weight. Occupancy counts only actual value writes, not key/filler writes.

Interpretation: the memory branch is essential; scalar forgetting is not the obvious immediate failure. Many values collide in a few buckets, and selected hyperplanes admit many distractors. Rank-one transport helps this checkpoint, but the geometry is providing little target-specific filtering. This identifies a bottleneck, not a proof of its optimization cause or a complete explanation of the gap to independently trained GDN.

## Paused at epoch 23: matched comparison

Training was stopped at the user's request. No new training or GPU evaluation
was launched after the pause. The following analysis reads checkpoint tensors,
the existing probe outputs, and the source. The resumable checkpoint is local
and on Modal's persistent volume; see `paused.json`.

| Model, epoch 23 | Overall accuracy | 16 pairs | 32 pairs | 64 pairs |
|---|---:|---:|---:|---:|
| GDN | 68.71% | 90.40% | 43.87% | 9.86% |
| Original SMAT d=3 | 44.05% | 29.01% | 15.92% | 6.02% |
| Transport SMAT d=3 | 54.81% | 65.48% | 19.89% | 3.74% |

The transport variant improves overall performance over original SMAT at this
epoch, but does not beat GDN and is worse on the 64-pair cell than either
comparison. These are single-seed results, not estimates of seed variability.

## Saved weights: distinct hash coordinates have converged

For d=3 the address has two coordinates. The checkpoint shows that several
coordinate projections are nearly identical, including their learned scales
and offsets. Raw data are in `routing-parameter-analysis.json`.

| Transport epoch 23 | Cosine between normalized coordinate projections | Gamma pair | Bias pair |
|---|---:|---|---|
| Layer 1, T=64, head 1 | 0.99999976 | (0.68159, 0.68170) | (-0.25290, -0.25278) |
| Layer 1, T=64, head 2 | 0.99999952 | (0.61097, 0.61091) | (-0.39156, -0.39175) |
| Layer 1, T=256, head 2 | 0.99999297 | (0.58858, 0.58972) | (-0.62405, -0.62251) |
| Layer 2, T=64, head 1 | 0.99999964 | (0.78910, 0.78907) | (-0.62415, -0.62427) |

If the two coordinates were exactly equal, almost all ordinary writes would lie
on the diagonal `(x,x)`: q address cells rather than q^2. Near equality supports
that interpretation but does not itself count bucket occupancy. Fixed planted
anchors are an exception. The epoch-16 activation probe independently found
poor occupancy: at 64 pairs the first layer uses 4.72/49 value buckets and each
value shares a bucket with 43.59 other values on average; layer 2 uses 7.76/49
and has 35.11 other-value collisions. Independent uniform assignment would use
35.91/49 buckets on average and produce 1.29 other-value collisions. Uniform
assignment is a reference calculation, not a required optimal routing policy.

The original epoch-32 d=3 checkpoint also has near-duplicate coordinate
projections in some routers (for example, layer 2/T=256/head 2 cosine 0.999909).
That is evidence this failure mode is not exclusive to boundary transport;
these checkpoint ages differ and this is not a matched-epoch causal comparison.

## Source audit: why the learning signal can permit this

The run uses the frozen `modal_bundle/deltaai/content_addr.py`, not the latest
working file. The frozen and current files differ in optional read-kernel
dispatch; the write-gradient and occupancy-loss code discussed here is the same.

With `hard_k1=True`, the chosen cell receives a straight-through scalar write
weight with forward value 1. All zero-valued neighboring routes are pruned.
Consider one non-clamped hash coordinate, away from bin boundaries, and an
ordinary, non-planted write. Let `s` be its continuous bin coordinate, `c` the
selected cell, and `g_c` the loss derivative with respect to that write's scalar
weight in cell c, holding the write payload fixed. The task-loss contribution is

```
current pruned estimator:       dL/ds_l = -g_c
unpruned hard STE:              dL/ds_l = g_(c+e_l) - g_c
```

The first expression gives the same scalar signal to every address coordinate
before each coordinate's CDF/projection Jacobian. It lacks the term that says
whether the neighboring bucket would be better. As a sanity argument, if all
bucket destinations have identical downstream effects, the second expression
is zero while the first can still move the hash. These are surrogate gradients;
the actual hard bin map is piecewise constant. This analysis identifies a
training-estimator weakness, not a failure of the transport Triton gradients.

The auxiliary loss does not close this gap. It balances the soft bin histogram
of each coordinate separately, over beta-weighted prefix tokens. It neither
balances joint hard bucket occupancy nor explicitly separates different values.
Both coordinates can have uniform marginals while always satisfying x1=x2;
then the marginal balance loss is zero although only q of q^2 joint cells are
used. Value positions are also only a subset of the tokens included in that
loss. Increasing its coefficient alone would not eliminate the joint-collapse
loophole.

The independent categorical reader is another limitation: its task gradient
goes through softmax weights of the selected four planes, not through the
discrete choice of an unselected plane. It has no explicit writer/query address
coupling. This can make learning coordinated selective access difficult, but
the existing evidence does not establish it as the primary cause.

## Priority and limits

The first change I would investigate is restoring neighboring-cell terms in
the **write-hash backward estimator**, while keeping one nonzero forward write
and four distinct query reads. The existing unpruned hard STE provides a simple
reference implementation: neighboring routes have zero forward weight but
carry gradient. It adds training work; a custom backward could retain the
single-route forward and gather neighboring bucket gradients only backward.
No implementation or training change has been made as part of this pause.

This is more directly motivated than changing the transport again. The
checkpoint intervention removing scalar decay barely changes accuracy, and
removing the rank-one transforms makes it worse. That supports keeping transport
while addressing hash learning first. It does not prove retraining without
decay or transport would produce the same outcome.

Joint occupancy or coordinate-diversity regularization would be a separate
follow-up, not a substitute for examining the surrogate. Diversity of projection
rows alone cannot guarantee diverse buckets on learned, anisotropic inputs.
Likewise, balanced buckets alone cannot guarantee correct query-to-value access.

The train/eval length-density mismatch remains a possible contributor, but the
64-pair/T=256 cell is length/density matched and still performs poorly. Thus the
failure cannot be explained solely as a dataset length mismatch. The current
evidence supports a routing/optimization bottleneck; it does not establish that
the general SMAT geometry or the transport idea cannot work.
