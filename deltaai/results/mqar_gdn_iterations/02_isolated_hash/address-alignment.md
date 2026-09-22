Training remains paused. This analysis uses saved first-layer weights and the
same 128 examples per cell as the completed GPU probe. It constructs no model,
runs no recurrent kernels, changes no weights, and launches no GPU jobs.

The first-layer reconstruction reproduces the saved GPU geometric target-weight
shares within 0.000008 absolute in every cell. Tiny arithmetic differences can
change a few discrete selections. It includes RMSNorm, the causal writer
convolution, hash projection/CDF/binning, categorical read scores, and planted
position overrides.

ContentAssign defaults to `plant=True`, and these models never disable it. At
lengths 64 and 128, writes at zero-based positions 5 and 1 are forced into fixed
buckets 5 and 1. At length 256, positions 7 and 1 are forced into buckets 7 and 1.
They still perform one write each, but these two assignments bypass the learned
hash. These are inherited, data-independent finite-geometry anchors, not a new
MQAR-specific change. Describing every write as learned content hashing would
be inaccurate.

In this MQAR layout the anchor positions happen to contain values. On the
16-pair evaluation cell, the first-layer geometric reader allocates:

| Requested value position | Target share of value-routing weight | Target coverage per head |
|---|---:|---:|
| One of the two fixed anchors | 23.15% | 89.26% |
| Any of the other fourteen positions | 3.74% | 53.52% |
| Uniform share across 16 values | 6.25% | — |

Anchor targets receive about 6.2 times the share of other targets. The overall
6.17% average conceals this preference. These are first-layer geometric weights,
not final prediction accuracy. They do not prove that removing anchors from
this trained network would improve accuracy.

This interacts with the density mismatch: the length-64 router trains on four
pairs, two of which are anchored, then evaluates on sixteen, only two of which
are anchored. On four-pair validation, anchor and nonanchor target shares are
much closer: 26.30% and 24.52%. At length 256 the trained router instead gives
anchored targets less weight. The bias is not uniform across lengths; routing
parameters are separate for each length.

The proposed shared-address reader needs a qualification. Applying the saved
writer hash to query features produces the exact correct value-position bucket
only 1.29% of the time on 16 pairs, versus 1.03% for a uniformly chosen incorrect
value from the same sequence. Excluding anchored targets, those rates are
1.23% and 1.07%. The writer hashes a learned causal mixture, whereas this query
would hash its current token features. Sharing a projection does not make the
two feature distributions agree.

Restricting the current categorical reader to four planes through that query
address gives 6.28% target geometric share, versus 6.17% currently and 6.25%
uniform. Per-head target coverage falls from 57.98% to 48.97%. Equal read weights
give 6.32% share. These are routing statistics, not a prediction of model
accuracy; no counterfactual model inference was performed.

For d=3, four distinct lines through a query bucket give that bucket total
weight one. Any different bucket lies on at most one selected line, giving
weight at most the largest line weight (one quarter with equal weighting).
This can favor matching addresses, but cannot create writer/query agreement
or distinguish multiple values in the same bucket.

A future experiment should therefore learn writer/query address alignment
jointly and explicitly decide whether to retain positional anchors. Merely
constraining the reader to hard planes gives no direct gradient through the
query's discrete bucket selection. It needs a deliberate surrogate or a
differentiable address-based score. Continuous routing features could also be
shared across lengths with different geometry sizes. These are design
hypotheses, not proven fixes. Training code has not been changed.

Reproduction: `analyze_address_alignment.py`. Raw results are in
`address-alignment.json` and `address-alignment.log`.
