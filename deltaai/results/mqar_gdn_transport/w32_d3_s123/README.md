# GDN + SMAT boundary transport: width 32, d=3

Paused at the user's request after epoch 23/32 on 2026-09-16 at 23:42 CDT.
Modal app `ap-Nh8gXpdPFFxWIaPh125oFn` is stopped with zero tasks; the local
collector is stopped. `w32-d3.pt` contains the model, optimizer, scheduler,
module counters, and CPU/CUDA/NumPy/Python RNG states. Its epoch and metrics
match `w32-d3.json`; `SHA256SUMS` records the downloaded checkpoint digest.
The same checkpoint remains on volume `smat-gdn-transport-w32-d3`, at
`/seed123/w32-d3.pt`. `paused.json` records the pause. Training is incomplete
and has not been resumed. `diagnosis.md` records the analysis without new runs.

One requested 32-epoch MQAR experiment. Same seed 123, data cache, interleaved
batches, LR 0.01, AdamW/cosine, two layers, two heads and head/state 16 as the
previous width-32 sweep. One hard write per distant token and four distinct
learned weighted hyperplane reads per head/query. Local GDN and short
convolutions still reset at the midpoint. Routing remains length-specific.

For A_t = alpha_t (I - beta_t k_t k_t^T), form distant write keys
beta_j A_n ... A_{j+1} k_j and recent read queries
(A_i ... A_{n+1})^T q_i. Use additive bucket pooling and the original incidence
aggregation. Apply the GDN read scale 1/sqrt(head_dim). Beta enters each write
exactly once. No additional bucket-level delta update is applied.

The operators use GDN's normalized key/query features, not the old independent
tied-causal memory projection. That old projection remains registered but frozen
to preserve module initialization ordering. Causal convolution for write hashes
is retained. Exact-operator claims condition on these features and omit the
learned final mixture gate; this is not identical to unreset GDN's convolution
features, and the gate still scales the selected memory output.

An additional batched FLA recurrence with identity initial states and zero
values computes both matrix-product scans without explicit inverse matrices.
GPU checks compare against FP64 explicit products and a dense masked operator,
including gradients, then check full-mixer causality/reset and serialization.
Training starts only if all checks pass. GPU validation performs a few optimizer
steps on random inputs in an isolated model; it does not warm-start the trial.

The Modal volume retains an epoch checkpoint with optimizer, scheduler and RNG.
`collect_gdn_transport.py --watch` writes a comparison at matched epochs and
downloads the final checkpoint. `source/` freezes the experiment overrides;
the base image uses the same frozen bundle as the earlier width sweeps.
