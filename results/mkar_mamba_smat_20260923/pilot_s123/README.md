# Incidence normalization plus writer-feature gradient isolation

One seed123, separate k1/k2/k3 models, one job/three fits. This is the fourth
cell of the normalization x gradient-isolation pilot. Parent, either change
alone, and their combination are all retained, including failed results.

The only change from the normalized pilot is detach(key_source) before writer
address assignment. Routing/balance gradients cannot train the shared causal
key convolution; content retrieval still trains it and the shared projection.
Router weights retain their gradients. Forward values and parameters at
initialization match the normalized parent, which validation checks.

The existing geometry-derived inverse-incidence-density normalization remains.
Local Mamba, shared memory features, independent write gates and key-dependent
delta transitions remain unchanged. No added parameters, task-specific token
conditions, targets in forward, changed data/loss or extra updates.
Same original task/evaluator, width256/layer1, batch32, 12000 updates, LR3e-4,
AdamW WD0.1, FP32 and routing balance0.01; use only final checkpoints.

This is an exploratory hybrid memory architecture, not a routing-only result.
A selected candidate requires four independent-seed confirmations and a
matched full-global control with the same gradient-isolation setting.
