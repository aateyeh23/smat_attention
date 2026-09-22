# Prepared write-hash gradient change

Prepared after the user's acknowledgement of the gradient diagnosis. Training
remains paused at epoch 23. This does not modify the downloaded checkpoint,
historical source snapshot, or reported results.

The new opt-in config is `deltaai/zoo_gdn_transport_neighbor_configs.py`.
The original config and model default keep the original estimator. The Modal
entrypoint accepts `--neighbor-write-grad` and puts that experiment under
`/seed123/neighbor_write_grad/`, separate from the paused trial. It has not run.

## Implementation

- Compute the existing hard straight-through weights, including zero-weight
  neighbors. Pool only the first, nonzero route, with its scalar weight detached.
- Attach an identity autograd operation to the resulting bucket memory. In
  backward, each candidate write weight receives the contraction of that
  bucket's memory gradient with the write's key/value outer product.
- Return the memory gradient unchanged to ordinary pooling, which handles
  key/value gradients. The adapter has no additional gradient to payloads,
  preventing double counting.
- Keep four distinct query reads, planted anchors, auxiliary balancing, model
  parameters, and the transport operator as before. Evaluation under no-grad
  uses the existing hard path. The estimator supports first-order gradients.

This restores the neighboring-cell difference in the hard-STE estimator for
additive pooling. It does not make the hard quantizer differentiable, change
the reader's discrete top-k estimator, or guarantee that training will improve.
Neighboring bucket-gradient gathers add backward work; no speed measurement
has been made for the change.

## Validation status

Static syntax and whitespace checks only; no model computation or GPU job was
run. The GPU-only validation suite was extended to compare outputs and all
write/payload gradients against a dense reference, include duplicate bucket
indices, verify the routed operator's hash gradients, and exercise full-mixer
output equivalence plus existing causality/checkpoint checks. Those new numerical
checks are pending and gate training in the launch entrypoint.
