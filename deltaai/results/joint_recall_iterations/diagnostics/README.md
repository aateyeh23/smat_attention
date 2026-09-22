# Memory signal diagnostics

`probe_joint_recall_memory.py` runs read-only forward hooks on a saved checkpoint.
The first probes use the first 32 validation tables of the large C8/K16 cell.
GDN gate estimates use actual layer inputs; memory feature norms and memory output
are captured from the actual forward computation. These are small-batch mechanism
measurements, not full-dataset accuracy estimates.

Observed contrast:

- `mamba_memory.json`: default SMat Mamba, completed epoch 32. Median query
  feature norms remain about 0.42–0.48 across query-position quartiles. Median key
  norms are about 0.52 in layer 1 and 8.3–8.6 in layer 2. This model's full
  validation accuracy is 79.63%.
- `stable_gdn_memory.json`: SMat GDN with tau3088 initialization and no weight
  decay, checkpoint epoch 23. Median transported key and query norms are zero in
  every quartile in both layers. The mean learned memory gates are approximately
  [0.00098, 0.01196] and [0.00038, 0.00025]. Scalar retention across the target
  value-to-query path is also extremely small. The model is near chance.

The current GDN memory branch carries GDN scalar forgetting and delta transitions
into distant keys and queries. The current Mamba extra-memory branch has time
decay disabled. Thus identical geometric routing does not imply identical preserved
memory signals. The probes establish vanishing signals in this failed GDN
checkpoint; they do not establish which optimization event caused the failure.
Long-retention initialization alone did not prevent later collapse.

Target-value routing scores are deliberately qualified: causal convolutions can
carry a value's information at neighboring positions, and different heads/layers
can divide the work. Even successful Mamba has weak scores when considering only
the literal value position. Consequently these scores must not be interpreted as
a complete explanation of recall or as proof that the mask learned no useful
routing. Norm comparisons across backbone families are descriptive, not directly
calibrated measures of representational capacity.

An exact-checkpoint counterfactual strengthens the scalar-decay diagnosis:
`stable_gdn_final_memory.json` and `stable_gdn_counterfactual_no_decay.json`
both use the completed epoch32 checkpoint and the same32 validation tables.
Only the latter forward pass disables scalar decay in the extra memory.
With decay, all quartiles have zero median transported key/query norms in both
layers. Without it, median query norms are approximately0.11–0.16, and key norms
approximately0.001–0.009. Accuracy remains near chance in both cases (5.57% and
5.54% on this small batch). Removing decay restores numerical memory signals in
this checkpoint but does not retroactively train useful retrieval. Full no-decay
training is a separate controlled ablation.

`stable_gdn_initial_memory.json` shows that the tau3088 initialization starts
with nonzero transported features. Their later disappearance is a training outcome,
not a constructor failure. A lower-learning-rate original SMat GDN run subsequently
began learning successfully; the failed checkpoint's behavior is therefore not a
claim that every trained SMat GDN must lose its memory signal.

The CPU reconstruction in `mixed_lr003_decay_probe.log` quantifies that contrast
at epoch19: on C8/K16, the original decay-enabled SMat GDN has mean scalar
value-to-query retention approximately[0.9804,0.1109] across its two first-layer
heads. Its validation macro accuracy is88.16% at that checkpoint, with66.16% on
the largest tables. These retention estimates omit delta transitions, so they
are not full memory-transfer magnitudes. They do show that the unchanged model
can learn to preserve a useful long-range scalar path with a lower learning rate.
