# Normalized Mamba delta-memory SMat: four confirmation seeds

Seeds456,789,2026,2027; independent k1/k2/k3 models; four jobs/twelve fits.
Seed123 selected the architecture and remains in the five-seed report.
Selection requires all three final pilot scores >=99%, specified before k3
finished. All confirmation outcomes are retained; no best-checkpoint selection.

Only seeds differ from the frozen pilot. Same original task, BCE+balance0.01,
width256/layer1, batch32, 12000 updates, LR3e-4, AdamW WD0.1 and FP32.
The local Mamba recurrence is unchanged. The distant memory uses shared learned
query/key features, causal writer features, delta transport without scalar
decay, and division of its read by mean incidence before the existing gate.
Writer-address features are detached; content retrieval still trains the
shared convolution/projection. The same setting is used for every seed.
This normalization is geometry-derived and uses no task labels or token IDs.
No additional architecture change, parameters or training steps are introduced.

Native and softmax comparisons match width and exposure, not all resources.
Report the full-global memory control alongside routed results; do not infer
VC causality or a routing-specific advantage from native-baseline gains.
