# MQAR transport with neighboring write-hash gradients

Paused at the user's request after epoch 6/32 on September 17, 00:10:23 CDT.
Modal is stopped with zero tasks and the collector is stopped. The resumable
`w32-d3.pt` is downloaded, its model/optimizer/scheduler/RNG fields verified,
and its digest recorded in `SHA256SUMS`. See `paused.json` and `diagnosis.md`.
The bounded evaluation-only routing probe finished before shutdown; subsequent
analysis used local saved weights, probe output, and cached examples only.

Fresh width32/d3 run, two layers, two heads, head/state 16, seed123,
32 epochs, LR0.01, AdamW/epoch cosine, unchanged cached MQAR data.
One pooled forward write per token and four distinct weighted hyperplane
reads. Only the write-hash backward estimator changes from the previous
transport model: zero-forward-weight neighbors contribute their gradients.

Modal app: https://modal.com/apps/archerdwang/main/ap-NzjPeJyFxnpnf17VzPcp9o
Checkpoint volume: smat-gdn-transport-w32-d3:/seed123/neighbor_write_grad/
The original transport checkpoint remains paused at epoch23 in /seed123/.
GPU numerical/reference checks gate training. Checkpoints include optimizer,
scheduler and RNG state and are saved at each completed epoch. The collector
refreshes history and matched-epoch comparisons every 60 seconds and downloads
the final checkpoint. The previous transport history ends at epoch23.

Frozen override sources and SHA256 hashes are in source/ and manifest.json.
The base image/data are the same frozen bundle used for the previous sweep.

GPU checks passed for both estimator modes: FP64 transport algebra, dense
write-gradient/payload reference, routed operator, complete-mixer backward,
causality, reset isolation, checkpoint restoration and forward equivalence.
Short B256/T256 mixer benchmark: original 0.0137s, neighbor-gradient 0.0146s.
These are isolated mixer timings, not full-trainer throughput.
