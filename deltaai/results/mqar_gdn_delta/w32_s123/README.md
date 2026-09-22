# MQAR per-profile delta memory experiment

Width 32, SMAT d=3, head dimension 16, seed 123, two layers. Same data, batch sizes (256 train / 32 validation), interleaved batch order, learning rate 0.01 and 32-epoch cosine schedule as the existing width-32 interleaved GDN/SMAT comparison. Fresh initialization; no extra seeds.

Only the SMAT memory update changes: for each profile, merge a token's duplicate route weights, then chronologically apply S <- S + beta * route_weight * k (v - k^T S)^T. Keys are L2-normalized. There is no extra decay. The local GDN state and convolution still restart at the midpoint; hash geometry, read direction, and read gate remain the same.

Implementation stably packs each profile into a variable-length sequence and uses FLA chunk_gated_delta_rule (chunk size 16), with gradients through its final state. It does not reconstruct the full-sequence GDN state. TRITON_F32_DEFAULT=tf32x3 avoids accumulated TF32 truncation error in forward/backward; default TF32 initially differed from the reference by up to ~0.004 in state and ~0.02 in gradients on the repeated-write stress case. With tf32x3, the tested maximum state/gradient errors were below 1e-6 / 4e-6. Existing comparison runs used their original default precision.

The priority Slurm step temporarily suspends the existing sequential worker and its trainer while this experiment trains, then automatically resumes them. Epoch checkpoints retain the original 32-epoch scheduler even if the priority time slice ends. The delta task is first in the campaign manifest for future allocations.

Validation: `test_delta_pool.py` passed final-state and gradients for 19-token/4-route, 137-token repeated-profile/duplicate-route, and 70-token/1-route cases; overwrite/empty cells; full mixer gradients and midpoint isolation; all three data lengths and both soft/hard routing. Shared-GPU microbenchmark (batch 32, mixer only): delta 23.7–24.6 ms versus additive 19.7–24.4 ms per forward/backward. This excludes startup compilation and the full training model/optimizer.
