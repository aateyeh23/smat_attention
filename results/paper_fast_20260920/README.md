# Fast joint-recall Log-Linear runs

Ten fresh runs: Mamba-2 and GDN, each seeds 123, 1, 2, 3, 4. User requested launch after the isolated optimized benchmark. Same frozen dataset, architecture, seed-dependent batch ordering, optimizer and full 32-epoch budget as the dense campaign; only the validated operator backend changes. No dense checkpoint is resumed or overwritten.

Mamba uses the upstream Triton WEAK implementation with fixed block-16/four-warp/two-stage settings and BF16 value inputs. GDN uses the FP32 binary-tree implementation with batch chunks 256 for T<=256 and 64 otherwise. Kernels, operator wrappers, and mixer are frozen under source/ with SHA256 manifest. Original trainer/backbone sources remain in paper_seeds_20260920/source/joint and are checked against that campaign's manifest at dispatch.

Initial the GPU cluster array: 3182641, 10 one-GPU workers. Each two-hour allocation runs up to 105 minutes and automatically continues from an exact saved trainer checkpoint if necessary. Source and backend provenance are saved for every run. Any nonzero process return stops that worker for inspection. Dense campaign remains separate and continues running.

Reports must distinguish final epoch equal-cell validation accuracy (Table 3 metric) from independent final-test scores and best-checkpoint scores. Do not combine dense and optimized replicates into a single five-seed mean.
