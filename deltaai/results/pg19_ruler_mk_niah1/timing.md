# Cached evaluation timing

One H100 per model; batch size 4. Evaluation loop duration includes initial kernel compilation and per-batch saving, but excludes checkpoint loading and earlier checks.

| Model | 500-example duration (min) | Median warm batch (s) | Score |
|---|---:|---:|---:|
| gdn-baseline | 5.96 | 2.50 | 0.00% |
| gdn-smat-d2 | 12.88 | 5.42 | 0.00% |
| gdn-smat-d3 | 10.06 | 4.61 | 0.00% |
| mamba2-baseline | 5.62 | 2.39 | 0.00% |
| mamba2-smat-d2 | 10.70 | 4.78 | 0.00% |
| mamba2-smat-d3 | 9.80 | 4.43 | 0.00% |
