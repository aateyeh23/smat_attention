Source-verified MQAR result: GDN + SMAT d=3, width32

Final epoch32 accuracy: 81.77875% versus GDN 69.86219% (+11.91656 percentage points).
All15 required even-epoch gates,4 through32, passed strictly. This is the final epoch32 checkpoint, not a selected peak.

| Key–value pairs | GDN + SMAT | GDN | Margin (pp) |
|---|---:|---:|---:|
| 4 | 99.72% | 99.98% | -0.25 |
| 8 | 98.56% | 99.71% | -1.15 |
| 16 | 90.86% | 92.91% | -2.05 |
| 32 | 70.67% | 46.49% | +24.17 |
| 64 | 49.08% | 10.22% | +38.86 |

The aggregate gain comes from the32- and64-pair cells; the4-,8-,16-pair cells remain slightly below GDN.

Protocol: two layers, width32, two heads, head/state16, seed123. One learned hard bucket write and four distinct learned softmax-weighted hyperplane reads per head. Content-only writes, detached hash input, full neighboring write gradients, no scalar transport decay, and incidence rescaling. No full unreset GDN value-state recurrence crosses the memory boundary.
Optimizer: original single-group AdamW, initial LR0.01, weight decay0.1,32-epoch cosine schedule; no gradient clipping. Original cached MQAR training/evaluation mixture and baseline history are preserved.

Best observed validation checkpoint was epoch24: 82.42562%. It is reported separately from the final comparison.

Files: `w32-d3.pt` is the final resumable checkpoint (model, optimizer, scheduler, RNG state). Local immutable snapshots also include epochs04,08,16,24. All32 epoch checkpoints remain on Modal volume `smat-gdn-w32-d3-iterations`, under `/22_verified_transport/checkpoints/`.
`comparison.png`, `comparison.pdf`, `comparison.tex`, and `gate-comparison.csv` contain the plot, LaTeX table, and every gate comparison. `final-verification.json` records checkpoint integrity and protocol checks.

Source integrity: `training-sources.json` and `gpu-validation-sources.json` match `expected-sources.json`; overlaid module hashes also match the local launch snapshot. Both entrypoints use Python safe-path mode. Preserve that setting when reproducing the run so the working directory cannot shadow the intended modules.
Earlier trials01–21 used a frozen bundled router during training. Their claimed routing ablations are unverified; raw scores and checkpoints remain archived. See `../SOURCE-RESOLUTION-AUDIT.md`. The earlier standalone restored-write-gradient run has the same caveat. The pure-GDN baseline is unaffected.

Scope: this is a single-seed MQAR validation comparison. It does not establish selective-copying or NLP gains; no such downstream runs were launched here.

Training completed and the Modal app was stopped.
