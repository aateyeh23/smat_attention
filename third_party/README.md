# Third-party code, as the runs used it

`zoology/` -- the Zoology copy frozen with the recall campaigns (every frozen
bundle's copy was byte-identical to this one).  Two properties matter for
anyone timing Mamba-2 with it: `zoology.mixers.mamba2.Mamba2.forward` calls
`torch.autograd.set_detect_anomaly(True)`, which stays on for the rest of the
process, and its non-fused path runs an unfused `nn.Conv1d`.  Neither changes
the computed values.  The SMat Mamba-2 mixer (`src/smat_lm/zoo_smat_mixer.py`)
reimplements the forward pass and is affected by neither, so throughput of a
plain `Mamba2` from this copy is understated relative to it.

`log_linear/` -- the upstream Log-Linear attention kernels (WEAK binary levels),
isolated from their Transformers registrations; `upstream/PROVENANCE.md` gives
the upstream and FLA commits.  They import `jaxtyping`, which nothing else here needs.  `upstream_adapter.py` wraps them for the width-16
joint-recall models: it pads the feature width to 16, which *crops* wider heads,
so call `hattention_kernel` directly for other widths.  The upstream GDN kernel
failed a gradient check against the dense reference (`src/smat_lm/zoo_log_linear.py`)
and is not used; see `results/loglinear_backend_timing/report.md`.
