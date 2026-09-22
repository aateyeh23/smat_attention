# PG-19, 300M training tokens

Per-arm evaluation of the 8-layer, width-384 models trained by `tasks/lm.py`
(arms `mamba2`, `gdn`, `attn`, `smat` with `--d 2/3/4`) at 16K and 32K context:
`<run>.json` holds mean NLL, perplexity, bits per byte and NLL by position;
`<run>.csv` the per-position curve; `<run>.png` its plot.  The SMat arms use the
Mamba-2 backbone, so Mamba-2 is their matched baseline; `gdn-d1` is a separate
backbone.
