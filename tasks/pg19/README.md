# PG-19 language models

Entry points for the `src/smat_lm` models, unchanged from the runs:

- `prep_pg19_scale.py` -- tokenise PG-19 (GPT-2 BPE) into fixed 16K windows
- `train_pg19_scale.py` -- resumable matched-token training; the model class is
  chosen by `PG19_MEMORY_VARIANT` (`gdn_baseline`, `mamba2_baseline`,
  `updated_transport` for GDN + SMat, `mamba2_fixed_write` for Mamba-2 + SMat)
- `test_pg19_*_gpu.py`, `test_write_hash_fused_gpu.py`, `validate_pg19_transport_opt.py`
  -- the GPU gates each campaign passed before training: kernel outputs and
  gradients against references, causal inference, cached vs full decoding
- `bench_pg19_{reader,write,transport}.py` -- kernel timing

Configurations of the reported runs are in `results/pg19_six_500m/campaign.json`.
Some checks assert source hashes under the campaign's container path
(`/opt/...`); run them with that assertion's root pointed here, or read them as
the record of what was checked.
