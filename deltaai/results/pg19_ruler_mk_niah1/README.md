# PG19 checkpoints on RULER MK-NIAH-1

Evaluation apps: [five arms](https://modal.com/apps/archerdwang/main/ap-q0XZKzTUemQmrdE6QRN2pu), [Mamba-2 SMAT d3](https://modal.com/apps/archerdwang/main/ap-R3QklB1SdwIR6hoaeMjldT)

Six immutable 500,006,912-token checkpoints: GDN and Mamba-2, each plain,
SMAT d=2, and SMAT d=3. Training checkpoints are read only.

## Protocol

- NVIDIA/RULER commit `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`.
- Official `niah_multikey_1` generator: essay haystack, four word keys,
  one seven-digit value per key, one queried key; seed 42, 500 examples.
- Official base template and answer prefix; no demonstrations or fine-tuning.
- GPT-2 tokenizer; 16,384-token total budget with 128 tokens reserved for output.
  Actual prompts are 15,662–16,256 tokens, following the official generator.
- Greedy decoding, up to 128 tokens or GPT-2 EOS; original fixed 16K model
  geometry, right padding only. All arms see identical input token IDs.
- Official case-insensitive substring recall metric, plus descriptive breakdowns
  by whether the target value occurs before or after token 8,192.
- Checkpoint/source SHA-256 checks and GPT-2/tiktoken token-ID agreement
  run before inference. Additional cached/full comparisons are disabled
  at the user's request.

`report.md` and `summary.json` refresh every minute while the collector runs.
Per-arm directories retain predictions, progress, logs, and provenance. Official
upstream sample `index` is a character offset rather than a unique example ID;
`sample_id` records the unique line number without altering generated examples.
The upstream essay downloader's ordering and text conversion are retained, with
parallel HTTPS downloads; original/resolved URLs and raw hashes are recorded.

Results are on Modal volume `pg19-ruler-evaluation` under
`/mk-niah1-16k-500m/recurrent/<arm>/`. The app uses one H100 per arm. Batch size
starts at four and retries at a smaller size if GPU memory is exhausted.
Each completed batch is saved and committed; unchanged evaluations can resume
from saved predictions. Do not launch a duplicate of a live evaluation.

Collect: `/u/archerdw/venvs/modal-mqar/bin/python deltaai/collect_pg19_ruler.py --watch`

These PG19-only base models have not been instruction-tuned. A low score alone
cannot distinguish retrieval failure from inability to follow the prompt.

## Cached recurrent decoding

`deltaai/lm/pg19_recurrent.py` provides `RecurrentDecoder.prefill(input_ids)`
and `step(next_ids)`. Prefill uses the original fixed-window forward pass,
retains the SMAT summaries, and initializes convolution and recurrent states.
Each generated token then updates those states without recomputing the prompt.
GDN SMAT additionally updates its boundary transport; Mamba-2 SMAT updates its
decay. Decode work is independent of the already-consumed sequence length for
this fixed trained geometry, but includes SMAT summary scoring and reads.

The current adapter supports SMAT prompts beyond the trained midpoint (8,192)
and generation within the 16,384-token window. All benchmark inputs satisfy
these limits. It does not implement arbitrary-length streaming or crossing the
SMAT midpoint during generation. Inference runs under BF16 autocast; recurrent
and full-window kernels need not produce bitwise-identical logits.

The cached evaluation starts fresh on all 500 examples for every model. Earlier
`full` and `fast` predictions are retained as historical artifacts on the results
volume but are excluded from the current report. The previous speculative
method has been superseded. No repeat-run numerical-variation checks or further
cached/full correctness runs are scheduled.
