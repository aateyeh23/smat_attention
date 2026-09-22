# RULER MK-NIAH-1: PG19 500M checkpoints

Updated: 2026-09-18T01:04:06.903223+00:00

Official task generator and substring-recall scoring. GPT-2 tokenizer, base prompt,
16,384-token budget including up to 128 greedy output tokens. Same 500 examples for all arms.
Models retain their trained fixed 16K geometry. No fine-tuning or checkpoint updates.

| Model | State | Examples | Accuracy (%) |
|---|---|---:|---:|
| gdn-baseline | complete | 500/500 | 0.00 |
| gdn-smat-d2 | complete | 500/500 | 0.00 |
| gdn-smat-d3 | complete | 500/500 | 0.00 |
| mamba2-baseline | complete | 500/500 | 0.00 |
| mamba2-smat-d2 | complete | 500/500 | 0.00 |
| mamba2-smat-d3 | complete | 500/500 | 0.00 |

Decoding uses a prompt prefill followed by cached recurrent updates. Earlier
full-window and speculative outputs are excluded from these scores.

Scores are provisional until all examples finish. These are PG19 base models,
so failures can reflect prompt-following ability as well as retrieval.

[Official RULER](https://github.com/NVIDIA/RULER/tree/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a)
[Cached evaluation app](https://modal.com/apps/archerdwang/main/ap-q0XZKzTUemQmrdE6QRN2pu)
[Mamba-2 d3 continuation](https://modal.com/apps/archerdwang/main/ap-R3QklB1SdwIR6hoaeMjldT)
