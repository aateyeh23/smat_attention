# PG19: six models, 500M tokens each

Launched September 17, 2026 at the user's request. The six arms are plain GDN,
GDN + SMAT d=2/d=3, plain Mamba-2, and Mamba-2 + SMAT d=2/d=3. d=4 is excluded.
Target: 500,000,000 tokens per arm, rounded up to full 16,384-token windows
(500,006,912 actual tokens). Each arm uses one H100 80GB; six GPU tasks were
launched concurrently. GDN + SMAT d=2/3 were paused, then relaunched with faster
kernels at the user's request. The other four continue in the original app.

Active Modal app: [ap-hpwI1brPNCjWfs3nvgW2VP](<modal-app>).
Optimized GDN continuation app: [ap-IeUwYiy5yeh7EdOo5GJEWm](<modal-app>).
The first app, `ap-ZqVoo1IBt7X3njBJOckg0K`, failed on a launcher import before
training and was stopped. The corrected launcher is self-contained.

## Matched recipe

| Arm | Trainable parameters |
|---|---:|
| GDN baseline | 161,603,264 |
| GDN + SMAT d=2 | 163,191,968 |
| GDN + SMAT d=3 | 166,105,088 |
| Mamba-2 baseline | 172,769,920 |
| Mamba-2 + SMAT d=2 | 176,913,280 |
| Mamba-2 + SMAT d=3 | 188,565,760 |

These are matched backbone shapes, not exactly matched total parameter counts.
Counts and complete configurations are recorded in `campaign.json`.

- 16 layers, residual width 768, SwiGLU FFN width 2048, tied GPT-2 embeddings.
- Fixed 16K context, seed 123, identical shuffled unique within-book data windows
  from `smat-pg19-scale-data-v1:/pg19-16k-2b`.
- GDN: six heads, key/value dimension 128. Mamba-2: 24 heads, head/state dimension
  64, native expansion two. Each backbone has 98,304 recurrent state elements
  per layer. Plain baselines keep uninterrupted full-sequence recurrence;
  SMAT arms reset local convolutions/recurrence at the midpoint.
- SMAT: one hard write, four weighted reads, rank-64 reader, corrected neighbor
  write gradients and fused Triton kernels. d=2 uses point summaries; d=3 uses
  hyperplane summaries. Baselines have no routing or hash-balancing auxiliary loss.
- Global batch eight sequences (131,072 tokens), microbatch two except Mamba-2
  SMAT d=3 at one. BF16 autocast, fused AdamW, LR 3e-4, 40M-token warmup, cosine
  decay to 10% over the 500M-token schedule, gradient clipping at one.
- Fresh initialization for all six; old pilot checkpoints remain separate.

Each arm first runs GPU correctness gates, then a four-step full-size pilot and
resumable checkpoint, then automatically continues the same run to 500M tokens.
The shared trainer restores optimizer/RNG/data cursor from that pilot checkpoint.
The cumulative training cap is 11.8 hours per arm; reaching it preserves resumable
state. This is a cap, not a promise that every arm will finish within that time.

## Results and recovery

[Live report](report.md) and `summary.json` refresh every minute while the local
collector is running. `collector.pid` records its PID; `collector.log` records
updates and transient collection errors. The collector stops when all six arms
are complete, paused or failed. It does not launch training or send notifications.
Validation comparisons use identical training-token counts and the same validation
subset; per-arm latest scores may correspond to different amounts of training.

Checkpoint volume: `pg19-w768-rank64`, root
`/seed123/six-500m-20260917/{gdn,mamba2}-{baseline,smat-d2,smat-d3}/`.
Every arm retains its source hashes, copied source files, runtime package record,
recipe, GPU validation result, metrics and execution phase. `latest.pt` contains
model, optimizer, RNG and data cursor; it is saved every ten minutes. Weights-only
milestones are retained at 250M tokens and completion. Validation uses 64 fixed
rows every 50M tokens and the full validation set on completion.

Launch/recover (same source and recipe required):

```bash
modal run --detach modal_pg19_six.py::sweep
python collect_pg19_six.py --watch
```

Only run one collector per output directory; it holds an advisory lock.
To recover a single stopped arm, invoke `modal_pg19_six.py::run` with its `--family`
and `--d` (1 identifies the plain baseline). Do not duplicate an active arm.

## September 17 update: GDN + SMAT optimized continuation

At the user's request, GDN + SMAT d=2 and d=3 were gracefully paused with
resumable checkpoints at 39,452,672 and 34,603,008 tokens (steps 301 and 264).
The GDN baseline and all three Mamba2 arms continue training.

Separate checkpoint-copy pilots validated FP32 kernel optimizations at
55.6k tokens/s for GDN d=2 and 42.3k for d=3, approximately 2.3x/2.1x faster.
The user subsequently authorized resumption with the faster kernels. Both
original checkpoints were copied with SHA-256 verification to
`pg19-w768-rank64:/seed123/six-500m-20260917-opt-v1/gdn-smat-d{2,3}/`.
The original source files, recipes and checkpoints remain in their parent
directories. Model, optimizer, RNG, cursor, evaluation schedule and the 500M
token target are retained. Each new arm has `migration.json` provenance and
an audited `tiled_fp32_v1` source manifest. The original directories contain
`continuation.json` pointers, which the live collector follows automatically.

Launch logs are in `optimized-resume.log`; details of the numerical checks
are in `../pg19_transport_opt/README.md`. To recover an optimized arm after
it stops, use `modal_pg19_six.py::run --family gdn --d 2` (or 3) with
`--campaign six-500m-20260917-opt-v1 --optimized`. Do not duplicate a live arm.
