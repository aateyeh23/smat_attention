# Context-dependent recall

A 500-step comparison of the current Mamba-2/GDN backbones and their existing
SMat d=2,3,4 variants. Every record is an explicit (context, key, value) triple.
Keys are either unique across contexts or reused in both contexts. Reuse
conditions have the same record count, sequence length, values, record positions,
and query entry indices. Only key identities change.

- `FINDINGS.md`: final interpretation and the three-seed comparison.
- `report.md`: all model/condition tables, parameter counts, and plots.
- `results.csv`: recipes and per-run results.
- `PROTOCOL.md`: design, calibration decisions, and execution history.
- `audit.json` and `audit_results.py`: final independent data/metric checks.
- `jobs.txt`: completed interactive-job accounting.
- `source-{pilot,final}.{tar.gz,json}`: source snapshots and hash manifests.
- `snapshots/one-seed-before-confirmation/`: the complete 32-cell screen before
  extra seeds. All negative/tied variants and the unique-key control are retained.
- Each run folder contains its recipe, learning curve, checkpoint, final metrics,
  and raw test predictions/targets. Optional source-position diagnostics were
  computed from the same saved predictions without further training.

The study uses width64, two layers, T64, two contexts, four random queries,
16 possible values, 500 steps, batch32, and the previous matched AdamW recipe.
Screen loads are4 and8 records. Only4 records receives three seeds123/456/789.
This is an explicit-context variant of joint recall, not an exact replication
of the paper's block-context encoding.

`context_binding_gap_pp` compares correct-query accuracy with accuracy against
the same key's value in the other context. It is undefined for unique keys.
Positive gaps indicate contextual discrimination; raw gains that ignore context
are not sufficient evidence. The key-only oracle reports how well an oracle
that knows the key-value associations but ignores contexts can do.

Reproduce/resume from the repository root:

```bash
sbatch --time=00:20:00 deltaai/run_context_recall.sbatch \
  --records 4,8 --ds 1,2,3,4 --max-minutes 18
sbatch --time=00:20:00 deltaai/run_context_recall.sbatch \
  --records 4 --ds 1,2,3,4 --seeds 123,456,789 --max-minutes 18
module load python/miniforge3_pytorch/2.11.0
python deltaai/collect_context_recall.py
```

The interactive QoS permits one running job per user. Completed cells are skipped;
interrupted training resumes from its optimizer/model/RNG checkpoint. The batch
script also supports two GPUs with one backbone per GPU, but the actual study
used single-GPU jobs because the shorter queue was faster. All training ran on
DeltaAI's ghx4-interactive partition. The existing BF16/custom-kernel pipeline
does not enforce bitwise deterministic training; matching seeds/data does not
imply that repeated executions produce identical floating-point trajectories.
