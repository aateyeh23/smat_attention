# Block-context joint recall

The user reduced this to seed123 and requested further task iterations.
The original three-seed continuation chain is cancelled. Current scheduling
is in ../joint_recall_iterations/campaign.json and run_joint_recall_iterations.sbatch.
The original manual continuation command below is historical and should not be run.

Status and accuracy: [report.md](report.md). The report labels completed runs
and interim validation results separately. Final-epoch test is the primary
endpoint; test at the best validation checkpoint is secondary.

- [PROTOCOL.md](PROTOCOL.md): task, controls, budget, adaptations and execution history.
- [data/manifest.json](data/manifest.json): fixed datasets, shapes and SHA256 hashes.
- [data_audit.json](data_audit.json): independent decoding, masked answers, paired conditions and split disjointness.
- [padding_audit.json](padding_audit.json): compatibility padding preserves all original token data and targets.
- [source/manifest.json](source/manifest.json): frozen training/model Python files used by the full campaign.
- [planned_runs.json](planned_runs.json): all24 model/condition/seed runs.
- [lookup_controls.json](lookup_controls.json): analytical context/key ablations.
- [result_audit.json](result_audit.json): completed-run budgets, validation selection and prediction recomputation.
- `results.csv`: final and validation-best test scores by table size and macro average, when runs finish.
- `learning_curves{,_by_load}.{png,pdf}`: validation trajectories, refreshed by the collector.

Each run has recipe.json, resumable checkpoint.pt, best.pt, metrics.jsonl,
status.json and eventually result.json plus raw test prediction NPZ files.
All training uses ghx4-interactive, account bekw-dtai-gh. Full campaign starts
at Slurm3172175; continuation job IDs are retained in continuation_jobs.txt.
The failed unpadded pilot is retained separately in ../joint_recall_unpadded_pilot.

From the repository root, refresh reports without a GPU:

```sh
module load python/miniforge3_pytorch/2.11.0
python deltaai/audit_joint_recall_results.py
python deltaai/collect_joint_recall.py
```

To resume manually after an interrupted chain (first ensure no campaign job is
running or pending):

```sh
sbatch deltaai/continue_joint_recall.sbatch
```

The launcher uses the frozen training snapshot automatically. It resumes the
same optimization schedule and data-order cursor. It never ends a run early
because of its accuracy. The continuation launcher stops once all24 runs finish
or on a training/audit failure, and otherwise submits the next two-hour slice.
