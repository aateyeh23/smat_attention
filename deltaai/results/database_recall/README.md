# Database-query experiment

Start with [FINDINGS.md](FINDINGS.md). Eight500-step pilot runs completed; all
models remain near50% conjunction balanced accuracy. The broader sweep was
stopped at this learning floor. [PROTOCOL.md](PROTOCOL.md) records the original
design and the separately labeled balanced-loss follow-up.

Artifacts: `report.md`, `results.csv`, PNG/PDF plots, `audit.json`,
`object_binding_diagnostics.json`, `jobs.txt`, and source archives/manifests.
Each run directory contains recipe, checkpoint, learning curve, results,
`test_predictions.npz`, and `heldout_predictions.npz`.

From the repository root:

```bash
sbatch deltaai/run_database_recall.sbatch --records 4 --ds 1,3 --seeds 123
sbatch deltaai/run_database_recall.sbatch --records 4 --ds 1,3 --seeds 123 --loss balanced
module load python/miniforge3_pytorch/2.11.0
python deltaai/collect_database_recall.py
python deltaai/results/database_recall/audit_results.py
```

Completed runs are skipped; interrupted runs resume from their checkpoints.
Submit the two training commands sequentially under the interactive QoS.
Use `--root` for a fresh output directory if intentionally rerunning the study.
The collector/audit target this study's default directory. The audit checks its
eight-run grid. Training is not guaranteed bitwise deterministic.
