# Two-key retrieval and AND

Completed36 runs. Start with [FINDINGS.md](FINDINGS.md): local AND is learned
perfectly; retrieval and end-to-end results do not show a consistent SMat gain.

Three matched conditions isolate bit retrieval, local conjunction, and their
end-to-end combination. All use the same memory rows and queried bit pairs;
the local control receives the two bits directly. Truth cases00/01/10/11 are
equally frequent. Each model trains for500 steps on DeltaAI ghx4-interactive.

- `PROTOCOL.md`: design and chronological execution decisions.
- `report.md`, `results.csv`, PNG/PDF plots: collected completed runs.
- `FINDINGS.md`: final interpretation after confirmation.
- `audit.json`, `audit_results.py`: independent token/metric and pairing checks.
- `expected_grid.json`: the intended final run grid.
- `shortcut_controls.json`, `diagnose_shortcuts.py`: privileged count/one-bit controls.
- `jobs.txt`, source archives and hash manifests: execution and source provenance.
- `snapshots/one-seed-before-confirmation/`: the complete initial12-run screen.
- Each run: recipe, checkpoint, learning curve, final metrics and raw predictions.

From the repository root:

```bash
sbatch --time=00:20:00 deltaai/run_boolean_recall.sbatch \
  --records 4 --ds 1,3 --seeds 123,456,789 --max-minutes 18
module load python/miniforge3_pytorch/2.11.0
python deltaai/collect_boolean_recall.py
python deltaai/results/boolean_recall/audit_results.py
python deltaai/results/boolean_recall/diagnose_shortcuts.py
```

Completed cells skip; interrupted training resumes from model/optimizer/RNG
checkpoints. Resubmit if the job time slice ends before the grid completes.
Use `--root` for an intentional fresh output directory. Collector and audit
target the default study directory. Training is not guaranteed bitwise
deterministic. Sample SD across seeds does not describe test-sample uncertainty.
