# Synthetic memory experiments on DeltaAI

The primary question is now **whether SMat beats its exact corresponding
backbone after 500 steps**. Earlier 200/1,000-step calibration histories are
retained, but are not substituted for the 500-step comparisons.

- `report.md`: live tables, accuracy-versus-load plots, and learning curves.
- `FINDINGS.md`: final interpretation, all seven scientific questions, and limitations.
- `primary-main500-overview.png` / `.pdf`: the current eight-model main comparison.
- `confirmation.md`: focused three-seed tables, including paired gains, exact
  query-set accuracy, query binding, and total/trainable parameter counts.
- `results.csv`: completed-run metrics, parameter counts, and data hashes.
- `paired_comparisons.csv`: each SMat run paired with its same-seed backbone.
- `PROTOCOL.md`: generation, fairness controls, calibration and execution history.
- `screen500/`: width-32, all eight models, one-seed 500-step screening.
- `main500/`: width-64 500-step runs, gated by completed pilots.
- `snapshots/`: immutable copies of tables/plots before later stages or seeds.
- Each run directory contains its recipe, metric history, full resumable
  checkpoint, and final independent-test results. Later runs also retain raw
  final predictions and a query-binding diagnostic.

Primary families are `mamba2` and `gdn_current`. The latter uses the repository's
selected transport SMat recipe (learned scalar decay, no output rescaling).
Runs labeled `gdn` use an older delta-pool SMat recipe and are retained as legacy
controls. Both GDN families use the identical native GDN baseline at d=1.
The initial `screen500-one-seed` snapshot predates this recipe correction;
see PROTOCOL.md for its provenance.

Mamba2 and GDN head/state dimensions are 16, with two layers. Mamba2 preserves
its native 2x expansion (four heads at width32, eight at width64); GDN uses two
heads at both widths. Every baseline/SMat comparison has matched backbone
sizes, training/evaluation data, optimizer, schedule, steps, and seed. The existing
SMat variants include a halfway recurrence reset and their existing routing
auxiliary loss; this is a comparison of the existing MQAR architectures, not an
ablation isolating memory count alone. Extra routing parameters are reported.

Inputs are random synthetic assignments, independent of the SMat mask. State
tracking requires the latest of three updates per entity; variable tracking uses
disjoint H=2 chains (two variable links followed by a terminal value assignment).
Assignments are shuffled and randomly placed, with query-only supervision.
All architectures use the same padding, including the 192→256 compatibility fix.

The unconditional random-value chance rate is 6.25%. Query-blind context controls can
exceed it substantially. Raw accuracy gains below those controls do not by
themselves establish entity-specific tracking or chain following. Main runs also
report a binding gap: accuracy for the queried value minus accuracy for another
entity/chain's value from the same example; query-independent strategies have
expected gap zero.

From the repository root, an individual matched screen can be run/resumed with:

```bash
sbatch --time=00:20:00 deltaai/run_synthetic_memory.sbatch \
  --task state --stage screen500 --steps 500 --loads 4,8,16,32 \
  --families mamba2,gdn_current \
  --ds 1,2,3,4 --min-length 64 --eval-every 100 \
  --eval-examples 1024 --test-examples 4096 --max-minutes 18
```

The parallel script supports one, two, or four allocated GPUs and distributes
d groups across them. Each worker runs its models sequentially. Completed cells
are reused and interrupted cells resume their model, optimizer, schedule, and RNG.
All training uses account `bekw-dtai-gh`, partition `ghx4-interactive`.
No Modal or external GPU service is used.

The complete primary one-seed sweep is preserved in
`snapshots/complete-current-one-seed-before-confirmation/`. It contains all
requested main loads plus easy anchors, before any extra training seeds.
The selected three-seed checks can be resumed with:

```bash
sbatch deltaai/run_synthetic_memory_confirmation.sbatch
```

The native GDN width64/state/load4/seed123 recipe produced different later
training trajectories on two executions despite identical first-step loss and
data hashes (69.87% in the legacy-family run, 45.98% in the current-family run).
The pipeline does not enforce bitwise deterministic custom kernels. Both results
are retained; they are execution repeats, not independent seeds. Interpret the
three-seed comparison together with this baseline sensitivity.
