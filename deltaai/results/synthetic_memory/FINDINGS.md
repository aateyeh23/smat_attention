# Synthetic memory findings at 500 steps

The clearest positive result is **current GDN transport SMat d=4 on four-entity
state tracking at width 64**. It wins all three observed seeds and reaches
83.57% ± 9.08% query accuracy, versus 46.08% ± 36.51% for native GDN. Its positive
query-binding gap and exact-set accuracy establish entity-specific tracking.
This is a narrow low-load learning result: the advantage does not grow with load.
There is no robust Mamba-2 advantage in the state confirmations.

All main comparisons use 500 steps, two layers, matched backbone head/state
dimensions 16, batch 32, AdamW LR 0.01, weight decay 0.1, cosine decay to zero,
clipping 1, and BF16. Data/order/evaluation are matched within seed; training
examples are fresh, validation and test independent. Final tests contain 4,096
examples. Training ran exclusively on DeltaAI `ghx4-interactive` GH200 GPUs.
Earlier 200/1,000-step calibrations remain in the archive and are not used as
substitutes for 500-step results.

The complete one-seed primary sweep contains 64 width 32 and 104 width 64 cells.
The main width 64 sweep includes every requested load plus easy anchors.
Confirmations use seeds 123/456/789 at five settings, retaining both backbones
and all d=2,3,4 variants: 120 cells, 40 reused and 80 newly trained. Other loads
remain one-seed evidence. Error bars and ± values are sample standard deviations,
not confidence intervals; three seeds are a small sample.

At width 64 with four entities:

| Model | Query accuracy (%) | Exact four-query accuracy (%) | Binding gap (pp) | Total / trainable parameters |
|---|---:|---:|---:|---:|
| Mamba-2 | 47.93 ± 26.41 | 7.07 ± 6.96 | 32.51 ± 28.48 | 74,480 / 74,480 |
| Mamba-2 + SMat d=2 | 21.08 ± 15.29 | 0.15 ± 0.27 | 7.07 ± 12.23 | 84,912 / 84,912 |
| Mamba-2 + SMat d=3 | 9.32 ± 4.09 | 0.00 ± 0.00 | -0.01 ± 0.30 | 116,032 / 109,888 |
| Mamba-2 + SMat d=4 | 30.16 ± 22.60 | 1.82 ± 3.16 | 14.35 ± 24.37 | 133,616 / 120,304 |
| GDN | 46.08 ± 36.51 | 15.64 ± 24.03 | 36.53 ± 36.36 | 39,592 / 39,592 |
| GDN + SMat d=2 | 40.45 ± 37.52 | 14.19 ± 24.56 | 26.10 ± 40.35 | 46,548 / 42,452 |
| GDN + SMat d=3 | 68.70 ± 25.39 | 30.38 ± 25.19 | 57.74 ± 27.70 | 54,328 / 48,696 |
| GDN + SMat d=4 | 83.57 ± 9.08 | 48.53 ± 22.58 | 74.48 ± 10.68 | 58,724 / 51,300 |

The paired d=4 gain is 37.49 ± 27.63 percentage points. Native GDN seed 123 produced
69.87% in an earlier execution and 45.98% on rerun, despite the same recipe, data
hashes and first-step loss/gradient. Model sources did not change. Later training
is not bitwise reproducible in the existing BF16/custom-kernel pipeline; the exact
source of divergence remains unresolved. Both executions are retained and are
not counted as independent seeds. Substituting the stronger earlier execution
raises the baseline three-seed mean to 54.05% ± 39.00%; d=4 still wins each observed seed.
The table uses the current-family execution consistently; this sensitivity is
part of the evidence, not a discarded result.

At two chains, the width-64 variable confirmation gives:

| Model | Terminal-value accuracy (%) | Binding gap (pp) | Paired accuracy gain (pp) |
|---|---:|---:|---:|
| Mamba-2 | 30.43 ± 3.23 | 0.18 ± 1.23 | — |
| Mamba-2 + SMat d=2 | 22.51 ± 18.93 | -0.62 ± 1.97 | -7.92 ± 20.37 |
| Mamba-2 + SMat d=3 | 26.24 ± 16.96 | -0.34 ± 0.13 | -4.19 ± 14.94 |
| Mamba-2 + SMat d=4 | 32.64 ± 5.89 | -0.94 ± 0.57 | 2.21 ± 3.81 |
| GDN | 28.25 ± 17.25 | -1.33 ± 1.34 | — |
| GDN + SMat d=2 | 20.37 ± 16.38 | 0.05 ± 1.07 | -7.88 ± 9.80 |
| GDN + SMat d=3 | 29.83 ± 8.02 | 0.81 ± 0.93 | 1.59 ± 12.54 |
| GDN + SMat d=4 | 20.22 ± 12.90 | -0.39 ± 0.81 | -8.02 ± 19.62 |

The original Mamba d=2 raw-score win reverses on average after adding seeds.
Mamba d=4 and GDN d=3 have small positive mean raw gains, but these are minor
relative to seed variation. No variant shows convincing chain binding. With only
two chains, simply choosing one of the observed terminal values can achieve
roughly 50% without following the query; the measured query-blind majority control
is 54.00% ± 0.23% across the three test seeds. That is why exceeding the unconditional 6.25% value-chance line is
not enough. The binding diagnostic compares the model's same prediction against
the queried and other chain values; its expectation is zero for a query-blind
strategy. At the originally requested loads 8–128, the one-seed sweep is near
chance throughout.


The eight-chain, three-seed control also stays near chance, with no convincing binding.

The requested scientific questions:

1. **Does SMat improve its corresponding backbone?** Current GDN + SMat d=4 does in the
   width 64/four-entity setting. GDN + SMat d=3 also improves the mean (68.70% versus 46.08%),
   with gains ranging from 0.49 to 37.60 points. GDN + SMat d=2 is worse on average.
   Width 32 Mamba-2 + SMat d=2's original single-seed gain shrinks to a paired 3.21 ± 6.81 points;
   it loses one of three seeds. At width 64 all Mamba-2 + SMat variants lose on average on state tracking.
2. **Does the gap increase with memory load?** No. The four-entity transport gain
   disappears by eight entities in the one-seed sweep. At 16 entities, the
   three-seed checks are near chance with almost no binding. The larger-load
   results are dominated by failure to learn within 500 steps and cannot isolate
   asymptotic memory capacity. Load and sequence length also covary.
3. **Does it happen for both backbones?** No convincing shared advantage. The
   strongest state result belongs to current GDN transport; Mamba state results
   are negative or seed-sensitive. See the variable confirmation above.
4. **How do d=2,3,4 differ?** Current GDN d=4 is the strongest and most consistent
   low-load width 64 variant, d=3 improves the mean less reliably, and d=2 varies
   widely. This ordering does not generalize across widths or loads. All variants,
   including losing ones and the older GDN delta-pool recipe, remain reported.
5. **Are tasks saturated?** Not across models. One d=4 run reaches 91.57%, but its
   three-seed mean is 83.57%. Most requested larger-load settings are at the floor,
   not the ceiling. There was no calibration aimed at making a particular SMat
   variant win.
6. **Could parameter count explain the gain?** It cannot be ruled out. GDN + SMat d=4 has
   58,724 total / 51,300 trainable parameters versus 39,592/39,592 for GDN: 48.3% more
   total and 29.6% more trainable. The comparison matches backbone dimensions,
   not total parameters. Existing SMat routing loss and halfway recurrence reset
   are retained, so this is not an isolated ablation of structured memory count.
7. **Are the capabilities different?** State tracking requires overwriting an
   entity's earlier values and answering multiple latest-state queries. Variable
   tracking requires composing disjoint stored relations, x→y→z→VALUE, after
   globally shuffled assignments. These extend direct MQAR, subset routing and
   multi-key retrieval, but the present results do not demonstrate high-load
   state capacity or successful multi-hop chain following. No optional hop sweep
   was run because H=2 binding was not established within the requested budget.

The input generator never consults SMat routing geometry or reset boundaries.
Every entity receives three randomly interleaved updates; four distinct entities
are queried without feeding answers back. Variable chains use fresh disjoint
identifiers and independently sampled terminal values. Random record gaps and
query identities prevent a fixed-position answer rule. A shared 192→256 padding
fix accommodates the existing d=4 constructor for every architecture.

The initial GDN SMat screen used an older frozen MQAR delta-pool recipe. A later
provenance audit identified the repository's selected transport recipe; all
primary comparisons were completed with it. Legacy results are labeled controls,
not silently replaced. The current recipe, calibration changes, baseline repeat,
and launch history are documented in [PROTOCOL.md](PROTOCOL.md).

[All three-seed tables](confirmation.md), [all load plots and learning curves](report.md),
[per-run metrics](results.csv), and [paired comparisons](paired_comparisons.csv).
The complete one-seed tables/plots were saved before extra seeds in
[snapshots/complete-current-one-seed-before-confirmation](snapshots/complete-current-one-seed-before-confirmation/).

Verification: all 354 runs completed; all 248 primary 500-step results passed the matched-recipe/data-hash audit. Saved predictions reproduce the reported metrics. See [audit.json](audit.json), [environment.json](environment.json), and the versioned source snapshots for reproducibility.
