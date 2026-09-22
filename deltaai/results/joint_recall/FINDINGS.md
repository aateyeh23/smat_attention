# Joint recall findings

Completed 4/8 planned runs.

The campaign is still running. Available results and curves are interim; do not treat incomplete seed averages as the final experiment.

Each completed run used180,000 fixed training examples for32 epochs (22,624 optimizer updates;5.76M example presentations). Final-epoch test is the primary endpoint.

| Condition | Backbone | Paired seeds | Native macro test (%) | SMat macro test (%) | SMat − native (pp) |
|---|---|---:|---:|---:|---:|
| shared | Mamba-2 | 1 | 72.89 | 69.01 | -3.88 |

Mean ± sample SD across paired seeds; three seeds do not establish statistical significance. A positive difference favors SMat at this fixed data-exposure budget. Models are not parameter-, compute-, or memory-matched.

Inspect the per-table results and analytical lookup controls in [report.md](report.md). Scoring above random guessing alone does not establish context binding: a context-blind lookup can recover many targets. The paired unique-key condition measures how much performance changes when context is needed.

This is an adaptation of published block-context joint recall: masked inquiry answers, smaller table-size grid and the MQAR training budget. The equal information/inquiry boundary aligns with the SMat reset. Splits are independent draws from the same distribution; this is not a held-out-combination test. See [PROTOCOL.md](PROTOCOL.md) for the complete design and failed unpadded pilot.

