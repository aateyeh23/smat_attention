# Learned Mamba-2 + SMat on multi-key subset recall

The learned-routing Mamba-2 + SMat model at d = 4 (one hashed write per
distant token, four learned reads per recent query) on the multi-key subset
recall task of `tasks/mkar.py`, with that task's data, loss, evaluator and
12,000-step budget unchanged.  Each seed trains a separate model for each of
k = 1, 2, 3.  Code: `tasks/mkar_mamba_smat/`; job: `experiments/jobs/run_mkar_mamba_smat.sbatch`.

Final-step exact-support accuracy (%), mean (sample SD):

| seeds | k=1 | k=2 | k=3 |
|---|---|---|---|
| 123 (selection pilot) | 100.00 | 100.00 | 99.80 |
| 456, 789, 2026, 2027 (confirmation) | 100.00 (0.00) | 99.94 (0.12) | 99.06 (1.10) |
| all five | 100.00 (0.00) | 99.95 (0.11) | 99.21 (1.01) |

## Layout

| path | contents |
|---|---|
| `pilot_s123/` | seed 123: the run that selected this configuration |
| `selection_rule.json` | the rule fixed before the pilot's k = 3 fit finished: all three final scores >= 99%, then four new seeds |
| `confirm/` | seeds 456, 789, 2026, 2027, trained only after the rule was met |
| `five_seed_summary.json` | per-seed scores and result hashes for all fifteen fits |

Each campaign folder holds `protocol.json` (recipe), `tasks.json`,
`validated.json` (pre-launch GPU checks), `source_sha256.json` (the frozen
source it ran from), `provenance.json`, `results.csv` / `summary.*`, and per
run `recipe.json`, `metrics.jsonl` (every 2,000 steps) and `result.json`
(final step; no checkpoint selection).  Checkpoints and logs are not kept.
`SCRUBBED.json` lists the files whose paths were anonymised.

## The model

The local Mamba-2 recurrence is unchanged.  The distant memory uses a query/key
projection shared between memory reads and writes, causal writer features,
delta transport of the profile memories to the boundary without scalar decay,
and divides its read by the mean incidence weight before the existing gate.
The features that choose a token's write address are detached, so the routing
balance loss does not train the shared key convolution; retrieval still does.
No task labels or token identities enter the model.  These are the switches
`memory_tied_features`, `memory_boundary_transport`, `memory_incidence_rescale`
and `detach_write_hash_features` of `src/smat_lm/zoo_smat_mixer.py`.

The configuration was chosen on seed 123 from a small set of variants of this
hybrid memory (the pilot's `provenance.json` records the two it combines:
incidence normalization alone reached 100 / 100 / 18.8, the stop-gradient
alone 99.8 / 99.9 / 0.0).  Those earlier variants are not in this repository.
The selection seed and the confirmation seeds are reported separately above.

Comparisons to native backbones or softmax match width and training exposure,
not state memory or FLOPs; these runs alone do not establish a routing-specific
advantage.
