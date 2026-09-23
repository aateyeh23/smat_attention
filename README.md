# SMat-Attention

Causal masks indexed by their VC dimension, and the schedule that makes them
cheap. A mask `M^(d)` is built from incidences between points and hyperplanes
over a finite field; attention under it is exact, costs `O(T^{2-3/d} + T)` work
up to a chunking term, and decodes from a cache of `O(T^{1-1/d})` states. `d=1`
is ordinary causal masking.

This repository holds the construction, the tasks the paper reports, the job
scripts that produced the numbers, and the analysis that turns them into tables
and figures. Nothing in it names a machine, a site or an account: cluster
settings live in `experiments/site.conf`, which is not committed.

## Layout

```
src/smat/          the library
  mask.py          M^(d), the incidence C, the audit, the shattering witness
  attention.py     the chunkwise schedule: pooled long-range branch + causal scan
  assign.py        how tokens map to profiles and types -- positional, content
                   hashed, or a closed-form rule
  vc.py            exact VC / pseudo-dimension by branch and bound
  model.py         the model the subset-routing and multi-key tasks train
  kernels/         fused Triton forward/backward paths; torch fallbacks
  mixers/          SMat as a Zoology mixer (PG-19 300M) and the recurrent baselines
src/smat_lm/       the learned-routing models behind the PG-19 500M/750M runs:
                   GDN / Mamba-2 + SMat with hashed writes and four-read selection,
                   their Triton kernels, the PG-19 LMs and the cached decoder.
                   Flat modules, byte-identical to what the campaigns ran
tasks/             one entry point per experiment
  routing_train.py    subset routing (Table routing, Figure routing-pattern)
  routing_ceiling.py  the landmark sets and pattern counts routing_train uses
  routing_bench.py    end-to-end cost of the routing models (Table e2e-routing)
  mkar.py             multi-key subset recall (Table mkar and its appendix)
  window_schedules.py stepped-window decoding (the horizon-free appendix)
  lm.py               PG-19, 300M tokens (Table pg19-300m)
  pg19/               PG-19 500M/750M training, data prep, GPU checks, benchmarks
  data/               PG-19 tokenisation (GPT-2 BPE) and raw-byte preparation
analysis/          the scripts that turn results/ into the paper's tables and figures
experiments/
  jobs/            the job scripts behind every kept result; no site details inside
  submit.sh        supplies partition/account/walltime from site.conf
  prelude.sh       records library versions and device capability, not identity
results/           every number the paper cites, as CSV or JSON
figures/           each data figure in the paper with the CSV of its plotted values
tests/             correctness; run tests/test_smat.py first
docs/              construction notes, the learned-routing model definition
                   (practical_model_revision.md), the MoM appendix snippet
third_party/       zoology (the frozen copy the runs used) and the upstream
                   Log-Linear kernels, each with its provenance
```

## Quick start

```bash
pip install -e .                       # torch + numpy; extras: [kernels] [zoology] [figures]
python tests/test_smat.py --skip-vc    # ~1 min, CPU, fp64
python analysis/verify_theory.py       # VC, counting and density claims
```

The learned-routing models import each other by flat module name, as they did
when they ran, so put their directories on the path rather than installing them:

```bash
pip install -e '.[kernels,baselines]'  # the runs pinned flash-linear-attention==0.5.2,
                                       # causal-conv1d==1.6.2.post1, mamba-ssm==2.3.2.post1;
                                       # zoology comes from third_party/, not pip
export PYTHONPATH=$PWD/src/smat_lm:$PWD/tasks/pg19:$PWD/third_party:$PWD/third_party/log_linear
python tasks/pg19/test_pg19_rank64_gpu.py --help
```

The GDN + SMat PG-19 arms trained with `pg19_optimized_kernels.enable()` called
first (tiled FP32 read/write kernels, H100-class GPUs only); the other arms did not.

To run a sweep on a cluster:

```bash
cp experiments/site.conf.example experiments/site.conf   # fill in partition etc.
experiments/submit.sh experiments/jobs/run_mkar_fair2.sbatch base
experiments/submit.sh experiments/jobs/run_prefill_512k.sbatch results_bf16_512k_h200
```

## Where the paper's numbers come from

Every table and data figure of `paper_drafts/iclr_submission.tex`, the script
that prints it, and the results it reads.

| paper | produced by | results |
|---|---|---|
| Fig. `prefill` (prefill cost) | `analysis/fig_prefill.py` -> `figures/prefill_cost.*` | `results/results_bf16_512k_h200.csv` (`experiments/jobs/run_prefill_512k.sbatch`) |
| Fig. `routing-pattern`, Table `routing` | `analysis/fig_routing.py` -> `figures/fig17_routing.*`; `analysis/table_meanstd.py` | `results/routing_learned*.csv` (`run_routing*.sbatch`) |
| Table `mkar`, `mkar-seeds`, `mkar-softmax-sweep`, `mkar-posctrl`, Fig. `mkar-sanity` | `analysis/table_meanstd.py`, `analysis/mkar_sanity.py` -> `figures/fig18_mkar_sanity.*` | `results/mkar_fair2_*.csv`, `mkar_posctrl.csv`, `mkar_softsweep_*.csv` (`run_mkar_*.sbatch`, `run_baseline_gaps.sbatch`) |
| Table `mqar-results` (MQAR, learned routing) | frozen in `results/paper_seeds_20260920/source/` | `results/paper_seeds_20260920/per_seed.csv`, Log-Linear in `original_loglinear/` |
| Table `joint_recall` (JCKR) | frozen in `results/paper_seeds_20260920/source/joint/` | same `per_seed.csv`; Log-Linear rows in `results/paper_fast_20260920/` |
| Table `mom_macro` (MoM comparison) | frozen in `results/joint_mom_20260921/source/` | `results/joint_mom_20260921/`; snippet in `docs/mom-comparison.tex` |
| Table `pg19-300m` | `tasks/lm.py`, `analysis/eval_pos_loss.py` (`run_pg19.sbatch`) | `results/pg19_300m/` |
| Table `pg19-750m` | `tasks/pg19/train_pg19_scale.py` + `src/smat_lm/` | configuration in `results/pg19_six_500m/campaign.json`; 500M record in `results/pg19_six_500m/`, 750M launch record in `results/pg19_six_750m/`; kernel validation in `results/pg19_transport_opt/` |
| Table `e2e-routing` | `tasks/routing_bench.py` (`tasks/routing_bench.sbatch`) | `results/routing_bench/rows*.jsonl` |
| Tables `mqar-uniform`, `mqar-placed`, `pg19` (stepped window) | `analysis/table_window_schedules.py` | `results/window_schedules/` (`run_window_schedules.sbatch` with `winsched_*.txt`) |
| Theorem 1 (VC, density) and the Section 3 counts | `analysis/verify_theory.py` | `results/theory_*.csv` |

Also kept, as measurements of the paper's models that no table prints yet:
`results/pg19_e2e_efficiency/` (end-to-end cost of the PG-19 500M models,
`tasks/pg19/bench_e2e.py`), `results/results_bf16_512k_a100pcie.csv` (the
prefill sweep repeated on an A100), and `results/loglinear_backend_timing/`
(why the Log-Linear baselines use the dense reference for GDN).

```bash
python analysis/table_meanstd.py            # Tables routing and mkar
python analysis/mkar_sanity.py              # MKAR appendix tables and fig18
python analysis/fig_routing.py              # routing figure
python analysis/fig_prefill.py              # prefill figure
python analysis/table_window_schedules.py   # stepped-window appendix tables
```

The recall campaigns (Tables MQAR, JCKR, MoM) ran from source bundles frozen at
launch, not from the live tree, so each keeps its bundle under
`results/<campaign>/source/` with a `source_sha256.json` of every file.
Third-party copies that were byte-identical to `third_party/` or to pip
`fla==0.5.2` were removed from the bundles; `THIRD_PARTY.md` in each says which.
The seed-123 runs those tables repeat are copied to
`results/paper_seeds_20260920/original_seed123/`.

Development campaigns, ablations and tasks the paper does not report (counting,
needle, streaming, selective copying, in-tree MQAR, the byte-level LM sweeps,
chunk and rank sweeps, the earlier figures) are not in this tree. The state
before this trim is tagged `archive/pre-audit-2026-09-22`, and per-example
arrays, checkpoints and launchers from before that are at
`archive/pre-cleanup-2026-09-22`.

## Reading a result

Every task writes one row per evaluation with the run's full configuration, so
a CSV is self-describing and rows from different sweeps concatenate. Two habits
matter when reading them:

- **Check the step count before quoting a number.** Rows are written at every
  evaluation, not only at the end; `analysis/table_meanstd.py` counts a
  multi-key run only once it has reached its full step budget.
- **Do not mix protocols.** Batch size and learning rate decide whether a
  baseline trains at all on some of these tasks -- softmax on multi-key subset
  recall sits at exactly 0.000 through 16k steps at batch 8 and reaches 0.999 at
  batch 32. Rows carry both fields; compare only within one setting.

## Anonymity

Job scripts carry no partition, account, node list or GPU model; `submit.sh`
supplies them from an uncommitted `site.conf`. `prelude.sh` records
`torch`/`triton` versions and the device's compute capability and memory, never
a hostname or a product name. Recorded results and frozen bundles had names,
home paths, accounts, cluster names and service URLs replaced; every frozen file
this changed is listed with its original SHA-256 in that bundle's `SCRUBBED.json`,
so it still verifies against `source_sha256.json` at the archive tag.

Git history is **not** anonymous: commits carry author names and institutional
email addresses. Scrubbing that needs a history rewrite, which changes every
commit id; for a blind submission, export a fresh snapshot instead of rewriting
this repository.
