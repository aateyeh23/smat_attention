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
  model.py         the byte-level LM the synthetic tasks train inside
  kernels/         fused Triton paths (tile, incidence, pool); torch fallbacks
  mixers/          SMat as a Zoology / GDN sequence mixer, and the recurrent
                   baselines it is compared against
src/smat_lm/       the learned-routing models behind the recall and PG-19 results:
                   GDN / Mamba-2 + SMat with hashed writes and four-read selection,
                   their Triton kernels, the PG-19 LMs and the cached decoder.
                   Flat modules, byte-identical to what the campaigns ran
tasks/             one entry point per experiment, all writing results/*.csv
  routing_ceiling.py  what a mask permits, counted without training
  routing_train.py    the same landmark sets, trained
  mqar.py  mkar.py  counting.py  needle.py  streaming.py  copying.py  lm.py
  pg19/            PG-19 LM training, data prep, GPU checks and kernel benchmarks
                   for the src/smat_lm models
analysis/          figures.py, table_*.py, verify_theory.py, bench_*.py
experiments/
  jobs/            job scripts, one per sweep; no site details inside
  zoology/         Zoology sweep configs
  submit.sh        supplies partition/account/walltime from site.conf
  prelude.sh       records library versions and device capability, not identity
results/           every number the paper cites, as CSV or JSON
figures/           each figure with the CSV of its plotted values
tests/             correctness; run tests/test_smat.py first
docs/              construction notes, experiment notes, kernel defects, the
                   learned-routing model definition (practical_model_revision.md)
third_party/       zoology (the frozen copy the runs used) and the upstream
                   Log-Linear kernels, each with its provenance
paper_drafts/      the manuscript
```

## Quick start

```bash
pip install -e .                       # torch + numpy; extras: [kernels] [zoology] [figures]
python tests/test_smat.py --skip-vc    # ~1 min, CPU, fp64
python analysis/verify_theory.py       # VC, counting and density claims
python tasks/routing_ceiling.py --T 4096 --d 1 2 3 4 5 --kmax 6
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
```

## Where the paper's numbers come from

| paper section | task | results |
|---|---|---|
| d-Subset Routing | `tasks/routing_ceiling.py`, `tasks/routing_train.py` | `results/routing_*.csv` |
| Multi-query associative recall | `experiments/zoology/*.py` via Zoology | `results/mqar_*.csv`, `results/gdn_smat_summary/` |
| Multi-key subset recall | `tasks/mkar.py` | `results/mkar_*.csv` |
| MQAR, learned routing, repeated seeds (Table 2) | frozen in `results/paper_seeds_20260920/source/` | `results/paper_seeds_20260920/per_seed.csv` |
| Joint context-key recall (Table 3) | frozen in `results/paper_seeds_20260920/source/joint/` | same `per_seed.csv`; Log-Linear rows in `results/paper_fast_20260920/` |
| MoM memory-budget comparison | frozen in `results/joint_mom_20260921/source/` | `results/joint_mom_20260921/` |
| Selective copying | `tasks/copying.py`, `tasks/copying_gdn.py` | `results/sc_*/` |
| Language modelling, byte level | `tasks/lm.py` | `results/lm/` |
| PG-19, 300M tokens | `tasks/lm.py` | `results/pg19_300m/` |
| PG-19, 500M/750M tokens | `tasks/pg19/train_pg19_scale.py` + `src/smat_lm/` | `results/pg19_six_500m/`, `results/pg19_six_750m/` (launch record only) |
| Prefill and decode cost | `analysis/bench_prefill.py` | `results/results_*.csv` |
| Training throughput of the learned models | `tasks/pg19/bench_pg19_*.py` | `results/pg19_scale_pilot/`, `results/pg19_transport_opt/`, `results/loglinear_backend_timing/` |

Tables and figures are regenerated from those files alone:

```bash
python analysis/figures.py results/results_bf16.csv --results-dir results --outdir figures
python analysis/table_mkar.py                    # multi-key subset recall
```

The recall campaigns (Tables 2-3, MoM) ran from source bundles frozen at launch,
not from the live tree, so each keeps its bundle under `results/<campaign>/source/`
with a `source_sha256.json` of every file. Third-party copies that were
byte-identical to `third_party/` or to pip `fla==0.5.2` were removed from the
bundles; `THIRD_PARTY.md` in each says which. The seed-123 runs those tables
repeat are copied to `results/paper_seeds_20260920/original_seed123/`.
Per-example arrays, checkpoints, job logs, launchers and development campaigns
that no reported number depends on are not in this tree; they remain at the tag
`archive/pre-cleanup-2026-09-22`. The MoM appendix snippet is in
`docs/mom-comparison.tex`, with its aggregation details in `docs/mom-comparison.md`.

## Reading a result

Every task writes one row per evaluation with the run's full configuration, so
a CSV is self-describing and rows from different sweeps concatenate. Two habits
matter when reading them:

- **Check the step count before quoting a number.** Rows are written at every
  evaluation, not only at the end; `analysis/table_mkar.py` prints the step each
  row reached and marks the unfinished ones rather than printing them as final.
- **Do not mix protocols.** Batch size and learning rate decide whether a
  baseline trains at all on some of these tasks -- softmax on multi-key subset
  recall sits at exactly 0.000 through 16k steps at batch 8 and reaches 0.999 at
  batch 32. Rows carry both fields; compare only within one setting.

`results/lm/invalid_frozenhash/` is kept deliberately: those runs were
invalidated by a lazy-initialisation bug that hid the hash parameters from the
optimizer, so the "learned hash" they report was a frozen random projection.
They are retained as the record of a fixed defect, not as results.

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
