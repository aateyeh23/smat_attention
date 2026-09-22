# SMat-Attention

Causal masks indexed by their VC dimension, and the schedule that makes them
cheap. A mask `M^(d)` is built from incidences between points and hyperplanes
over a finite field; attention under it is exact, costs `O(T^{2-3/d} + T)` work
up to a chunking term, and decodes from a cache of `O(T^{1-1/d})` states. `d=1`
is ordinary causal masking.

This repository holds the construction, the tasks the paper reports, the job
scripts that produced the numbers, and the analysis that turns them into tables
and figures. Current cluster settings live in `experiments/site.conf`, which
is not committed. The legacy `deltaai/` and `smat/` experiment trees retain
their original source paths, launch recipes, and result provenance.

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
tasks/             one entry point per experiment, all writing results/*.csv
  routing_ceiling.py  what a mask permits, counted without training
  routing_train.py    the same landmark sets, trained
  mqar.py  mkar.py  counting.py  needle.py  streaming.py  copying.py  lm.py
analysis/          figures.py, table_*.py, verify_theory.py, bench_*.py
experiments/
  jobs/            job scripts, one per sweep; no site details inside
  zoology/         Zoology sweep configs
  submit.sh        supplies partition/account/walltime from site.conf
  prelude.sh       records library versions and device capability, not identity
results/           every number the paper cites, as CSV or JSON
figures/           each figure with the CSV of its plotted values
tests/             correctness; run tests/test_smat.py first
docs/              construction notes, experiment notes, kernel defects
deltaai/           legacy experiment runners, frozen sources, and recorded results
smat/              legacy standalone implementation and supporting experiment files
paper_drafts/      the manuscript
```

## Quick start

```bash
pip install -e .                       # torch + numpy; extras: [kernels] [zoology] [figures]
python tests/test_smat.py --skip-vc    # ~1 min, CPU, fp64
python analysis/verify_theory.py       # VC, counting and density claims
python tasks/routing_ceiling.py --T 4096 --d 1 2 3 4 5 --kmax 6
```

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
| Selective copying | `tasks/copying.py`, `tasks/copying_gdn.py` | `results/sc_*/` |
| Language modelling | `tasks/lm.py` | `results/lm/` |
| Prefill and decode cost | `analysis/bench_prefill.py` | `results/results_*.csv` |

Tables and figures are regenerated from those files alone:

```bash
python analysis/figures.py results/results_bf16.csv --results-dir results --outdir figures
python analysis/table_mkar.py                    # multi-key subset recall
```

The legacy campaigns also retain results under `deltaai/results/`, including
the joint-recall, MoM, incidence, repeated-seed, and PG-19 experiments. These
folders preserve their original layouts because runners and source manifests
refer to those paths. The current packaged implementation remains in
`src/smat/`; the legacy trees are separate experiment implementations.
The MoM appendix snippet is in `docs/mom-comparison.tex`, with its aggregation
details in `docs/mom-comparison.md`.

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

Current `experiments/jobs/` scripts carry no partition, account, node list or GPU model; `submit.sh`
supplies them from an uncommitted `site.conf`. `prelude.sh` records
`torch`/`triton` versions and the device's compute capability and memory, never
a hostname or a product name. Legacy `deltaai/` and `smat/` experiments and
their frozen provenance retain site-specific information and absolute paths.

Git history is **not** anonymous: commits carry author names and institutional
email addresses. Scrubbing that needs a history rewrite, which changes every
commit id; for a blind submission, export a fresh snapshot instead of rewriting
this repository.
