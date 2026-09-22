# Randomly rewired incidence with exact VC dimension 2

This adds three fresh 32-epoch GDN d=3 joint-recall runs, using training seeds
123, 456 and 789 and the same data, initialization, writer, four-read selector,
transport, reset, optimizer and batch size as the first incidence campaign.
The original campaign and its frozen code are unchanged.

The matched object is the family of **individual summary supports over profile
buckets**. Its VC dimension is exactly 2, matching geometric point-line
incidence. Unions of four summaries are still permitted; this does not constrain
the learned four-read family or the entire network to VC dimension 2.

## Construction

For each layer and sequence length, initialize from geometric incidence with a
random permutation of profile labels. Mix pairs of groups within one randomly
chosen partition, preserving their sizes. Then make 500 random proposals that
swap two profiles between two groups in one partition. Accept a proposal only
if every summary remains distinct and no triple of profiles is shattered.
The RNG seed is `SeedSequence([17, layer, length, 2202])`.

Every final matrix has the same q^2 profiles, q(q+1) **distinct** summaries,
q profiles per summary, q+1 summaries per profile, and partition structure as
the corresponding geometric and original random-incidence models. Parameter
counts and matrix-table memory are unchanged. Matrices are shared across heads
and training seeds, and fixed throughout training.

This is a constrained random walk initialized from geometry, **not uniform
sampling over all VC-2 matrices**. It retains substantial geometric structure;
the global column permutation also changes alignment with the coordinate hash.
Only 1–40 of the 500 later swap proposals were accepted per matrix, after the
initial group mixing. Full counts and distances from the initial geometry are
recorded in `construction.json`. Interpret it as a degree- and VC-matched
rewiring control, not as an experiment changing only an abstract VC number.

## Verification

All ten matrices were checked with exact branch-and-bound VC calculation and
an independent exhaustive test that no triple is shattered. A verified
two-profile shattering witness establishes VC at least 2. The independent
triple test enumerates triples contained in some summary (required for pattern
111), filters only by necessary pair multiplicity, and checks all eight patterns.

Profile-pair intersection counts include values other than 1, proving that the
matrices are not merely row/column permutations of affine-plane incidence.
Unique-summary counts and every degree are checked explicitly. GPU validation
also checks unchanged parameters and RNG state, gradients to learned writers
and readers, exactly one write and four distinct reads, and the full 256-example
batch at sequence length 3076.

## Runs and results

Validation job: 3189287. Training starts only after validation succeeds.
The three training tasks each use one GPU, retain the full 32-epoch budget,
and checkpoint for automatic continuation if needed. No accuracy-based early
stopping or test-based topology selection is used.

- `protocol.json`: recipe and comparison scope.
- `construction.json` and `incidence.npz`: exact matrices and certificates.
- `validated.json`: GPU validation and repeated combinatorial checks.
- `tasks.json` and `jobs.txt`: commands and submitted jobs.
- `report.md`: results paired by training seed with the first campaign.

Refresh results with
`python /u/archerdw/smat_attention/deltaai/incidence_vc2/collect.py`.
