# SMAT: the revised Section 3

Construction, exact verification and GPU timing for the revised `M^(d)` --- the
one whose top-left block is the causal triangle `L_n` rather than the identity
`I_n`, with the global set `P` removed and the long-range block factored as
`G = R C S^T`.

```
smat_mask.py      construction of M^(d), the incidence C, the audit, the
                  shattering witness, dense materialisation
smat_attn.py      the Section 3.3 schedule: one segmented scan + S^T, C, R
smat_triton.py    the three fused CUDA kernels
test_smat.py      correctness -- run this first
verify_theory.py  the checkable claims of Section 3.1 (VC, counting, density)
vc_dimension.py   exact VC / pseudo-dimension of a matrix by branch and bound,
                  standalone: `python vc_dimension.py --selftest`
bench_routing.py  Section 4's d-Subset Routing: VC as an operational capacity
train_routing.py  the learned run on that same task, with the mask fixed
bench_smat.py     the timing sweep and the fitted-exponent report
plot_smat.py      every figure (PDF + PNG) with a companion CSV per panel
_prelude.sh       sourced by the job scripts: cd, host, GPU, toolchain versions
run_smat.sbatch      the GPU job: correctness + the bf16 sweep + figures
run_smat_aux.sbatch  fp32 and the r / c sensitivity sweeps; submit alongside
run_routing.sbatch   the certified ceiling and the learned run, same marked sets
smoke_gpu.sbatch     a 35-minute version, to check the kernels first
results/          every CSV the jobs write: the timing sweeps, the theory
                  tables, and the routing benchmarks
figs/             the figures, each with the CSV of its plotted values
logs/             the sweep logs, and the three kernel defects the smoke runs
                  caught (see logs/README.md)
```

Every script writes into `results/` by default and every figure is drawn by
`plot_smat.py`, so there is one place to look for a number and one place to
change how it is drawn.

## Quick start

```bash
python test_smat.py --skip-vc              # ~1 min, cpu, fp64
python verify_theory.py                    # ~20 min, cpu; VC + counting + density
python bench_routing.py --plot             # ~10 min, cpu; the expressivity benchmark
python bench_smat.py --quick --device cpu --threads 1
sbatch smoke_gpu.sbatch                    # check the triton kernels
sbatch run_smat.sbatch                     # the real sweep + figures
sbatch run_smat_aux.sbatch                 # fp32 and the constant sweeps
sbatch run_routing.sbatch                  # the ceiling and the learned run
```

Redrawing every figure from what is already in `results/`, without re-running
anything:

```bash
python plot_smat.py results/results_bf16.csv --results-dir results --outdir figs
python plot_smat.py results/results_bf16.csv --results-dir results \
    --outdir figs_dark --dark
```

Each panel whose CSV is missing is skipped with a line saying so, so this is
also the way to redraw a subset.

## What is implemented

`M^(d)` in the block form of Eq. (3.9), for the whole admissible range:

```
M^(d) = [ N       0      ]     n = c*floor(T/2c),  T_R = T - n
        [ G     L_{T_R}  ]     N = L_n  (geometric)   N = Z_k  (endpoint)
```

- **Geometric family, `1 <= d <= d_max(T)`.** `q = nextprime(n^{1/d})`,
  profiles `prof(j) = (j - |P|) mod q^{d-1}` round-robin over `F_q^{d-1}`, and
  `B = q(q^{d-1}-1)/(q-1)` affine hyperplanes with the `2^{d-1}` **planted**
  ones of Eq. (3.5) placed at type indices `0 .. 2^{d-1}-1`.  `d_max` is
  reported by `d_max_geometric(T)`; the binding conditions are `q^{d-1} <= n`
  and `T_R >= 2B`.
- **`d = 1` is a genuine degenerate member**, not a special case: `B = 1`,
  `H_0 = F_q^0`, `G` all ones, `M^(1) = L_T`, and it runs the same code path.
- **Endpoint `d = floor(log2 T)`.** Boolean zeta landmarks `Z_k = L_2^{(x)k}`
  and complemented Walsh rows.  Both are applied by butterflies rather than
  enumerated --- see the note below.
- **Remark 3.4's global channel `P`** is available behind `--n-P` and defaults
  to off.
- **Three long-range backends**: `pooled` (the schedule of the draft),
  `direct` (the unpooled scatter, for the `T^{1/d}` comparison), and
  `subtractive` (Remark 3.10, computed so it can be measured).

Numerator and normalizer share one code path via the appended ones-column on
`V`, and the branches are summed before the single division.

## The GPU kernels

Three Triton kernels.  The first two exist because the obvious torch expression
allocates an intermediate asymptotically larger than the work itself; the third
because the shape cuBLAS is handed is too small to be worth launching.

| kernel | replaces | why |
|---|---|---|
| `chunk_local` | the within-chunk score tile of Eq. (3.19) | the torch form builds `(nb, nc, c, c)` = 1.1 GB at `T=2^18, nb=8`, though Sec. 3.3 says the tile never leaves on-chip memory |
| `incidence` | `F[:, plane_pts].sum(2)` for `U = C F` | allocates `nnz(C) * r * p` = 35 GB at `d=3, T=2^18` to perform that many adds |
| `pool` | the batched GEMM for `F = S^T Z` | decomposes into `N0` GEMMs of shape `(r x g)(g x p)` with `g ~ 20`, so cuBLAS spends its time on launch |

All three hold their accumulator in registers.  Measured against the torch path:
1.3-5.2x faster at 2-3x less peak memory, and peak memory is *flat in the chunk
width* `c`, which is the direct check that the score tile stays in registers.

`pool` is used only when the shape suits it (`pool_ok`): at small `N0` --- `d=1`
above all, where `N0 = 1` --- the torch route is one large GEMM and wins.

Two constraints shaped these, both worth knowing before editing them:

- **Triton 3.5.1 on Python 3.14 cannot compile `range()` with runtime bounds,
  nor the two-argument form** (its frontend reaches for `ast.Num`, removed in
  3.12).  Every loop here uses the three-argument form with `constexpr` bounds,
  which is why the row degree and the group count are passed as `constexpr`.
- **The kernels accumulate and return fp32**, so the fp64 reference path must
  not go through them --- silently downgrading it would make the reference agree
  with whatever it was meant to check.  `_tri_ok` enforces this and
  `test_smat.py` asserts it.

If a kernel fails at run time, `smat_attn` retires *that* kernel for the session
with a warning and uses torch for that step; the others keep running, so an
unattended sweep degrades rather than dies.  An explicit `backend="triton"`
still raises, so the tests can tell a fallback from a pass.

## Reading the output

`bench_smat.py` fits `log(time) = alpha log T + beta` and reports `alpha`
against the prediction.  Two cautions are printed with the table:

- Only the **incidence** phase carries `2 - 3/d`.  Pooling (`S^T`) and the
  type-major GEMM (`R`) are `Theta(T)` at every `d`, and total prefill is the
  blend, so for `d <= 3` total prefill should come out `~1`.
- A phase pinned at the kernel-launch floor fits `alpha ~ 0` however small its
  count is.  The achieved-rate column says when that is happening; the exact
  incidence counts in `results/theory_counting.csv` are free of the problem
  entirely.

`T*` is reported as the first *measured* `T` at which SMAT beats causal SDPA,
not as a fitted crossing.

The sweep writes its CSV after every configuration, not only at the end, so a
job that reaches its wall clock one point short still hands back what it
measured.  Submit `run_smat.sbatch` and `run_smat_aux.sbatch` together: they
write disjoint outputs and asking one job to do both risks the clock.

Figures `fig10_rank` and `fig11_chunk` come from the `results_r*.csv` and
`results_c*.csv` that the aux job writes, and are produced automatically when
those files are in the `--results-dir`.

## Figures

| figure | shows |
|---|---|
| `fig1_scaling` | prefill vs `T` against causal SDPA, fitted `alpha` per `d` |
| `fig2_incidence` | the incidence phase alone: pooled `2-3/d` vs direct `2-2/d`, beside the exact counts |
| `fig3_phases` | where prefill time goes, one panel per `d` |
| `fig4_decode` | per-token cost (flat in `T`) and cache size vs a KV cache |
| `fig5_counting` | Table 1: `I_prof`, `I_tok`, and the dense `T^2` reference |
| `fig6_vc` | Theorem 3.1(iii): measured VC against `d`, at all three evidence levels |
| `fig7_density` | Theorem 3.1(iv): `L_n` against the previous `I_n` block |
| `fig8_triton` | the fused kernels against torch, time and peak memory |
| `fig9_subtractive` | Remark 3.10: retained mass against `1/q`, and what it costs |
| `fig10_rank` | sensitivity to the feature rank `r`: moves `T*`, not the slope |
| `fig11_chunk` | sensitivity to the chunk width `c` |
| `fig12_routing` | Section 4: `Pi_M(k)`, the answerable fraction, and task error vs `k` |
| `fig13_routing_learned` | the learned run against that certified ceiling, and the influence of the unrequested payloads |

`fig12` is drawn from the largest `routing_T*.csv` that was run *with* the task,
since a `--no-task` table has no error panel to draw; `fig13` is paired with the
ceiling at the `T` it was trained at, so the two come from the same marked sets.

`--dark` re-steps the palette for a dark surface.  Every series carries a direct
label and its own marker, so identity never rests on colour alone, and each
panel writes a companion `.csv` of the exact plotted values.

`--threads 1` matters on a shared login node: with an oversubscribed OpenMP pool
a small matmul can take 5 ms instead of 25 us, which swamps every phase this
harness measures.  `bench_smat.py` warns when it detects this.

## Verifying the theory rather than assuming it

`verify_theory.py` writes four CSVs and the corresponding figures:

| claim | how it is checked |
|---|---|
| Thm 3.1(i),(ii),(iv) | dense materialisation at small `T`: causal, unit diagonal, `M <= L_T`, density vs the `I_n` variant |
| Thm 3.1(iii), lower | the construction's own shattered set, replayed against the definition in `O(2^d d)` --- certified at **every** `T` |
| Thm 3.1(iii), upper | exact branch-and-bound (`vc_dimension.py`) on the materialised mask; deciding VC is LOGNP-complete so this only reaches small `T` |
| the geometric ingredient | exact VC of the `B x N0` incidence `C`, which the proof says is `d-1` --- much cheaper than the full mask |
| Table 1 | `I_prof` and `I_tok` counted exactly from the built structure, with the closed forms |
| Remark 3.10 | the retained fraction against `1/q`, and the error the cancellation actually costs |

The three levels of VC evidence are kept distinct and are not interchangeable:
the **witness** is a certified lower bound available at every `T`; **`VC(C)`**
is the geometric ingredient, cheap because `C` is far smaller than `M`; and
**`vc_exact`** is the real claim but affordable only for small `T`.  A row
flagged `[lb]` means the search hit its time limit with a lower bound that
matches the prediction --- that is not the same as `[!!]`, which means the
measured value disagrees.

## Measuring expressivity rather than asserting it

`bench_routing.py` runs the `d`-Subset Routing task of Section 4.  The lemma of
"Notions of Expressiveness" makes it measurable *without training*: mark `k`
positions, set `v_{j_l} = x_l e_l` and zero every other value, and for any
kernel strictly positive on allowed edges the output support at row `t` is
exactly `S_t ∩ J`.  So the routing patterns the layer can produce are exactly
`{S_t ∩ J}`, and a request `A` is answerable iff some row realises it.  What
the benchmark reports is therefore the *ceiling* for any model carrying the
mask; a trained model can approach it and cannot exceed it.

Three levels, as with the VC evidence, and they are not interchangeable:

| level | what it is |
|---|---|
| `n_patterns` | `Pi_M(k)`, counted exactly over all `T` rows, against `2^k` and Sauer--Shelah |
| `answerable_frac` | the fraction of the `2^k` requests some row realises |
| `nmse`, `exact_support_frac` | the task run end to end through `smat_attention` itself |

The third level is the one that can fail for reasons the first two cannot see
-- a kernel bug or a normalisation error shows up as task error even when the
combinatorics are right.  `mask_columns` and `row_support_size` read entries off
the definition in `O(Tk)` rather than materialising `M`, so the benchmark runs
at the `T` of the timing sweep; both are asserted against `materialise` at small
`T`.

Measured at `T=4096`, fp64 (`results/routing_T4096.csv`), and identically at
`T=32768` (`results/routing_T32768.csv`, combinatorics only):

```
        k=1  k=2  k=3  k=4  k=5  k=6     R(M)   VC
d=1       2    3    4    5    6    7        1    1
d=2       2    4    7   10   13   16        2    2
d=3       2    4    8   15   24   35        3    3
d=4       2    4    8   16   31   54        4    4
d=5       2    4    8   16   32   63        5    5
2^k       2    4    8   16   32   64
```

`R(M) = d` exactly, at every `d` tested.  Three things worth drawing out:

- **`d=1` realises exactly `k+1` patterns**, which is the lemma's claim for
  ordinary causal/linear attention -- the routes are nested prefixes, so no
  two marked channels can be selected independently.
- **`Pi_M(d+1)` hits the Sauer--Shelah ceiling exactly** at every `d` (3, 7,
  15, 31, 63).  The masks are not merely VC-`d`; on the marked sets found they
  are *extremal* for that VC dimension.
- **The counts are identical at `T=4096` and `T=32768`.**  Routing capacity is
  a function of `d` alone, not of context length.

Through the kernel, normalised squared error sits at fp64 round-off
(`1e-17`--`1e-13`) for every `k <= d` and jumps by ten or more orders of
magnitude at `k = d+1`; `exact_support_frac` tracks `answerable_frac` to the
digit, which is the lemma holding through the real implementation.

Two limits to state plainly.  This measures what the mask *permits*, not what a
trained model *finds* -- it is a necessary condition for the expressivity claim
and not a sufficient one, and a learned run on this task is the natural next
step.  And for `k > d` the marked set is chosen by greedy search over a
subsample of columns, so those `Pi_M(k)` are lower bounds on the best set; the
`k <= d` rows, which carry the claim, use the construction's own witness and
are exact.

## The learned run

`train_routing.py` trains an actual model on the same task, against the same
marked sets, with the mask *fixed*: `W_Q, W_K, W_V, W_O` and a positional table
are learned.  Four arms, because the interesting comparison is not SMAT against
nothing -- `smat`, `smat_content` (a control: SMAT routes on position, so a
content signal must not lift its ceiling), `softmax` (causal attention, request
conveyed positionally) and `softmax_content` (full causal attention that can
address the request by content -- the baseline the comparison has to survive).

`run_routing.sbatch`, GPU, `T=1024`, 4 arms x 6 `k` x 3 seeds, 180 runs, 69 min.

**SGD finds what the mask permits, and nothing more.**  Measured exact-routing
match against the certified ceiling, over all 24 `(d,k)` cells:

```
correlation(learned, ceiling) = 0.9938
mean gap -0.030      mean |gap| 0.032      max |gap| 0.116
```

The learned curve sits just under the ceiling everywhere -- never above it,
which would be impossible, and never far below it, which would mean the
optimiser rather than the mask was the binding constraint.

**The transition at `k = d` survives training.**  Normalised error is `~5e-7`
for every `k <= d` and jumps four to five orders of magnitude at `k = d+1`:

```
        k=1      k=2      k=3      k=4      k=5      k=6
d=1  1.5e-04  1.2e-01  1.4e-01  1.8e-01  2.0e-01  2.2e-01
d=2  5.3e-05  5.7e-07  4.8e-02  9.6e-02  1.4e-01  1.5e-01
d=3  3.0e-05  8.4e-07  4.8e-07  1.6e-02  4.9e-02  8.2e-02
d=4  9.1e-06  8.8e-07  3.7e-07  9.0e-07  3.8e-03  2.7e-02
```

**It is a ceiling, not a failed fit.**  Every one of the 30 `k <= d` runs
finished with training loss below `1e-4` (median `~6e-7`), so the `k > d`
numbers are the mask binding, not the optimiser giving up.  This is the obvious
reviewer objection and the `loss_q*` columns answer it without a dashboard.

**Influence is exactly zero, and only for SMAT.**  Resample the payloads the
request did not ask for and measure how far the prediction moves:

```
smat, all k <= d      0 of 60 runs with nonzero influence   (exact zero)
softmax_content       never zero: min 2.3e-02, median 4.4e-01, max 1.5e+00
```

**The control behaves.**  `smat_content` matches `smat` exactly at `k <= d`
(diff `+0.0000` at every `d`) and moves it by at most `+0.05` beyond -- the
content signal cannot lift a ceiling that is positional, which is what says the
mask is doing the work.

**What the baselines actually show, stated plainly.**  `softmax` with the
request conveyed positionally fails as soon as `k >= 2` (nmse `3.4e-01`, match
`0.52 -> 0.05`): causal attention with no content signal cannot tell requested
landmarks from unrequested ones.  But `softmax_content` *largely solves the
task at every* `k` (nmse `1e-03` to `2e-02`) -- it simply never solves it
exactly, matching the routing pattern only `0.98 -> 0.03` of the time, because
it attends to every allowed position with positive weight.  And for `k > d`
its error is *lower* than SMAT's.

So the claim these figures support is not that full attention cannot do
`d`-Subset Routing.  It is that SMAT does it **exactly, with unrequested items
provably non-influential, at `Theta(T)` cost**, for every `k` up to `VC(M) = d`
-- and that `d` is the knob setting how far that goes.  Beyond `d`, full
attention is the better tool.  Leading with the expressivity claim alone would
not survive a referee who runs the content-addressed baseline.

`d=5` is absent: `d_max(1024) = 4`.  Rerun at `T >= 4096` to extend the sweep.

## A correctness fix the routing benchmark forced

`bench_routing.py` asserts that a masked-out column is *exactly* non-influential
-- Sec. 4's claim that unrequested items are non-influential "since the mask
removes their edges".  Measuring it found that the implementation did not have
that property, in two places, both the same mistake: an exclusive scan written
as a subtraction.

```
Sin = torch.cumsum(D, dim=1) - D          # (a + x) - x is not a
Sin[:, nc0:] -= Sin[:, nc0:nc0 + 1]       # and the reset repeats it
```

`cumsum[b] - D[b]` is not bitwise `sum_{a<b} D[a]`, so perturbing chunk `b`
moved `S^in_b`, and thence rows of chunk `b` that *precede* the perturbed
column.  The reset did the same at segment scale, routing the whole landmark
block's mass through every recent chunk's prefix, where it cancelled only to
`O(eps)`.  Both are now a shifted scan taken per segment: same cost, and a
landmark payload can no longer perturb a recent row the mask excludes it from.

This is the cancellation Remark 3.10 rejects the *subtractive* long-range route
for, and it was sitting in the adopted causal one.  Sec. 3.3's "the segmented
scan is free and numerically benign" is true of the segmented form and not of
the subtractive one; the draft should say which it means.

The existing tests did not catch it: they compare against a dense reference to
`1e-10`, and an `O(eps)` leak passes that comfortably.  `test_exact_noninfluence`
now perturbs a column and requires every row the mask excludes it from to be
**bit-identical**.  It fails on the pre-fix code at every `d` tested (4-5 of 6
columns move excluded rows) and passes after.  An exactness claim needs an
exactness test, not a tolerance.

The long-range path was exactly clean throughout -- pooling never leaked.

## Known deviations from the draft

1. **The endpoint's shattered recent column.** The draft takes `t = B+1` and so
   requires `m >= 2B`, which fails at an exact power of two (`T_R = B`).  The
   *first* recent column works instead and needs only `T_R >= B`: every recent
   row contains it and no zeta row does, so the `2^k` zeta rows supply every
   pattern with a `0` there and the `B` recent rows supply every pattern with a
   `1`.  The exact VC search confirms `VC = floor(log2 T)` at `T = 64, 128, 256`
   --- all exact powers of two --- so the theorem holds; only the witness in the
   proof needs changing.
2. **The endpoint is not `Theta(T^2)` in practice.** Table 1 reports
   `I_prof = Theta(T^2)` for the endpoint, which is right about `nnz(C)`.  But
   `C_{ha} = (1 + (-1)^{<h,a>})/2`, so `U = C F = (J F + H F)/2` is a Walsh
   butterfly: `O(T log T)` states touched, not `T^2`.  The zeta block is
   `L_2^{(x)k}` and is likewise `O(T log T)`.  Section 3.3 already allows "a
   structured finite-geometric transform" as a backend; the endpoint row of the
   table should say so.
3. **The fitted exponents in `T` approach their limits from below.**
   `q = nextprime(n^{1/d})` moves in coarse steps, so `I_prof` and `I_tok` are
   staircases in `T`.  The closed forms
   `I_prof = q^{d-1}(q^{d-1}-1)/(q-1)` and `I_tok = n_X (q^{d-1}-1)/(q-1)` hold
   exactly at every `T` and are what the tests assert; the `T`-fits are
   reported as evidence, not as the check.

## Not implemented

Backward / products with `M^T` beyond the adjoint sketch of Theorem 3.2, the
recursive/hierarchical variant of the "Against recursion" remark, mask *banks*,
exception edges, and the block-periodic variant.
