# Run logs

| file | what it is |
|---|---|
| `sweep_bf16.out` | the bf16 sweep behind `results/results_bf16.csv`, with the full fitted-exponent report |
| `sweep_aux.out` | fp32 and the `r` / `c` sensitivity sweeps |
| `smoke_passing.out` | the GPU smoke run in which all three Triton kernels passed |
| `verify_theory.log` | the VC, counting and density checks behind the `theory_*.csv` tables |
| `routing.out` | the Sec. 4 job: the certified ceiling and the 180-run learned sweep |

The job scripts write new logs here too (`slurm-%j.out`, `smoke-%j.out`).

## The three defects the smoke runs caught

The superseded smoke logs have been removed; what they documented is this, and
each fix is asserted by `test_smat.py`, so the tests rather than the logs are
now what keeps these from coming back.

1. `range()` with runtime bounds does not compile under Triton 3.5.1 + Python 3.14
   (`ast.Num` was removed in 3.12).  The incidence kernel now takes the row
   degree as a `constexpr`.
2. The two-argument `range()` does not compile either, for the same reason.  The
   pooling kernel uses the three-argument form.
3. The fused kernels return fp32, so routing the fp64 reference path through
   them would have made the reference agree with whatever it was meant to
   check.  `_tri_ok` now keeps fp64 on torch, and `test_smat.py` asserts it.
