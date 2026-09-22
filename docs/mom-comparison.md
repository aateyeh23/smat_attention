# MoM appendix comparison

`mom-comparison.tex` is an insertion snippet for the manuscript, which lives
outside this repository. Replace `du2026mom` with its bibliography key for
Du et al., *MoM: Linear Sequence Modeling with Mixture-of-Memories*
(ICLR 2026; https://arxiv.org/abs/2502.13685).

The snippet uses final-epoch **test** macro accuracy, averaged equally across
five binding loads, and sample standard deviations (`ddof=1`). The existing
`results/paper_seeds_20260920/table3.tex` reports validation accuracy;
its values must not be substituted into this test-accuracy table.

| Model | Included training seeds | Mean (%) | Sample SD (%) |
|---|---|---:|---:|
| Plain GDN, middle three of five scores | 1, 3, 4 | 53.80 | 2.86 |
| GDN + SMat, d=3 | 123, 456, 789 | 58.92 | 4.22 |
| MoM-derived, profile-count matched | 123, 456, 789 | 43.73 | 1.48 |
| MoM-derived, matrix-storage matched | 123, 456, 789 | 35.10 | 3.82 |

Selected GDN scores: 50.708125%, 54.33076171875%, 56.3587890625%.
Omitted scores: 49.37993164062501% (seed 123), 62.543291015625% (seed 2).
Across **all five** GDN runs, the mean and sample SD are **54.66 (5.21)%**.
Selecting the middle three after observing test scores reduces the reported
variation; the snippet discloses that selection. The other rows include all
runs in their three-seed campaigns.

Sources:

- GDN: `results/paper_seeds_20260920/per_seed.csv`, selecting
  `task=joint`, `family=gdn_current`, `d=1`, `split=test`, `cell=macro`.
- GDN + SMat: `results/joint_incidence_20260921/geometry/`,
  each seed's `result.json`, `final_test.accuracy`.
- MoM: `results/joint_mom_20260921/mom_profiles/` and `mom_bytes/`,
  each seed's `result.json`, `final_test.accuracy`.

Our MoM variants use top-4 routing with no shared memory. The original MoM
paper's main configuration has four routed memories, top-2 activation, and
an additional shared memory. GDN + SMat retains its local GDN recurrent state.
The full protocol is in `results/joint_mom_20260921/protocol.json`.
