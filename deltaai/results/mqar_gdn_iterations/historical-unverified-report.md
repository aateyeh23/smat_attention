WARNING: Trials01–21 are not verified architectural comparisons; see SOURCE-RESOLUTION-AUDIT.md.

Strict GDN comparison, width32/d3, seed123. Required gates: 4,6,...,32.

| Trial | Epoch4 | Latest epoch | Candidate | GDN | Decision |
|---|---:|---:|---:|---:|---|
| 01_no_scalar_decay | 42.25% | 4 | 42.25% | 60.66% | reject |
| 02_isolated_hash | 56.66% | 4 | 56.66% | 60.66% | reject |
| 03_content_only | 56.64% | 4 | 56.64% | 60.66% | reject |
| 04_shared_address | 43.79% | 4 | 43.79% | 60.66% | reject |
| 05_shared_hash | 43.38% | 4 | 43.38% | 60.66% | reject |
| 06_tied_content | 41.86% | 4 | 41.86% | 60.66% | reject |
| 07_incidence_scale | 61.03% | 6 | 60.34% | 64.48% | reject |
| 08_orthogonal_hash | pending | 3 | 57.41% | 57.27% | observe |
| 08b_orthogonal_hash | 52.52% | 4 | 52.52% | 60.66% | reject |
| 09_gdn_key_hash | 54.85% | 4 | 54.85% | 60.66% | reject |
| 10_address_reader | 37.00% | 4 | 37.00% | 60.66% | reject |
| 11_symmetric_write_grad | 53.71% | 4 | 53.71% | 60.66% | reject |
| 12_read_exploration | 53.91% | 4 | 53.91% | 60.66% | reject |
| 13_joint_hash_balance | 52.36% | 4 | 52.36% | 60.66% | reject |
| 14_tied_additive | 35.45% | 4 | 35.45% | 60.66% | reject |
| 15_aligned_additive | 39.47% | 4 | 39.47% | 60.66% | reject |
| 16_periodic_hash | 54.82% | 4 | 54.82% | 60.66% | reject |
| 17_profile_delta | 57.29% | 4 | 57.29% | 60.66% | reject |
| 18_aligned_profile_delta | 47.77% | 4 | 47.77% | 60.66% | reject |
| 19_clipped_transport | 56.51% | 4 | 56.51% | 60.66% | reject |

No candidate has yet passed all required gates.
Trial08 was technically aborted and is excluded from accuracy comparisons.
Trial19 changes the optimizer recipe by clipping global gradient norm at1; other completed trials retain the unclipped recipe.
