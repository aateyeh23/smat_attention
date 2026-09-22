SOURCE AUDIT WARNING: this run loaded the historical frozen router; intended routing changes are unverified. See ../SOURCE-RESOLUTION-AUDIT.md. Raw measurements below are preserved.

MQAR GDN + SMAT d=3, width32: 20_concurrent_profile_delta

Strict gates at epochs 4,6,8,...,32; accuracy must exceed GDN.

| Epoch | Candidate | GDN | Margin (pp) | Decision |
|---|---:|---:|---:|---|
| 1 | 0.07% | 0.82% | -0.75 | observe |
| 2 | 38.73% | 42.05% | -3.32 | observe |
| 3 | 44.30% | 57.27% | -12.97 | observe |
| 4 | 49.48% | 60.66% | -11.18 | reject |

Terminal result: reject at epoch 4.
