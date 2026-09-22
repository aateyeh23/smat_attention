# MQAR results

Current transport enables scalar decay, disables incidence rescaling, enables neighboring write-hash gradients, detaches write-hash inputs, and removes planted writes. The transport ablation instead disables scalar decay and enables incidence rescaling. Original Mamba-2 runs have not been rerun with these GDN-specific changes. GDN has1/2/2 heads at widths16/32/64; Mamba-2 has2/4/8 heads with native2x expansion. Head/state dimensions16. Width16 GDN baseline is historical; repeated GDN baselines are reused, not new runs. Single seed123, two layers, LR0.01, final epoch32. Invalid routing ablations01-21 are not included.

The compact table is in mqar_all_results.tex. The expanded table includes64-pair accuracy; CSV includes every validation cell and source paths.
