MQAR: GDN + SMAT with learned causal keys and sparse soft routing, d=3.

Same seed (123), two layers, head dimension 16, value expansion 1, one head at width 16 and two at widths 32/64. Training budget: 32 epochs, 707 optimizer steps per epoch. No specified key-token offset.

| Width | GDN (%) | Soft SMAT (%) | Gain (points) | Epochs | Complete |
|---|---:|---:|---:|---:|---|
| 16 | 35.71 | 78.05 | +42.35 | 32/32 | True |
| 32 | 68.93 | 94.29 | +25.36 | 32/32 | True |
| 64 | 60.52 | 96.85 | +36.33 | 32/32 | True |

These are final-checkpoint accuracies for completed runs, and current-checkpoint accuracies for unfinished runs; they are not best-epoch selections.

Selective copying: seed 0, width 64, two layers, four GDN heads of dimension 16, value expansion 2; length 4096, 16 content tokens, batch 32, 10,000 steps. All requested checkpoints passed the recipe audit.

| Model | Accuracy (%) | Steps |
|---|---:|---:|
| GDN | 96.02 | 10000 |
| GDN + SMAT d=2 | 88.11 | 10000 |
| GDN + SMAT d=3 | 84.11 | 10000 |
| GDN + SMAT d=4 | 97.75 | 10000 |

The selective-copying hybrids use the original hard-routing delta-memory variant, not the MQAR soft-routing variant.
