# GDN d=2 learning-rate follow-up

Requested by the user after the common-LR d=2 run completed below native GDN.
Run native GDN and SMat GDN d=2 from scratch at LR0.004, compared with the
completed LR0.003 pair. Keep seed123, width64, two layers, batch256, AdamW
weight decay0.1, cosine decay over32 epochs, and the unchanged source_v5 models.
Reuse the exact cached capacity512 data and batch-order seeds:180,000 training
examples,705 steps per epoch,22,560 updates,5.76 million presentations.

Rationale recorded before launching: the d=2 LR0.003 curve improves gradually,
with late progress on256/512-binding tables and its best macro score at epoch31.
There is no large early peak followed by collapse. A modest increase in initial
LR tests whether learning is too slow under the fixed budget. It does not isolate
all optimizer effects or establish an architectural explanation.

Report final-epoch validation and development-test results for both new runs,
including all five loads and negative outcomes. Keep the primary all-d LR0.003
sweep and its already planned nine-model fresh confirmation unchanged. This
follow-up is a separately labeled optimization comparison, not a replacement
for the original d=2 result. Any later confirmation selected from this follow-up
must use a new independent split, rather than selecting on the original fresh test.
