# Fast GDN seed 123 diagnostic

The full 32-epoch fast checkpoint scored 7.91630859375% validation accuracy. The same checkpoint evaluated with the original dense operator scored 7.91255859375% on the full validation set. Matched-dropout gradient/logit checks on four training examples at each of the five lengths also passed; maximum concatenated parameter-gradient relative error was about 0.218%.

This establishes that both implementations evaluate this checkpoint poorly. It does not establish why its training diverged from the independent dense run, or prove identical gradients at every earlier training step. Keep the seed in the five-seed aggregate and keep dense/fast cohorts separate. Do not replace this result with a favorable rerun.

Diagnostic Slurm job: 3183142 on the GPU cluster, completed successfully. No production checkpoints were modified.
