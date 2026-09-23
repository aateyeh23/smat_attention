"""SMat-Attention: masks of tunable VC dimension, and the schedule that makes
them cheap.

    mask.py       the M^(d) family, the point-hyperplane incidence C, the audit
    attention.py  the chunkwise schedule of the paper's Theorem 2
    assign.py     how tokens are assigned to profiles and types (positional,
                  content hashed, or a closed-form rule)
    vc.py         exact VC / pseudo-dimension of a 0/1 matrix, by branch and bound
    model.py      the byte-level LM the synthetic tasks are trained in
    kernels/      fused Triton forward and backward paths for the chunkwise schedule
    mixers/       SMat as a Zoology sequence mixer (the PG-19 300M models) and the
                  recurrent baselines it is compared against
"""
