"""Boundary transport using GDN transition products, without matrix inverses.

Inputs are normalized features in [batch, time, head, feature] order.
The prefix result includes beta once. Query results exclude the usual 1/sqrt(r)
read scale, which the caller supplies. Both scans reuse FLA's backward kernels.
"""
import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


def boundary_transport(q, k, beta, log_decay, n):
    b, length, h, r = q.shape
    if length != 2*n or q.shape != k.shape:
        raise ValueError('Transport requires equal distant/recent halves and equal Q/K dimensions')
    # Right-multiplying identity by reversed, one-step-shifted prefix
    # transitions gives A_n ... A_{j+1}, excluding the write's own transition.
    reverse_k = k[:, :n].flip(1)
    reverse_beta = beta[:, :n].flip(1)
    reverse_g = log_decay[:, :n].flip(1)
    scan_q = torch.cat((reverse_k*reverse_beta[..., None], q[:, n:]), dim=0).contiguous()
    scan_k = torch.cat((torch.cat((torch.zeros_like(reverse_k[:, :1]), reverse_k[:, :-1]), dim=1),
                        k[:, n:]), dim=0).contiguous()
    scan_beta = torch.cat((torch.cat((torch.zeros_like(reverse_beta[:, :1]), reverse_beta[:, :-1]), dim=1),
                           beta[:, n:]), dim=0).contiguous()
    scan_g = torch.cat((torch.cat((torch.zeros_like(reverse_g[:, :1]), reverse_g[:, :-1]), dim=1),
                        log_decay[:, n:]), dim=0).contiguous()
    identity = torch.eye(r, device=q.device, dtype=torch.float32).expand(2*b, h, r, r).contiguous()
    transformed, _ = chunk_gated_delta_rule(
        q=scan_q, k=scan_k, v=torch.zeros_like(scan_q), beta=scan_beta,
        g=scan_g, initial_state=identity, output_final_state=False,
        scale=1., use_qk_l2norm_in_kernel=False, state_v_first=True, chunk_size=16)
    # The second scan gives A_{n+1} ... A_i q_i = (A_i ... A_{n+1})^T q_i,
    # since each individual GDN transition is symmetric.
    return transformed[:b].flip(1), transformed[b:]
