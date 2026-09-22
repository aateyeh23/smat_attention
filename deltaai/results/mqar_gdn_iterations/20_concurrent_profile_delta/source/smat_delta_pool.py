"""Chronologically packed per-profile delta memory using FLA chunked training.

For each routed profile: S <- S + beta*w*k*(v - k^T S)^T.
No additional temporal decay. Duplicate routes for one token are merged before
the update. Zero forward STE weights are retained so their gradients survive.
"""
import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


def pack_routes(keys, values, idx, weights, beta, n_cells):
    b, length, routes = idx.shape
    # Each duplicate contributes its weight to the first occurrence only.
    equal = idx.unsqueeze(-1) == idx.unsqueeze(-2)
    first = ~torch.tril(equal, diagonal=-1).any(-1)
    merged = (equal.to(weights.dtype) * weights.unsqueeze(-2)).sum(-1)
    cells = idx + torch.arange(b, device=idx.device)[:, None, None] * n_cells
    events = torch.arange(b * length, device=idx.device).reshape(b, length, 1).expand_as(idx)
    cells, events, merged = cells[first], events[first], merged[first]
    order = cells.argsort(stable=True)
    cells, events, merged = cells[order], events[order], merged[order]
    active, counts = torch.unique_consecutive(cells, return_counts=True)
    cu = torch.cat([counts.new_zeros(1), counts.cumsum(0)]).to(torch.int32)
    k = keys.reshape(b * length, -1)[events][None, :, None, :].contiguous()
    v = values.reshape(b * length, -1)[events][None, :, None, :].contiguous()
    rate = (beta.reshape(-1)[events] * merged)[None, :, None].contiguous()
    return k, v, rate, active, cu


def pool_delta(keys, values, idx, weights, beta, n_cells, max_chunks=60000):
    # FLA places packed chunks on CUDA grid.y (limit 65535). Bound the
    # chunk count by events/16 plus one rounding chunk per possible profile.
    # Split independent batch/head entries only; never split a profile's history.
    events_per_head = idx.shape[1] * idx.shape[2]
    bound_per_head = (events_per_head + 15) // 16 + min(n_cells, events_per_head)
    shard_heads = max(1, max_chunks // bound_per_head)
    if keys.shape[0] > shard_heads:
        return torch.cat([pool_delta(keys[i:i+shard_heads], values[i:i+shard_heads],
            idx[i:i+shard_heads], weights[i:i+shard_heads], beta[i:i+shard_heads], n_cells,
            max_chunks=max_chunks)
            for i in range(0, keys.shape[0], shard_heads)], dim=0)
    k, v, rate, active, cu = pack_routes(keys, values, idx, weights, beta, n_cells)
    _, state = chunk_gated_delta_rule(
        q=torch.zeros_like(k), k=k, v=v, beta=rate, g=torch.zeros_like(rate),
        scale=1., output_final_state=True, cu_seqlens=cu, chunk_size=16)
    flat = state.new_zeros(keys.shape[0] * n_cells, keys.shape[-1], values.shape[-1])
    return flat.index_copy(0, active, state[:, 0]).reshape(
        keys.shape[0], n_cells, keys.shape[-1], values.shape[-1])
