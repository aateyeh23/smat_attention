"""Neighbor-aware hash gradients attached to a single-route additive pool.

The memory input is already pooled using detached, hard write weights. Its
ordinary autograd path supplies key/value gradients. This identity supplies only
the missing write-weight gradients, including zero-valued neighboring routes.
"""
import torch
from torch.autograd.function import once_differentiable


class _WriteHashGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, memory, keys, values, indices, weights):
        ctx.save_for_backward(keys, values, indices)
        ctx.weight_dtype = weights.dtype
        return memory

    @staticmethod
    @once_differentiable
    def backward(ctx, memory_grad):
        keys, values, indices = ctx.saved_tensors
        batch, length, routes = indices.shape
        cells, key_dim, value_dim = memory_grad.shape[1:]
        gradient = torch.empty(indices.shape, device=keys.device, dtype=ctx.weight_dtype)
        flat_grad = memory_grad.reshape(batch * cells, key_dim, value_dim)
        offsets = torch.arange(batch, device=keys.device)[:, None] * cells
        # Bound the gathered state-gradient temporary; do not materialize
        # [batch, length, routes, key_dim, value_dim].
        for start in range(0, length, 64):
            end = min(start + 64, length)
            for route in range(routes):
                address = indices[:, start:end, route] + offsets
                selected = flat_grad.index_select(0, address.reshape(-1)).reshape(
                    batch, end - start, key_dim, value_dim)
                gradient[:, start:end, route] = torch.einsum(
                    'bnrp,bnr,bnp->bn', selected,
                    keys[:, start:end].to(selected.dtype),
                    values[:, start:end].to(selected.dtype))
        return memory_grad, None, None, None, gradient


def with_write_hash_gradient(memory, keys, values, indices, weights):
    """Keep hard memory unchanged; restore the full hard-STE write gradient.

Keys/values must receive their gradients through ``memory`` only. The caller
must detach the single forward route's weight to avoid counting it twice.
This implements first-order training gradients for additive pooling only.
"""
    return _WriteHashGradient.apply(
        memory, keys.detach(), values.detach(), indices, weights)
