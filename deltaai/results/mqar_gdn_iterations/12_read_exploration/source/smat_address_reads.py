"""Score hyperplanes using the writer's continuous content address.

Only scalar address probabilities are dense. The caller still gathers exactly
four matrix summaries. No payload is read while computing these scores.
"""
import torch


def symmetric_write_routes(coordinate, q, sigma=.75):
    """One hard bucket forward; a centered, symmetric local surrogate backward.

    Each axis considers the current bin and its two neighbors. The first route
    is always the hard cell, allowing the caller to pool exactly once. Invalid
    edge neighbors have zero probability. Occupancy regularization is unchanged.
    """
    batch, length, dim = coordinate.shape
    hard = coordinate.floor().long()
    idx = torch.zeros(batch, length, 1, device=coordinate.device, dtype=torch.long)
    weights = torch.ones(batch, length, 1, device=coordinate.device, dtype=coordinate.dtype)
    soft = weights
    for axis in range(dim):
        center = hard[..., axis]
        bins = torch.stack((center, center-1, center+1), -1)
        valid = (bins >= 0) & (bins < q)
        bins = bins.clamp(0, q-1)
        logits = -.5 * ((coordinate[..., axis, None] - (bins.to(coordinate.dtype)+.5))/sigma).square()
        probability = logits.masked_fill(~valid, -torch.inf).softmax(-1)
        one = torch.zeros_like(probability)
        one[..., 0] = 1.
        straight_through = one + (probability - probability.detach())
        place = q ** (dim-1-axis)
        idx = (idx[..., :, None] + bins[..., None, :] * place).flatten(-2)
        weights = (weights[..., :, None] * straight_through[..., None, :]).flatten(-2)
        soft = (soft[..., :, None] * probability[..., None, :]).flatten(-2)
    return idx, weights, soft


def address_read_logits(ca, u):
    """Log probability mass on each hyperplane near the query's hash address."""
    W = ca.W / ca.W.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    z = torch.einsum('blm,hkm->bhlk', ca.ln(u), W)
    z = z * ca.gamma[None, :, None] + ca.b[None, :, None]
    coordinate = torch.special.ndtr(z) * ca.q
    centers = torch.arange(ca.q, device=u.device, dtype=u.dtype) + .5
    distance = (coordinate[..., None] - centers) / ca.address_read_sigma
    bin_probability = (-.5 * distance.square()).softmax(-1)
    # Lexicographic cell order, identical to the hard writer's base-q address.
    cell_probability = bin_probability[..., 0, :]
    for axis in range(1, ca.dim):
        cell_probability = (cell_probability[..., :, None]
                            * bin_probability[..., axis, None, :]).flatten(-2)
    if ca.mode == 'plane':
        probability = torch.einsum('bhln,en->bhle', cell_probability,
                                   ca.M.flatten(0, 1).to(u.dtype))
    else:
        probability = cell_probability
    return probability.clamp_min(1e-12).log()
