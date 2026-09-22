"""Score hyperplanes using the writer's continuous content address.

Only scalar address probabilities are dense. The caller still gathers exactly
four matrix summaries. No payload is read while computing these scores.
"""
import torch


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
