"""Change only incidence, or explicitly labeled independent-bucket controls."""
import hashlib
import math
import numpy as np
import torch
from torch import nn
from content_addr import ContentAssign


def balanced_incidence(shape, topology, layer, length):
    directions, cosets, profiles = shape
    assert profiles % cosets == 0
    rng = np.random.default_rng(np.random.SeedSequence([topology, layer, length]))
    matrix = np.zeros(shape, dtype=np.float32)
    for direction in range(directions):
        groups = rng.permutation(profiles).reshape(cosets, profiles // cosets)
        matrix[direction, np.arange(cosets)[:, None], groups] = 1
    return matrix


class RectangularBuckets(ContentAssign):
    """Same coordinate CDF writer/neighbor STE, with unequal coordinate bin counts.

    d=3 geometry has q^2 profiles and q(q+1) summaries. Radices (q,2q+1)
    give q^2+q(q+1) independent states with the same two write projections.
    Only this larger-capacity control changes the writer's bin counts.
    """
    def _cells(self, u, plant_at=False, tok_w=None, retain_neighbors=False):
        assert self.mode == 'point' and not self.plant and self.anneal == 1.0
        assert not getattr(self, 'periodic_hash', False)
        nb, length, _ = u.shape
        normalized = self.W / self.W.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        z = (torch.einsum('blm,hkm->bhlk', self.ln(u), normalized)
             * self.gamma[None, :, None, :] + self.b[None, :, None, :])
        bins = z.new_tensor(self.radices)
        s = torch.minimum((torch.special.ndtr(z)*bins).clamp_min(0), bins-1e-4)
        s = s.reshape(nb*self.h, length, self.dim)
        low = s.floor().long()
        fraction = s-low
        high = torch.minimum(low+1, low.new_tensor(self.radices)-1)
        idx = torch.zeros(nb*self.h, length, 1, device=u.device, dtype=torch.long)
        weights = torch.ones_like(idx, dtype=s.dtype)
        for coordinate in range(self.dim):
            place = math.prod(self.radices[coordinate+1:])
            frac = fraction[..., coordinate:coordinate+1]
            soft_low, soft_high = 1-frac, frac
            wlow = 1+(soft_low-soft_low.detach())
            whigh = soft_high-soft_high.detach()
            idx = torch.cat((idx+low[..., coordinate:coordinate+1]*place,
                             idx+high[..., coordinate:coordinate+1]*place), -1)
            weights = torch.cat((weights*wlow, weights*whigh), -1)
        if not retain_neighbors:
            hard_idx = torch.zeros_like(idx[..., :1])
            hard_weight = torch.ones_like(weights[..., :1])
            for coordinate in range(self.dim):
                hard_idx = hard_idx+low[..., coordinate:coordinate+1]*math.prod(self.radices[coordinate+1:])
                soft_low = 1-fraction[..., coordinate:coordinate+1]
                hard_weight = hard_weight*(1+(soft_low-soft_low.detach()))
            idx, weights = hard_idx, hard_weight
        self._sparse = idx, weights
        self._soft = None
        penalties = []
        for coordinate, count in enumerate(self.radices):
            low_w, high_w = 1-fraction[..., coordinate], fraction[..., coordinate]
            if tok_w is not None:
                tw = tok_w.reshape(nb*self.h, length).to(s.dtype)
                low_w, high_w = low_w*tw, high_w*tw
            occupancy = s.new_zeros(nb*self.h, count)
            occupancy = occupancy.scatter_add(-1, low[..., coordinate], low_w)
            occupancy = occupancy.scatter_add(-1, high[..., coordinate], high_w)
            occupancy = occupancy/occupancy.sum(-1, keepdim=True).clamp_min(1e-6)
            penalties.append((occupancy*(occupancy.clamp_min(1e-9)*count).log()).sum(-1).mean())
        self.aux = None if self.freeze else torch.stack(penalties).mean()
        return None


def replace_reader(ca, types, layer, length):
    """Preserve common logits; initialize additional ones without consuming training RNG."""
    old = ca.W_read
    generator = torch.Generator(device=old.device).manual_seed(87001+layer*10000+length)
    weights = torch.randn(ca.h, types, old.shape[-1], device=old.device,
                          dtype=old.dtype, generator=generator)*old.shape[-1]**-.5
    common = min(types, old.shape[1])
    weights[:, :common] = old.detach()[:, :common]
    ca.W_read = nn.Parameter(weights)
    ca.b_read = nn.Parameter(old.new_zeros(ca.h, types))


def apply_ablation(model, variant, topology=0):
    assert variant in ('geometry', 'random', 'buckets_profiles', 'buckets_bytes')
    rows = []
    mixers = [m for m in model.modules() if m.__class__.__name__ == 'SmatGDNTransport']
    assert len(mixers) == 2
    for layer, mixer in enumerate(mixers):
        assert mixer.memory_read_k == 4 and mixer.transport_scalar_decay
        assert not mixer.memory_incidence_rescale and not mixer.memory_plant
        for length_string, modules in mixer.ca.items():
            length = int(length_string)
            for ca in modules.values():
                assert ca.dim == 2 and ca.mode == 'plane' and ca.read_k == 4
                original = ca.M.detach().cpu().numpy()
                directions, cosets, profiles = original.shape
                types = directions*cosets
                reference_slots = profiles+types
                if variant == 'random':
                    matrix = balanced_incidence(original.shape, topology, layer, length)
                    assert np.array_equal(matrix.sum(-1), original.sum(-1))
                    assert np.array_equal(matrix.sum((0, 1)), original.sum((0, 1)))
                    assert np.array_equal(matrix.sum(1), np.ones((directions, profiles)))
                    ca.M.copy_(torch.from_numpy(matrix).to(ca.M.device))
                    ca.coset_of.copy_(ca.M.argmax(1))
                elif variant.startswith('buckets_'):
                    ca.mode = 'point'
                    if variant == 'buckets_bytes':
                        ca.__class__ = RectangularBuckets
                        ca.radices = (ca.q, 2*ca.q+1)
                        ca.N0 = math.prod(ca.radices)
                        assert ca.N0 == reference_slots
                    replace_reader(ca, ca.N0, layer, length)
                    del ca.M
                    del ca.coset_of
                active_types = ca.N0 if ca.mode == 'point' else ca.D*ca.n_cosets
                summary_slots = 0 if ca.mode == 'point' else active_types
                payload_bytes = 4*mixer.h*mixer.r*mixer.p
                row = dict(layer=layer, length=length, q=ca.q, profile_count=ca.N0,
                    read_type_count=active_types, materialized_summary_count=summary_slots,
                    heads=mixer.h, state_rank=mixer.r, value_dimension=mixer.p,
                    write_count=1, read_count=4,
                    profile_table_bytes_per_example=ca.N0*payload_bytes,
                    summary_table_bytes_per_example=summary_slots*payload_bytes,
                    combined_matrix_table_bytes_per_example=(ca.N0+summary_slots)*payload_bytes,
                    reference_combined_matrix_table_bytes_per_example=reference_slots*payload_bytes,
                    read_table_bytes_per_example=active_types*payload_bytes,
                    hash_radices=list(getattr(ca, 'radices', (ca.q, ca.q))),
                    incidence_edges=0 if ca.mode == 'point' else int(ca.M.sum().item()))
                if ca.mode == 'plane':
                    matrix = ca.M.detach().cpu().numpy()
                    flat = matrix.reshape(types, profiles)
                    gram = flat.T@flat
                    overlaps = gram[np.triu_indices(profiles, 1)]
                    counts, frequencies = np.unique(overlaps, return_counts=True)
                    row.update(incidence_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),
                        pairwise_profile_overlap_histogram={str(int(k)): int(v) for k,v in zip(counts,frequencies)})
                rows.append(row)
    return dict(variant=variant, topology_seed=topology,
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        matrix_tables=rows,
        byte_budget_definition='FP32 profile plus materialized summary tables, per example. '
            'This is not an implemented autoregressive KV cache or total training memory. '
            'Read-table-only bytes, peak CUDA bytes, and static buffers are reported separately.',
        registered_buffer_bytes=sum(b.numel()*b.element_size() for b in model.buffers()))
