"""MoM GDN, top-4/no shared memory, with chronological packed computation.

Uses the snapshotted FLA MoM initialization and parameterization. The packed
forward avoids the upstream unstable sort and training convolution crossing
memory boundaries. Memory-specific projections use a batched GEMM. Batch
sharding/checkpointing bounds temporary activation storage, not logical state.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from upstream_mom import MomAttention


class MatchedMom(nn.Module):
    def __init__(self, counts, width=64, layer=0, shard=32):
        super().__init__()
        self.counts = {int(k): int(v) for k, v in counts.items()}
        self.shard = shard
        self.h, self.r, self.p, self.topk = 2, 16, 16, 4
        self.balance_coef = .01
        source = MomAttention(hidden_size=width, head_dim=self.r, num_heads=self.h,
            expand_v=1, num_memories=max(counts.values()), topk=4,
            shared_mem=False, single_kv_proj=False, layer_idx=layer)
        # Preserve independent projections, including per-memory beta/decay gates.
        self.expert_weight = nn.Parameter(torch.stack([
            torch.cat([source.k_proj[i].weight, source.v_proj[i].weight,
                       source.b_proj[i].weight, source.a_proj[i].weight])
            for i in range(source.num_memories)]).detach().clone())
        for name in ('q_proj', 'q_conv1d', 'k_conv1d', 'v_conv1d',
                     'A_log', 'dt_bias', 'g_proj', 'o_norm', 'o_proj'):
            setattr(self, name, getattr(source, name))
        self.routers = nn.ModuleDict()
        for length, count in self.counts.items():
            gate = nn.Linear(width, count, bias=False)
            gate.weight.data.copy_(source.gate.weight[:count])
            self.routers[str(length)] = gate
        self._aux = None

    def _routed(self, x, selected, weights, count):
        b, length, width = x.shape
        # memory-major, then example-major; stable sort retains chronological order.
        ids = torch.arange(b * length, device=x.device).reshape(b, length, 1).expand_as(selected)
        batch = torch.arange(b, device=x.device)[:, None, None].expand_as(selected)
        cells = selected * b + batch
        order = cells.flatten().argsort(stable=True)
        cells = cells.flatten()[order]
        events = ids.flatten()[order]
        memories = selected.flatten()[order]
        _, lengths = torch.unique_consecutive(cells, return_counts=True)
        cu = torch.cat((lengths.new_zeros(1), lengths.cumsum(0))).to(torch.int32)
        # Pad only the per-memory projection GEMM, then immediately unpad.
        counts = torch.bincount(memories, minlength=count)
        starts = counts.cumsum(0) - counts
        max_events = int(counts.max())
        positions = torch.arange(max_events, device=x.device)
        gather = starts[:, None] + positions[None, :]
        valid = positions[None, :] < counts[:, None]
        gather = gather.clamp_max(events.numel() - 1)
        sorted_x = x.reshape(-1, width)[events]
        padded_x = sorted_x[gather]
        projected = torch.bmm(padded_x, self.expert_weight[:count].transpose(1, 2).to(x.dtype))
        projected = projected[valid]
        k, v, beta, raw_g = projected.split([32, 32, 2, 2], dim=-1)
        q = self.q_proj(x).reshape(-1, 32)[events]
        conv = []
        for values, module in ((q, self.q_conv1d), (k, self.k_conv1d), (v, self.v_conv1d)):
            values, _ = module(values[None].contiguous(), cu_seqlens=cu,
                               cache=None, output_final_state=False)
            conv.append(values.reshape(1, -1, self.h, self.r))
        q, k, v = conv
        g = -self.A_log.float().exp() * F.softplus(raw_g.float() + self.dt_bias.float())
        out, _ = chunk_gated_delta_rule(q=q, k=k, v=v, g=g[None].contiguous(),
            beta=beta.sigmoid()[None].contiguous(), cu_seqlens=cu,
            use_qk_l2norm_in_kernel=True, output_final_state=False,
            state_v_first=True, chunk_size=64)
        # The inverse permutation avoids atomic reductions across routed copies.
        restored = torch.empty_like(out[0]).index_copy(0, order, out[0])
        restored = restored.reshape(b, length, self.topk, self.h, self.p)
        return (restored * weights[..., None, None].to(restored.dtype)).sum(2)

    def forward(self, u, **_):
        count = self.counts[u.shape[1]]
        logits = self.routers[str(u.shape[1])](u)
        probabilities = logits.float().softmax(-1)
        weights, selected = probabilities.topk(self.topk, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        # Switch/MoM auxiliary objective without a huge one-hot tensor. Per-layer
        # average (coefficient .01) and division by two layers keep scale explicit.
        frequency = torch.bincount(selected.flatten(), minlength=count).float() / (u.shape[0]*u.shape[1])
        self._aux = count * (frequency * probabilities.mean((0, 1))).sum() * self.balance_coef / 2
        outputs = []
        for start in range(0, u.shape[0], self.shard):
            args = (u[start:start+self.shard], selected[start:start+self.shard],
                    weights[start:start+self.shard], count)
            if self.training and torch.is_grad_enabled():
                outputs.append(checkpoint(self._routed, *args, use_reentrant=False))
            else:
                outputs.append(self._routed(*args))
        o = torch.cat(outputs)
        gate = self.g_proj(u).reshape(*u.shape[:2], self.h, self.p)
        return self.o_proj(self.o_norm(o, gate).flatten(2))

    def get_auxiliary_loss(self):
        value, self._aux = self._aux, None
        return value if value is not None else 0.


def apply_ablation(model, variant, topology=0):
    assert variant in ('mom_profiles', 'mom_bytes')
    rows = []
    parents = [m for m in model.modules() if hasattr(m, 'mixer') and
               m.mixer.__class__.__name__ == 'SmatGDNTransport']
    assert len(parents) == 2
    for layer, parent in enumerate(parents):
        original = parent.mixer
        counts = {}
        for length, modules in original.ca.items():
            ca = next(iter(modules.values()))
            profiles, summaries = ca.N0, ca.D * ca.n_cosets
            count = profiles if variant == 'mom_profiles' else profiles + summaries
            counts[int(length)] = count
            rows.append(dict(layer=layer, length=int(length), memories=count,
                reference_profiles=profiles, reference_summaries=summaries,
                reads=4, writes=4, heads=2, key_dim=16, value_dim=16,
                matrix_state_bytes_per_example=count*2*16*16*4,
                reference_matrix_table_bytes_per_example=(profiles+summaries)*2*16*16*4,
                convolution_cache_elements=count*3*32*4,
                convolution_cache_bytes_if_fp32=count*3*32*4*4))
        parent.mixer = MatchedMom(counts, width=64, layer=layer).to(next(original.parameters()).device)
    return dict(variant=variant, parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        matrix_tables=rows, architecture='MoM Gated DeltaNet, top4, no shared memory, independent K/V/beta/decay projections',
        limitations=['Matrix-state matched, not parameter/compute/total-cache matched.',
            'Four reads and four writes; replaces the whole SMAT mixer, not incidence alone.',
            'Length-specific routers, prefix of shared expert bank across lengths.',
            'FP32 state budget assumes full provisioned bank; conv cache is reported separately.'],
        implementation='Frozen FLA MoM parameter initialization; stable chronological packing and segmented convolution; batched expert projections; batch32 activation checkpointing',
        balance_coefficient=.01, balance_objective='Mean of per-layer Switch loss (sum over all four route positions).')
