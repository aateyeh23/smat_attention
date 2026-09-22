"""Saved-weight arithmetic for first-layer addresses; no model or training run.

Reconstruct embedding, RMSNorm, causal hash convolution, and address projections
on the same 128 examples per cell as the GPU checkpoint probe. Counterfactual
routes report geometric selectivity only; they do not predict model accuracy.
"""
from pathlib import Path
import json
import torch
from torch.nn import functional as F

torch.set_num_threads(2)
OUT = Path(__file__).resolve().parent
DELTA = OUT.parents[2]
CACHE = DELTA / 'results/mqar_gdn_four_reads/modal_bundle/zoology_cache'
FILES = ['data_25f7a1e8f8e6d2ba1fdbfa481d66614a.pt',
         'data_5e0c59f2a5df62cc996f523c97b0d10a.pt',
         'data_9892a64887c1ed56c1c7421f492d76ed.pt',
         'data_b53959d96a217faa44c26a74b7d53685.pt',
         'data_2e72f812c43c426a106c777f791c5e7d.pt']


def routing_statistics(M, reads, weights, value_cells, expected, mask):
    incidence = M[reads[..., None], value_cells[:, :, None, None, :]]
    coefficients = (incidence * weights[..., None]).sum(-2)
    target = coefficients.gather(-1, expected[..., None]).squeeze(-1)
    target_share = target / coefficients.sum(-1).clamp_min(1e-12)
    target_covered = incidence.any(-2).gather(-1, expected[..., None]).squeeze(-1)
    valid = mask[:, None].expand_as(target)
    return dict(target_weight_share=target_share[valid].mean().item(),
                target_coverage_per_head=target_covered[valid].float().mean().item(),
                covered_values=incidence.any(-2).sum(-1)[valid].float().mean().item())


rows = []
with torch.no_grad():
    ck = torch.load(OUT / 'w32-d3.pt', map_location='cpu', weights_only=False)
    s = ck['model']
    pre = 'backbone.layers.0.mixer.'
    for filename in FILES:
        data = torch.load(CACHE / filename, map_location='cpu', weights_only=False)
        x, y = data['inputs'][:128], data['labels'][:128]
        batch, length = x.shape
        n, pairs = length // 2, data['slices']['num_kv_pairs']
        cp = pre + f'ca.{length}.1.'
        matrix = s[cp + 'M']
        directions, q, cells = matrix.shape
        M = matrix.flatten(0, 1)
        u = F.embedding(x, s['backbone.embeddings.word_embeddings.weight'])
        u = u * torch.rsqrt(u.square().mean(-1, keepdim=True) + 1e-5)
        u = u * s['backbone.layers.0.norm.weight']
        writer_u = F.conv1d(F.pad(u.transpose(1, 2), (3, 0)),
                           s[pre + 'kconv.weight'], groups=u.shape[-1]).transpose(1, 2)
        W = F.normalize(s[cp + 'W'], dim=-1, eps=1e-6)

        def cells_for(source):
            normalized = F.layer_norm(source, (source.shape[-1],), eps=1e-5)
            z = torch.einsum('blm,hkm->bhlk', normalized, W)
            z = z * s[cp + 'gamma'][None, :, None] + s[cp + 'b'][None, :, None]
            bins = (torch.special.ndtr(z) * q).clamp(0., q - 1e-4).long()
            return bins[..., 0] * q + bins[..., 1]

        writer_cells = cells_for(writer_u[:, :n])
        # The inherited finite geometry plants two anchor positions; reproduce
        # these overrides rather than assuming every write uses the learned hash.
        plant_cols = s[cp + 'plant_cols']
        plant_cells = s[cp + 'plant_cells']
        for col, cell in zip(plant_cols, plant_cells):
            if col < n:
                writer_cells[:, :, col] = cell
        query_cells = cells_for(u[:, n:])
        value_cells = writer_cells[:, :, 1:2*pairs:2]
        mask = y[:, n:] != -100
        expected = (x[:, n:, None] == x[:, None, :2*pairs:2]).long().argmax(-1)
        expected = expected[:, None].expand(-1, W.shape[0], -1)
        equal = query_cells[..., None] == value_cells[:, :, None]
        target_equal = equal.gather(-1, expected[..., None]).squeeze(-1)
        valid = mask[:, None].expand_as(target_equal)
        wrong_equal = (equal.sum(-1) - target_equal.long()).float() / (pairs - 1)
        logits = torch.einsum('blm,hem->bhle',
                             F.layer_norm(u[:, n:], (u.shape[-1],), eps=1e-5),
                             s[cp + 'W_read']) + s[cp + 'b_read'][None, :, None]
        scores, reads = logits.topk(4, dim=-1)
        original = routing_statistics(M, reads, scores.softmax(-1), value_cells, expected, mask)
        allowed = M.T[query_cells].bool()
        conditioned_scores, conditioned_reads = logits.masked_fill(~allowed, -torch.inf).topk(4, dim=-1)
        conditioned = routing_statistics(M, conditioned_reads, conditioned_scores.softmax(-1),
                                         value_cells, expected, mask)
        equal_weight = routing_statistics(M, conditioned_reads,
                                          torch.full_like(conditioned_scores, .25),
                                          value_cells, expected, mask)
        target_positions = 2 * expected + 1
        target_is_anchor = (target_positions[..., None] == plant_cols).any(-1)
        unplanted = valid & ~target_is_anchor
        anchor_stats = routing_statistics(M, reads, scores.softmax(-1), value_cells,
                                          expected, mask & target_is_anchor[:, 0])
        nonanchor_stats = routing_statistics(M, reads, scores.softmax(-1), value_cells,
                                             expected, mask & ~target_is_anchor[:, 0])
        row = dict(pairs=pairs, length=length, checkpoint_epoch=ck['next_epoch'],
                   examples=128, layer=0, available_buckets=cells,
                   anchor_positions=plant_cols.tolist(),
                   fraction_queries_targeting_anchor=target_is_anchor[valid].float().mean().item(),
                   exact_query_target_bucket_match=target_equal[valid].float().mean().item(),
                   exact_query_wrong_value_bucket_match=wrong_equal[valid].mean().item(),
                   nonanchor_query_target_bucket_match=target_equal[unplanted].float().mean().item(),
                   nonanchor_query_wrong_value_bucket_match=wrong_equal[unplanted].mean().item(),
                   original=original, four_planes_through_query_address=conditioned,
                   original_anchor_targets=anchor_stats,
                   original_nonanchor_targets=nonanchor_stats,
                   four_planes_through_query_address_equal_weight=equal_weight,
                   uniform_target_weight_share=1/pairs)
        rows.append(row)
        print(json.dumps(row), flush=True)
(OUT / 'address-alignment.json').write_text(json.dumps(dict(method=__doc__, rows=rows), indent=2) + '\n')
