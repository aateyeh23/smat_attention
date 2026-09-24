"""Shared task interface and backbone-specific learned memory mechanisms."""
import hashlib
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import PROTOCOL


def curveball(original, seed):
    shape = original.shape
    flat = np.asarray(original, dtype=np.uint8).reshape(-1, shape[-1])
    rows = [set(np.flatnonzero(row)) for row in flat]
    rng = np.random.default_rng(seed)
    trades = 100*len(rows)
    for _ in range(trades):
        a,b = map(int, rng.choice(len(rows), 2, replace=False))
        shared = rows[a] & rows[b]
        pool = np.asarray(sorted(rows[a] ^ rows[b]), dtype=np.int64)
        count = len(rows[a])-len(shared)
        rng.shuffle(pool)
        rows[a] = shared | set(pool[:count])
        rows[b] = shared | set(pool[count:])
    result = np.zeros_like(flat)
    for i, row in enumerate(rows):
        result[i, list(row)] = 1
    assert np.array_equal(result.sum(0), flat.sum(0))
    assert np.array_equal(result.sum(1), flat.sum(1))
    assert np.any(result != flat)
    result = result.reshape(shape)
    assert np.any(result.sum(1) != 1), 'Original partition constraint survived'
    return result, dict(trades=trades, changed_entries=int((result != original).sum()),
        old_partition_membership_min=int(result.sum(1).min()),
        old_partition_membership_max=int(result.sum(1).max()),
        sha256=hashlib.sha256(result.tobytes()).hexdigest())


class SoftmaxMixer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.heads = PROTOCOL['softmax_heads']
        self.qkv = nn.Linear(width, 3*width)
        self.out = nn.Linear(width, width)

    def forward(self, x):
        b,t,w = x.shape
        q,k,v = self.qkv(x).reshape(b,t,3,self.heads,w//self.heads).unbind(2)
        y = F.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2),
            v.transpose(1,2), dropout_p=0., is_causal=True)
        return self.out(y.transpose(1,2).reshape(b,t,w))


def make_mixer(config, layer):
    width, length = PROTOCOL['width'], PROTOCOL['length']
    d, family = config['d'], config['family']
    if family == 'softmax':
        return SoftmaxMixer(width)
    if family == 'mamba2':
        from zoo_mamba_four_reads import MqarMambaFourReads
        mixer = MqarMambaFourReads(width, layer_idx=layer, d=d,
            d_state=PROTOCOL['state_dim'], headdim=PROTOCOL['head_dim'], memory_tied_features=True,
            prebuild_lengths=(length,))
        mixer.detach_write_hash_input = config.get('detach_write_hash_input', False)
        mixer.memory_boundary_transport = config.get('memory_boundary_transport', False)
        mixer.memory_incidence_rescale = config.get('memory_incidence_rescale', False)
        mixer.detach_write_hash_features = config.get('detach_write_hash_features', False)
    elif family == 'gdn':
        from zoo_smat_gdn import SmatGDNReset
        if d == 1:
            mixer = SmatGDNReset(width, layer_idx=layer, d=1, headdim=PROTOCOL['head_dim'],
                n_heads=PROTOCOL['gdn_heads'], expand_v=1, reset=False, prebuild_lengths=(length,))
        else:
            from zoo_gdn_transport import SmatGDNTransport
            mixer = SmatGDNTransport(width, layer_idx=layer, d=d, headdim=PROTOCOL['head_dim'],
                n_heads=PROTOCOL['gdn_heads'], expand_v=1, reset=True, lam_bias=-2.197224577,
                anneal_steps=0, balance_coef=.01, hash_codim=1,
                memory_update='delta', memory_key_mode='tied_causal',
                memory_read_k=4, prebuild_lengths=(length,),
                write_hash_neighbor_grad=True, transport_scalar_decay=True,
                detach_write_hash_input=True, memory_plant=False,
                memory_incidence_rescale=False)
    else:
        raise ValueError(family)
    if d >= 2:
        for modules in mixer.ca.values():
            for ca in modules.values():
                ca.plant = False
                ca.write_hash_neighbor_grad = True
                assert ca.read_k == 4 and ca.dim == d-1
    return mixer


class Block(nn.Module):
    def __init__(self, mixer, layer, seed):
        super().__init__()
        width = PROTOCOL['width']
        self.mixer = mixer
        # Common task layers are identical at initialization across all configurations.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+50000+layer)
            self.norm1 = nn.RMSNorm(width)
            self.norm2 = nn.RMSNorm(width)
            self.ffn = nn.Sequential(nn.Linear(width, PROTOCOL['ffn_width']), nn.GELU(),
                nn.Linear(PROTOCOL['ffn_width'], width))

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class MKARModel(nn.Module):
    def __init__(self, config, seed):
        super().__init__()
        self.config = dict(config)
        p=PROTOCOL; w=p['width']
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+40000)
            self.key_embed = nn.Embedding(p['keys']+1, w, padding_idx=0)
            self.slot_embed = nn.Embedding(p['records']+1, w, padding_idx=0)
            self.value_embed = nn.Embedding(p['values']+1, w, padding_idx=0)
            self.role_embed = nn.Embedding(3, w, padding_idx=0)
            self.size_embed = nn.Embedding(max(p['requested_sizes'])+1, w)
            for emb in (self.key_embed,self.slot_embed,self.value_embed,self.role_embed,self.size_embed):
                nn.init.normal_(emb.weight,std=.02)
                if emb.padding_idx is not None:
                    with torch.no_grad():emb.weight[emb.padding_idx].zero_()
            self.final_norm=nn.RMSNorm(w)
            self.head=nn.Linear(w,p['records']*(p['values']+1))
        blocks=[]
        for layer in range(p['layers']):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+1000+layer)
                mixer=make_mixer(config,layer)
            blocks.append(Block(mixer,layer,seed))
        self.blocks=nn.ModuleList(blocks)
        self.incidence_audit=[]
        for layer,block in enumerate(self.blocks):
            mixer=block.mixer
            if config['d'] < 2:continue
            for modules in mixer.ca.values():
                for ca in modules.values():
                    ca.sparse_ops=True
                    row=dict(layer=layer,heads=mixer.h,head_dim=mixer.p,
                        state_dim=PROTOCOL['state_dim'],profiles=ca.N0,reads_per_head=ca.read_k)
                    if ca.mode == 'plane':
                        original=ca.M.detach().cpu().numpy()
                        if config['random_incidence']:
                            assert not getattr(ca,'address_reads',False)
                            replacement,audit=curveball(original,np.random.SeedSequence([seed,73001,layer]))
                            ca.M.copy_(torch.from_numpy(replacement).to(ca.M.dtype))
                            # No unique coset exists after unrestricted rewiring.
                            # Independent top-four selection never uses coset_of.
                            ca.coset_of.fill_(-1)
                            row.update(audit)
                        row.update(types=ca.D*ca.n_cosets,
                            cells_per_summary=int(ca.M.sum(-1).flatten()[0]),
                            profile_degree=int(ca.M.sum((0,1)).flatten()[0]))
                    else:
                        row.update(types=ca.N0,cells_per_summary=1,profile_degree=1)
                    row['profile_bytes_per_example']=row['profiles']*mixer.h*PROTOCOL['state_dim']*mixer.p*4
                    row['read_table_bytes_per_example']=row['types']*mixer.h*PROTOCOL['state_dim']*mixer.p*4
                    self.incidence_audit.append(row)

    def encode(self,tokens,requests,k):
        key,slot,value=tokens.unbind(-1)
        role=(key!=0).long()
        x=self.key_embed(key)+self.slot_embed(slot)+self.value_embed(value)+self.role_embed(role)
        query=self.key_embed(requests).sum(1)/k.float().sqrt()[:,None]
        query=query+self.role_embed.weight[2]+self.size_embed(k)
        return torch.cat((x[:,:-1],query[:,None]),dim=1)

    def forward(self,tokens,requests,k):
        x=self.encode(tokens,requests,k)
        for block in self.blocks:x=block(x)
        return self.head(self.final_norm(x[:,-1])).reshape(-1,PROTOCOL['records'],PROTOCOL['values']+1)

    def auxiliary(self):
        return sum(block.mixer.get_auxiliary_loss() for block in self.blocks
            if hasattr(block.mixer,'get_auxiliary_loss'))

    def audit(self):
        return dict(parameters=sum(x.numel() for x in self.parameters()),
            trainable_parameters=sum(x.numel() for x in self.parameters() if x.requires_grad),
            incidence=self.incidence_audit,
            residual_width=PROTOCOL['width'], layers=PROTOCOL['layers'],
            head_dim=PROTOCOL['head_dim'],
            mixer_heads=(PROTOCOL['softmax_heads'] if self.config['family']=='softmax'
                else self.blocks[0].mixer.h),
            state_dim=(None if self.config['family']=='softmax' else PROTOCOL['state_dim']),
            gdn_transport=('boundary transport including scalar decay'
                if self.config['family']=='gdn' and self.config['d']>=2 else None),
            matched_resources='within-backbone geometry/random pair only; native and '
                'different d/backbones are not equal-memory or equal-FLOP comparisons')
