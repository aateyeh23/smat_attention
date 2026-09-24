"""Original MKAR model shell with only the sequence mixer replaced."""
import numpy as np
import torch
from common import PROTOCOL
from mkar import MKARModel as OriginalModel
from smat.mask import build_mask
from smat.attention import to_device
from learned_mixers import make_mixer, curveball


class Model(OriginalModel):
    def __init__(self, config, seed):
        p = PROTOCOL
        # The entire shared shell, including output head, starts identically.
        # Softmax is the unchanged original arm, including QK LayerNorm.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            spec = to_device(build_mask(p['length'], 2, chunk=256), 'cpu')
            super().__init__(p['length'], p['width'], p['softmax_heads'], p['layers'],
                p['records'], p['vocab'], arm='softmax', short_conv=3, spec=spec,
                r=64, chunk=256, device='cpu', seed=seed, qk_norm=True,
                decay='mamba2', dt_init=(.1, 1.), A_init=(1., 16.),
                assign='content', read_mode='plane', hash_src='id', hash_freeze=True,
                hash_shift=1, hash_conv=0, dir_head='linear', dir_soft=False, dir_window=4)
        self.config = dict(config)
        self.incidence_audit = []
        if config['family'] != 'softmax':
            for layer, block in enumerate(self.blocks):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed+1000+layer)
                    block.attn = make_mixer(config, layer)
                mixer = block.attn
                if config['d'] < 2:
                    continue
                for modules in mixer.ca.values():
                    for ca in modules.values():
                        assert not ca.plant and ca.read_k == 4
                        row = dict(layer=layer, heads=mixer.h, head_dim=mixer.p,
                            state_dim=64, profiles=ca.N0, reads_per_head=4)
                        if ca.mode == 'plane':
                            original = ca.M.detach().cpu().numpy()
                            if config['random_incidence']:
                                replacement, audit = curveball(original,
                                    np.random.SeedSequence([seed,73001,layer]))
                                ca.M.copy_(torch.from_numpy(replacement).to(ca.M.dtype))
                                ca.coset_of.fill_(-1)
                                row.update(audit)
                            row.update(types=ca.D*ca.n_cosets,
                                cells_per_summary=int(ca.M.sum(-1).flatten()[0]),
                                profile_degree=int(ca.M.sum((0,1)).flatten()[0]))
                        else:
                            row.update(types=ca.N0,cells_per_summary=1,profile_degree=1)
                        row['profile_bytes_per_example']=row['profiles']*mixer.h*64*mixer.p*4
                        row['read_table_bytes_per_example']=row['types']*mixer.h*64*mixer.p*4
                        self.incidence_audit.append(row)

    def auxiliary(self):
        return sum(block.attn.get_auxiliary_loss() for block in self.blocks
                   if hasattr(block.attn, 'get_auxiliary_loss'))

    def audit(self):
        return dict(parameters=sum(p.numel() for p in self.parameters()),
            trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
            incidence=self.incidence_audit, oracle_directions=False,
            write_hash_neighbor_grad=True, memory_tied_features=True,
            detach_write_hash_input=self.config.get('detach_write_hash_input', False),
            memory_boundary_transport=self.config.get('memory_boundary_transport', False),
            memory_incidence_rescale=self.config.get('memory_incidence_rescale', False),
            detach_write_hash_features=self.config.get('detach_write_hash_features', False),
            memory_read_normalization='inverse mean incidence weight',
            memory_transition='GDN key-dependent delta transport; local Mamba unchanged',
            shared_shell='Original MKARModel/Block unchanged',
            native_implementation='FLA GDN / Mamba2 library block, not original custom recurrence',
            matched_resources='Same residual width/common shell/training exposure; not equal state memory or FLOPs')
