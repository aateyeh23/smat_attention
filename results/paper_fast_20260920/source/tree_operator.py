"""Differentiable binary-tree evaluation of the WEAK Log-Linear operator.

Each node stores its state transition and zero-initial-state write summary.
Queries in a right child read the left child's summary with that level's
lambda. This avoids sequence-by-sequence dense matrices and global solves.
Experimental: kept separate from the frozen production training operator.
"""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def tree_core(q, k, v, g, lam, beta=None):
    dtype = v.dtype
    length = q.shape[1]
    with torch.autocast('cuda', enabled=False):
        q,k,v,g,lam = [x.float().transpose(1,2) for x in (q,k,v,g,lam)]
        if beta is not None:
            beta = beta.float().transpose(1,2)
            q = q * torch.rsqrt(q.square().sum(-1,keepdim=True)+1e-6)
            k = k * torch.rsqrt(k.square().sum(-1,keepdim=True)+1e-6)
        padded = 1 << (length-1).bit_length()
        padding = padded-length
        q,k,v,lam = [F.pad(x,(0,0,0,padding)) for x in (q,k,v,lam)]
        g = F.pad(g,(0,padding))
        if beta is not None:
            beta = F.pad(beta,(0,padding))
        a = g.exp()[...,None,None]
        writes = k[..., :,None] * v[...,None,:]
        qk = (q*k).sum(-1,keepdim=True)
        out = qk * v * lam[...,0,None]
        if beta is None:
            transition = a
            queries = (q*a.squeeze(-1)).unsqueeze(-2)
        else:
            writes = writes * beta[...,None,None]
            out = out * beta[...,None]
            eye = torch.eye(k.shape[-1],device=k.device,dtype=k.dtype)
            transition = (eye-beta[...,None,None]*k[..., :,None]*k[...,None,:])*a
            queries = ((q-beta[...,None]*qk*k)*a.squeeze(-1)).unsqueeze(-2)
        for level in range(1,(padded-1).bit_length()+1):
            qleft,qright = queries[:,:,0::2],queries[:,:,1::2]
            wleft,wright = writes[:,:,0::2],writes[:,:,1::2]
            aleft,aright = transition[:,:,0::2],transition[:,:,1::2]
            read = qright @ wleft
            weights = lam[...,level].reshape(*read.shape[:3],2*read.shape[-2])
            read = read * weights[...,read.shape[-2]:,None]
            out = out + torch.cat((torch.zeros_like(read),read),dim=-2).flatten(2,3)
            if level < (padded-1).bit_length():
                if beta is None:
                    queries = torch.cat((qleft,qright*aleft),dim=-2)
                    writes = aright*wleft+wright
                    transition = aright*aleft
                else:
                    queries = torch.cat((qleft,qright@aleft),dim=-2)
                    writes = aright@wleft+wright
                    transition = aright@aleft
        return out[:,:,:length].transpose(1,2).to(dtype).contiguous()


def log_linear(q,k,v,g,lam,beta=None,chunk_size=16):
    outputs=[]
    for start in range(0,q.shape[0],chunk_size):
        args=[x[start:start+chunk_size] for x in (q,k,v,g,lam)]
        if beta is not None: args.append(beta[start:start+chunk_size])
        if torch.is_grad_enabled() and q.shape[1]>256:
            outputs.append(checkpoint(tree_core,*args,use_reentrant=False,preserve_rng_state=False))
        else:
            outputs.append(tree_core(*args))
    return torch.cat(outputs,dim=0)
