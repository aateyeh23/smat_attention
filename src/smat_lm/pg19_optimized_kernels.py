"""Opt-in, shape-guarded FP32 kernels for the width-768 PG19 GDN campaign."""
import torch
import smat_read_triton
import smat_write_hash_triton
import smat_read_tiled
import smat_write_hash_tiled
import smat_gdn_tiled

_original_read=smat_read_triton.read_topk
_original_write=smat_write_hash_triton.write_hash_gradient


def _h100(t):
    return t.is_cuda and torch.cuda.get_device_capability(t.device)==(9,0)


def _read(q,idx,weights,memory):
    if (_h100(q) and q.dtype==weights.dtype==memory.dtype==torch.float32
            and q.shape[-1]==memory.shape[-1]==128 and q.shape[1]>=1024):
        return smat_read_tiled.read_topk(q,idx,weights,memory)
    return _original_read(q,idx,weights,memory)


def _write(memory_grad,keys,values,indices,dtype):
    if (_h100(keys) and keys.dtype==values.dtype==memory_grad.dtype==torch.float32
            and keys.shape[-1]==values.shape[-1]==128 and keys.shape[1]>=1024):
        return smat_write_hash_tiled.write_hash_gradient(memory_grad,keys,values,indices,dtype)
    return _original_write(memory_grad,keys,values,indices,dtype)


def enable():
    smat_gdn_tiled.enable()
    smat_read_triton.read_topk=_read
    smat_write_hash_triton.write_hash_gradient=_write


def disable():
    smat_gdn_tiled.disable()
    smat_read_triton.read_topk=_original_read
    smat_write_hash_triton.write_hash_gradient=_original_write
