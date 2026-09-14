"""Compatibility shim, picked up automatically because this folder is first on
PYTHONPATH (see _prelude.sh).  The site PyTorch env ships numpy 1.26, and the
repo uses numpy.bitwise_count (numpy >= 2.0) in smat_mask.py and test_smat.py.
Defines an equivalent popcount when it is missing; a no-op otherwise."""
import numpy as _np

if not hasattr(_np, "bitwise_count"):
    def bitwise_count(x, /, **kw):
        x = _np.asarray(x)
        u = x.astype(_np.uint64) if x.dtype.kind == "i" else x
        out = _np.zeros(u.shape, dtype=_np.uint8)
        while _np.any(u):
            out += (u & 1).astype(_np.uint8)
            u = u >> 1
        return out
    _np.bitwise_count = bitwise_count
