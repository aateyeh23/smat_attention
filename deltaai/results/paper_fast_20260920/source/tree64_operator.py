"""Use larger batch chunks to reduce Python/kernel launch overhead."""
from tree_operator import log_linear as tree_log_linear

def log_linear(q,k,v,g,lam,beta=None):
    return tree_log_linear(q,k,v,g,lam,beta,chunk_size=256 if q.shape[1]<=256 else 64)
