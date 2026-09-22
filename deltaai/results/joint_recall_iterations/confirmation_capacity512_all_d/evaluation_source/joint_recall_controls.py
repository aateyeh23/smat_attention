"""Single-field majority oracles for tables in which every binding is queried."""
import numpy as np


def single_field_oracles(x,c,k,layout,vocab,values=16):
    value_base=vocab-values
    if layout=='record':
        ip=2+3*np.arange(c*k)
    elif layout=='block':
        ip=(1+np.arange(c)[:,None]*(1+2*k)+1+2*np.arange(k)).reshape(-1)
    else:raise ValueError(layout)
    scores={'key_only':[],'context_only':[]}
    for start in range(0,len(x),500):
        part=x[start:start+500];b=len(part);labels=part[:,ip+1]-value_base
        assert ((labels>=0)&(labels<values)).all()
        contexts=(part[:,ip-1] if layout=='record' else
                  np.repeat(part[:,1+np.arange(c)*(1+2*k)],k,axis=1))
        for name,identifiers in [('key_only',part[:,ip]),('context_only',contexts)]:
            counts=np.zeros((b,vocab,values),dtype=np.int16)
            np.add.at(counts,(np.arange(b)[:,None],identifiers,labels),1)
            scores[name].extend((counts.max(-1).sum(-1)/(c*k)).tolist())
    out={name:np.asarray(a) for name,a in scores.items()}
    # A deliberately strong control: choose the better single-field rule per
    # table. Each rule knows all stored values but ignores one identifier.
    out['best_single_field']=np.maximum(out['key_only'],out['context_only'])
    return out
