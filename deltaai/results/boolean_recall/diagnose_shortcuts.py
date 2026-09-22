"""Evaluation-only oracles: memory counts, or counts plus one queried bit."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
from boolean_recall import BooleanRecall

recipes=[json.loads(p.read_text()) for p in ROOT.glob('retrieval_and-*/recipe.json')]
groups=sorted({(q['load'],q['seed'],q['sequence_length']) for q in recipes})
output=[]
for load,seed,length in groups:
    task=BooleanRecall('retrieval_and',length)
    data=task.generate('boolean',load,4096,np.random.default_rng(np.random.SeedSequence([seed,30001,load])))
    pairs,tables=task.decode(data)
    ones=np.array([sum(t.values()) for t in tables])
    true=(data[2][:,0]-20).astype(bool)
    count=ones*(ones-1)/(load*(load-1))>.5
    row=dict(records=load,seed=seed,length=length,count_only_accuracy=float((count==true).mean()))
    for i,name in [(0,'first'),(1,'second')]:
        guess=(pairs[:,i]==1)&((ones-pairs[:,i])/(load-1)>.5)
        row[f'{name}_bit_plus_count_accuracy']=float((guess==true).mean())
    output.append(row)
(ROOT/'shortcut_controls.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
