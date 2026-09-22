"""Full two-layer training-step smoke at the largest joint-recall cell."""
import hashlib
import json
from pathlib import Path
import time
import torch
import numpy as np
import joint_recall
from zoology.config import ModelConfig
from zoology.model import LanguageModel
import zoology.mixers.mamba2 as zm
from zoo_smat_mixer import SmatMamba2Block
zm.Mamba2Block=SmatMamba2Block
root=Path(__file__).resolve().parent.parent/'results/paper_seeds_20260920'
joint_recall.configure_dataset(root/'joint')
arrays=joint_recall.load_data(root/'joint','shared','train')[-1]
batch=joint_recall.batch(arrays,np.arange(256))
results=[]
for family in ['mamba2','gdn']:
    torch.manual_seed(999)
    config=ModelConfig(block_type='Mamba2Block',d_model=64,n_layers=2,
        sequence_mixer=dict(name='joint_loglinear.JointLogLinearMixer',kwargs=dict(family=family)),
        max_position_embeddings=0,vocab_size=joint_recall.VOCAB,name='validation',embed_dropout=.1)
    model=LanguageModel(config).cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.1)
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits,targets=joint_recall.query_logits(model,batch)
            loss=torch.nn.functional.cross_entropy(logits.flatten(0,1).float(),targets.flatten())
        assert loss.isfinite()
        loss.backward()
        assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
        optimizer.step()
    torch.cuda.synchronize()
    record=dict(family=family,seconds_per_step=(time.monotonic()-started)/2,
                peak_memory_gb=torch.cuda.max_memory_allocated()/1e9,
                parameters=sum(p.numel() for p in model.parameters()),loss=loss.item())
    results.append(record);print('TRAIN_STEP_PASS',json.dumps(record),flush=True)
    del model,optimizer,logits,loss;torch.cuda.empty_cache()
gate=root/'loglinear_validated.json'
old=json.loads(gate.read_text())
gate.write_text(json.dumps(dict(operator_checks=old,model_checks=results,
    sha256=hashlib.sha256(Path(__file__).with_name('joint_loglinear.py').read_bytes()).hexdigest()),indent=2)+'\n')
print('INTEGRATION_PASSED',flush=True)
