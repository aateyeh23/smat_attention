"""Compare full two-layer training steps at the actual five joint-recall cells."""
import argparse
import json
import statistics
import time
from pathlib import Path
import numpy as np
import torch
import joint_recall
import joint_loglinear
from zoology.config import ModelConfig
from zoology.model import LanguageModel
import zoology.mixers.mamba2 as zm
from zoo_smat_mixer import SmatMamba2Block

parser=argparse.ArgumentParser()
parser.add_argument('--family',required=True,choices=['mamba2','gdn'])
parser.add_argument('--backend',required=True,choices=['dense','upstream','tree','tree64'])
args=parser.parse_args()
if args.backend=='upstream':
    from upstream_adapter import log_linear
    joint_loglinear.log_linear=log_linear
elif args.backend=='tree':
    from tree_operator import log_linear
    joint_loglinear.log_linear=log_linear
elif args.backend=='tree64':
    from tree64_operator import log_linear
    joint_loglinear.log_linear=log_linear
zm.Mamba2Block=SmatMamba2Block
root=Path('/u/archerdw/smat_attention/deltaai/results/paper_seeds_20260920')
joint_recall.configure_dataset(root/'joint')
arrays=joint_recall.load_data(root/'joint','shared','train')
records=[]
for data in arrays:
    batch=joint_recall.batch(data,np.arange(256))
    torch.manual_seed(999)
    config=ModelConfig(block_type='Mamba2Block',d_model=64,n_layers=2,
        sequence_mixer=dict(name='joint_loglinear.JointLogLinearMixer',kwargs=dict(family=args.family)),
        max_position_embeddings=0,vocab_size=joint_recall.VOCAB,name='speed-probe',embed_dropout=.1)
    model=LanguageModel(config).cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.1)
    timings=[]
    torch.cuda.reset_peak_memory_stats()
    for index in range(8):
        torch.cuda.synchronize();start=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits,targets=joint_recall.query_logits(model,batch)
            loss=torch.nn.functional.cross_entropy(logits.flatten(0,1).float(),targets.flatten())
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        assert loss.isfinite() and all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
        if index>=3:timings.append(elapsed)
    record=dict(family=args.family,backend=args.backend,length=int(batch[0].shape[1]),batch_size=256,
                mean_seconds=statistics.mean(timings),median_seconds=statistics.median(timings),
                samples_seconds=timings,peak_memory_gb=torch.cuda.max_memory_allocated()/1e9,
                parameters=sum(p.numel() for p in model.parameters()),loss=loss.item(),gpu=torch.cuda.get_device_name())
    records.append(record);print('TIMING',json.dumps(record),flush=True)
    Path(__file__).with_name(f'timing-{args.backend}-{args.family}.json').write_text(json.dumps(records,indent=2)+'\n')
    del model,optimizer,logits,targets,loss;torch.cuda.empty_cache()
print('BENCHMARK_COMPLETE',args.backend,args.family,flush=True)
