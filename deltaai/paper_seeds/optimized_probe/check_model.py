"""Compare real checkpoint logits and parameter gradients on actual task data."""
import argparse
import importlib
import json
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
parser.add_argument('--family',required=True)
parser.add_argument('--backend',required=True,choices=['upstream','tree64'])
parser.add_argument('--checkpoint-campaign',choices=['dense','fast'],default='dense')
parser.add_argument('--full-validation',action='store_true')
args=parser.parse_args()
zm.Mamba2Block=SmatMamba2Block
root=Path('/u/archerdw/smat_attention/deltaai/results/paper_seeds_20260920')
joint_recall.configure_dataset(root/'joint')
data=joint_recall.load_data(root/'joint','shared','train')
checkpoint_root=root if args.checkpoint_campaign=='dense' else root.with_name('paper_fast_20260920')
checkpoint=torch.load(checkpoint_root/'joint_loglinear'/f'shared-{args.family}-d0-w64-s123/checkpoint.pt',map_location='cpu',weights_only=False)
config=ModelConfig(block_type='Mamba2Block',d_model=64,n_layers=2,
    sequence_mixer=dict(name='joint_loglinear.JointLogLinearMixer',kwargs=dict(family=args.family)),
    max_position_embeddings=0,vocab_size=joint_recall.VOCAB,name='checkpoint-check',embed_dropout=.1)
model=LanguageModel(config).cuda().train()
model.load_state_dict(checkpoint['model'])
dense=joint_loglinear.log_linear
optimized=importlib.import_module({'upstream':'upstream_adapter','tree64':'tree64_operator'}[args.backend]).log_linear
records=[]
for arrays in data:
    batch=joint_recall.batch(arrays,np.arange(4))
    values=[]
    for operator in [dense,optimized]:
        joint_loglinear.log_linear=operator
        torch.manual_seed(101);torch.cuda.manual_seed_all(101)
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits,targets=joint_recall.query_logits(model,batch)
            loss=torch.nn.functional.cross_entropy(logits.flatten(0,1).float(),targets.flatten())
        loss.backward()
        gradients=torch.cat([p.grad.flatten().float() for p in model.parameters()])
        values.append((logits.detach().float(),loss.item(),gradients))
    old,new=values
    errors=[((new[i]-old[i]).norm()/old[i].norm().clamp_min(1e-8)).item() for i in [0,2]]
    record=dict(family=args.family,backend=args.backend,checkpoint_steps=checkpoint['steps'],length=int(batch[0].shape[1]),logit_error=errors[0],gradient_error=errors[1],dense_loss=old[1],optimized_loss=new[1])
    records.append(record);print('MODEL_CHECK',json.dumps(record),flush=True)
    assert max(errors)<.06 and torch.isfinite(new[2]).all(),record
prefix='' if args.checkpoint_campaign=='dense' else 'fast-checkpoint-'
Path(__file__).with_name(f'{prefix}model-check-{args.backend}-{args.family}.json').write_text(json.dumps(records,indent=2)+'\n')
print('MODEL_CHECKS_PASSED',args.backend,args.family,flush=True)
if args.full_validation:
    valid=joint_recall.load_data(root/'joint','shared','validation')
    evaluations={}
    for name,operator in [('dense',dense),(args.backend,optimized)]:
        joint_loglinear.log_linear=operator
        model.eval()
        evaluations[name]=joint_recall.evaluate_all(model,valid,256)
        print('FULL_VALIDATION',name,json.dumps(evaluations[name]),flush=True)
    Path(__file__).with_name(f'{prefix}full-validation-{args.family}.json').write_text(json.dumps(evaluations,indent=2)+'\n')
