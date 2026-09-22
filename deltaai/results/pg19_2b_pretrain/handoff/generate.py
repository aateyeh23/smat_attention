"""Load an exported checkpoint with the frozen model sources and generate text."""
import argparse
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
paths = ('optimized', 'optimized/lm', 'base/deltaai', 'base/smat',
         'base/zoology', 'base/fla')
if not (root / 'optimized/lm/gdn_smat_scale.py').exists():
    raise SystemExit('Extract source-bundle.tar.gz into this directory first.')
sys.path[:0] = [str(root / p) for p in paths]
os.environ.setdefault('FLA_TILELANG', '0')
os.environ.setdefault('TRITON_F32_DEFAULT', 'tf32x3')

import torch
from transformers import GPT2TokenizerFast
from gdn_smat_scale import ScaleConfig, ScaleLM

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--prompt', required=True)
parser.add_argument('--max-new-tokens', type=int, default=32)
args = parser.parse_args()
saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
config = ScaleConfig(**saved['config'])
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
ids = tokenizer(args.prompt, return_tensors='pt').input_ids
if ids.shape[1] == 0 or args.max_new_tokens < 1:
    raise SystemExit('Use a nonempty prompt and a positive number of new tokens.')
if ids.shape[1] + args.max_new_tokens > config.length:
    raise SystemExit(f'Prompt plus answer must fit within {config.length} tokens.')
model = ScaleLM(config).cuda().eval()
model.load_state_dict(saved['model'], strict=True)
del saved
with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
    output = model.generate(ids.cuda(), max_new_tokens=args.max_new_tokens,
                            eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(output[0].tolist(), skip_special_tokens=True))
