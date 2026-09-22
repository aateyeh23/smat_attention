"""Evaluate immutable PG19 checkpoints by position, retaining full causal context."""
import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F


def main(args):
    torch.set_num_threads(4)
    root = Path(args.checkpoint).parent
    manifest = json.loads((root / 'manifest.json').read_text())
    source_root = Path('/opt/pg19-rank64')
    for relative, expected in manifest['source_sha256'].items():
        actual = hashlib.sha256((source_root / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Source mismatch: {relative}')
    checkpoint = Path(args.checkpoint)
    if not checkpoint.name.startswith('model-tokens-'):
        raise ValueError('Use an immutable model milestone, not a moving latest checkpoint')
    with checkpoint.open('rb') as fp:
        checkpoint_sha256 = hashlib.file_digest(fp, 'sha256').hexdigest()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    family, d = manifest['family'], manifest['d']
    module_name = ('pg19_baselines' if d == 1 else
                   'gdn_smat_transport_scale' if family == 'gdn' else 'mamba_smat_scale')
    module = importlib.import_module(module_name)
    if Path(module.__file__).resolve() != source_root / 'lm' / (module_name + '.py'):
        raise ValueError('Unexpected model import path')
    if manifest.get('kernel_optimization') == 'tiled_fp32_v1':
        import pg19_optimized_kernels
        pg19_optimized_kernels.enable()
    cfg = module.ScaleConfig(**saved['config'])
    torch.manual_seed(cfg.seed)
    model = module.ScaleLM(cfg).cuda().eval()
    model.load_state_dict(saved['model'], strict=True)
    recipe = saved['recipe']
    expected_validation = saved['validation']
    tokens, step = saved['tokens'], saved['step']
    del saved
    data_root = Path(args.data)
    data_manifest = json.loads((data_root / 'manifest.json').read_text())
    if data_manifest['length'] != cfg.length or data_manifest['tokenizer'] != 'gpt2':
        raise ValueError('Dataset recipe mismatch')
    if data_manifest['splits']['train']['sha256'] != recipe['train_sha256']:
        raise ValueError('Training data provenance mismatch')
    valid = np.memmap(data_root / 'val.bin', mode='r', dtype=np.uint16).reshape(-1, cfg.length + 1)
    rows = recipe['eval_rows']
    if rows != np.random.default_rng(991).permutation(len(valid))[:len(rows)].tolist():
        raise ValueError('Validation row selection changed')
    chunk = 2048
    if cfg.length != 16384:
        raise ValueError('This evaluation is defined for the 16K experiment')
    losses = []
    start = time.monotonic()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for index, row_id in enumerate(rows):
            raw = torch.from_numpy(np.array(valid[row_id:row_id+1], dtype=np.int64)).cuda()
            hidden, _ = model.hidden(raw[:, :-1])
            targets = raw[:, 1:]
            bins = []
            for begin in range(0, cfg.length, chunk):
                logits = F.linear(hidden[:, begin:begin+chunk], model.embedding.weight).float()
                loss = F.cross_entropy(logits.flatten(0, 1), targets[:, begin:begin+chunk].flatten(), reduction='mean')
                bins.append(loss.item())
            losses.append(bins)
            if (index + 1) % 16 == 0:
                print(json.dumps(dict(event='eval_progress', arm=root.name, rows=index+1)), flush=True)
    array = np.asarray(losses, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite evaluation loss')
    means = dict(full=float(array.mean()), last_8k=float(array[:, 4:].mean()), last_2k=float(array[:, -1].mean()))
    # Milestone validation uses exactly these 64 rows; final validation uses all rows.
    reproduced = None
    if not expected_validation['full'] and expected_validation['rows'] == len(rows):
        reproduced = abs(means['full'] - expected_validation['loss'])
        if reproduced > 5e-4:
            raise ValueError(f'Full-window NLL failed reproduction: {reproduced}')
    result = dict(arm=root.name, training_tokens=tokens, step=step,
                  checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha256,
                  config=asdict(cfg), source_sha256=manifest['source_sha256'],
                  rows=rows, context_length=cfg.length,
                  scored_last_2k_positions_zero_based=[14336, 16384],
                  scored_last_2k_token_count=len(rows)*chunk,
                  nll=means, perplexity={key:math.exp(value) for key,value in means.items()},
                  per_2k_bin_nll=array.mean(axis=0).tolist(),
                  per_row_per_2k_bin_nll=losses,
                  original_validation=expected_validation,
                  full_nll_reproduction_absolute_error=reproduced,
                  evaluation_seconds=time.monotonic()-start)
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print('RESULT '+json.dumps({k:result[k] for k in ('arm','training_tokens','nll','perplexity','full_nll_reproduction_absolute_error','evaluation_seconds')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', default='/data/pg19-16k-2b')
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
