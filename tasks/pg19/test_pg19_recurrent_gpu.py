"""Checkpoint-level cached/full inference checks and an actual decode benchmark."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from pg19_recurrent import RecurrentDecoder


def load_model(checkpoint):
    root=Path(checkpoint).parent
    manifest=json.loads((root/'manifest.json').read_text())
    for name,expected in manifest['source_sha256'].items():
        assert hashlib.sha256((Path('/opt/pg19-rank64')/name).read_bytes()).hexdigest()==expected,name
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    module=importlib.import_module('pg19_baselines' if manifest['d']==1 else
        'gdn_smat_transport_scale' if manifest['family']=='gdn' else 'mamba_smat_scale')
    if manifest.get('kernel_optimization')=='tiled_fp32_v1':
        import pg19_optimized_kernels
        pg19_optimized_kernels.enable()
    cfg=module.ScaleConfig(**saved['config'])
    model=module.ScaleLM(cfg).cuda().eval()
    model.load_state_dict(saved['model'],strict=True)
    return model,manifest


def full_logits(model,rows):
    tokens=torch.full((len(rows),model.cfg.length),50256,device='cuda',dtype=torch.long)
    for i,row in enumerate(rows):
        tokens[i,:len(row)]=torch.tensor(row,device='cuda')
    hidden,_=model.hidden(tokens)
    return F.linear(hidden[torch.arange(len(rows),device='cuda'),
        torch.tensor([len(row)-1 for row in rows],device='cuda')],model.embedding.weight).float()


def main(args):
    torch.set_num_threads(4)
    model,manifest=load_model(args.checkpoint)
    data=[json.loads(line)['input_ids'] for line in Path(args.data).read_text().splitlines()][:2]
    results=[]
    for precision in ('fp32','bf16'):
        for near_boundary in (True,False):
            rows=[r[:8193+8*i] if near_boundary else list(r) for i,r in enumerate(data)]
            cache=RecurrentDecoder(model)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
                start=time.monotonic()
                cached=cache.prefill(rows)
                torch.cuda.synchronize()
                prefill_seconds=time.monotonic()-start
                reference=full_logits(model,rows)
                prefill_error=(cached-reference).float().square().mean().sqrt().item()
                print('PREFILL '+json.dumps(dict(precision=precision,near_boundary=near_boundary,
                    cached_rms=prefill_error)),flush=True)
                assert prefill_error < (.002 if precision=='fp32' else .06)
                checks=[]
                for step in range(args.steps):
                    next_ids=reference.argmax(-1)
                    for row,token in zip(rows,next_ids.tolist()):
                        row.append(token)
                    cached=cache.step(next_ids)
                    reference=full_logits(model,rows)
                    diff=(cached-reference).float()
                    kl=(reference.softmax(-1)*(reference.log_softmax(-1)-cached.log_softmax(-1))).sum(-1)
                    check=dict(step=step,max_abs=diff.abs().max().item(),rms=diff.square().mean().sqrt().item(),
                        max_kl=kl.max().item(),top1_matches=(cached.argmax(-1)==reference.argmax(-1)).sum().item())
                    checks.append(check)
                    print('CHECK '+json.dumps(dict(precision=precision,near_boundary=near_boundary,**check)),flush=True)
                result=dict(precision=precision,near_boundary=near_boundary,prefill_seconds=prefill_seconds,
                    prefill_rms=prefill_error,checks=checks)
                results.append(result)
            # Numerical gates: FP32 is the algorithm check; BF16 additionally
            # exercises the real evaluation kernels and their rounding error.
            if precision=='fp32':
                assert max(c['rms'] for c in checks)<.002,results[-1]
                assert max(c['max_abs'] for c in checks)<.03,results[-1]
            else:
                assert max(c['rms'] for c in checks)<.06,results[-1]
                assert max(c['max_kl'] for c in checks)<.005,results[-1]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        rows=[list(r) for r in data]
        cache=RecurrentDecoder(model)
        logits=cache.prefill(rows)
        # Prevent an accidental fallback to full-window inference in decode.
        original_hidden=model.hidden
        def forbid_full_window(*args,**kwargs):
            raise AssertionError('Decode re-entered the full-window model')
        model.hidden=forbid_full_window
        try:
            for _ in range(4):
                logits=cache.step(logits.argmax(-1))
            torch.cuda.synchronize()
            start=time.monotonic()
            for _ in range(64):
                logits=cache.step(logits.argmax(-1))
            torch.cuda.synchronize()
            seconds=time.monotonic()-start
        finally:
            model.hidden=original_hidden
    report=dict(arm=Path(args.checkpoint).parent.name,passed=True,checks=results,
        decoder_sha256=hashlib.sha256(Path(__file__).with_name('pg19_recurrent.py').read_bytes()).hexdigest(),
        decode_batch=2,decode_steps=64,decode_seconds=seconds,
        milliseconds_per_step=seconds/64*1000,output_tokens_per_second=128/seconds,
        gpu_peak_gb=torch.cuda.max_memory_allocated()/1e9)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print('RESULT '+json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--data',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=8)
    main(p.parse_args())
