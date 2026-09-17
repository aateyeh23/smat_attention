"""Matched-token PG19 training with resumable state and retained model milestones."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
if os.environ.get('PG19_MEMORY_VARIANT') == 'updated_transport':
    from gdn_smat_transport_scale import ScaleConfig, ScaleLM
else:
    from gdn_smat_scale import ScaleConfig, ScaleLM


def atomic_json(path, obj):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, indent=2)+'\n')
    temporary.replace(path)


def atomic_save(path, obj):
    temporary = path.with_suffix('.tmp')
    torch.save(obj, temporary)
    temporary.replace(path)


def train(args):
    cfg = ScaleConfig(d=args.d, layers=args.layers, width=args.width,
                      ffn_width=args.ffn_width, length=args.length, head_dim=args.head_dim,
                      activation_checkpointing=getattr(args, 'activation_checkpointing', False), loss_chunk=args.loss_chunk,
                      read_backend=args.backend, fused_norm=True)
    torch.set_num_threads(4)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data)
    manifest = json.loads((data_root/'manifest.json').read_text())
    if manifest['length'] != cfg.length or manifest['tokenizer'] != 'gpt2':
        raise ValueError('Dataset and model recipes differ')
    train_data = np.memmap(data_root/'train.bin', mode='r', dtype=np.uint16).reshape(-1, cfg.length+1)
    valid_data = np.memmap(data_root/'val.bin', mode='r', dtype=np.uint16).reshape(-1, cfg.length+1)
    # One shared shuffled pass. Row order depends only on the seed, never the arm.
    order = np.random.default_rng(cfg.seed).permutation(len(train_data))
    target_rows = math.ceil(args.tokens/cfg.length)
    global_rows = getattr(args, 'global_batch_rows', None) or args.batch*args.accumulation
    if target_rows > len(order):
        raise ValueError('Requested more unique tokens than the prepared corpus')
    eval_rows = np.random.default_rng(991).permutation(len(valid_data))[:args.eval_rows]
    model = ScaleLM(cfg).cuda().train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95),
                                 weight_decay=0.1, fused=True)
    recipe = dict(config=asdict(cfg), target_tokens=args.tokens, target_rows=target_rows,
                  batch=args.batch, accumulation=args.accumulation, lr=args.lr,
                  global_batch_rows=global_rows,
                  warmup_tokens=args.warmup_tokens, train_sha256=manifest['splits']['train']['sha256'],
                  eval_rows=eval_rows.tolist(), seed=cfg.seed)
    recipe_path = out/'recipe.json'
    if recipe_path.exists() and json.loads(recipe_path.read_text()) != recipe:
        raise ValueError('Refusing to resume a changed experiment recipe')
    atomic_json(recipe_path, recipe)
    cursor, step, elapsed_before = 0, 0, 0.0
    next_eval, next_milestone = args.eval_every, args.checkpoint_every
    checkpoint = out/'latest.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        cursor, step = saved['cursor'], saved['step']
        elapsed_before = saved['elapsed_seconds']
        next_eval, next_milestone = saved['next_eval'], saved['next_milestone']
        torch.set_rng_state(saved['rng_cpu'])
        torch.cuda.set_rng_state_all(saved['rng_cuda'])
        del saved
    start = time.monotonic()
    last_save = start
    log_clock, log_cursor = start, cursor
    last_eval = None
    print(json.dumps(dict(event='initialized', gpu=torch.cuda.get_device_name(),
                          trainable_parameters=sum(p.numel() for p in params),
                          recipe=recipe, resumed_step=step)), flush=True)

    def batch_from(data, rows):
        raw = torch.from_numpy(np.asarray(data[rows], dtype=np.int64)).cuda()
        return raw[:, :-1], raw[:, 1:]

    @torch.no_grad()
    def evaluate(full=False):
        model.eval()
        rows = np.arange(len(valid_data)) if full else eval_rows
        total = torch.zeros((), device='cuda')
        for begin in range(0, len(rows), args.batch):
            ids = rows[begin:begin+args.batch]
            x, y = batch_from(valid_data, ids)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, _ = model.loss(x, y)
            total += loss*len(ids)
        loss = (total/len(rows)).item()
        model.train()
        return dict(loss=loss, perplexity=math.exp(loss), rows=len(rows), full=full)

    def save(milestone=False, complete=False):
        elapsed = elapsed_before+time.monotonic()-start
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                     config=asdict(cfg), recipe=recipe, cursor=cursor, step=step,
                     elapsed_seconds=elapsed, next_eval=next_eval, next_milestone=next_milestone,
                     rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all(),
                     validation=last_eval, complete=complete)
        atomic_save(checkpoint, state)
        if milestone or complete:
            atomic_save(out/f'model-tokens-{cursor*cfg.length:010d}.pt',
                        dict(model=state['model'], config=state['config'], tokens=cursor*cfg.length,
                             step=step, validation=last_eval, recipe=recipe))
        atomic_json(out/'status.json', dict(tokens=cursor*cfg.length, target_tokens=args.tokens,
                    step=step, elapsed_seconds=elapsed, validation=last_eval, complete=complete,
                    checkpoint=checkpoint.name))
        print(json.dumps(dict(event='checkpoint', tokens=cursor*cfg.length, step=step,
                              complete=complete, milestone=milestone)), flush=True)

    while cursor < target_rows:
        tokens = cursor*cfg.length
        if tokens < args.warmup_tokens:
            lr = args.lr*max(1, tokens)/max(1, args.warmup_tokens)
        else:
            fraction = min(1, (tokens-args.warmup_tokens)/max(1, args.tokens-args.warmup_tokens))
            lr = args.lr*(0.1+0.9*0.5*(1+math.cos(math.pi*fraction)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        rows_this_step = min(global_rows, target_rows-cursor)
        loss_sum = torch.zeros((), device='cuda')
        aux_sum = torch.zeros((), device='cuda')
        for begin in range(0, rows_this_step, args.batch):
            ids = order[cursor+begin:cursor+min(begin+args.batch, rows_this_step)]
            x, y = batch_from(train_data, ids)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, aux = model.loss(x, y)
                objective = (loss+aux)*len(ids)/rows_this_step
            objective.backward()
            loss_sum += loss.detach()*len(ids)/rows_this_step
            aux_sum += aux.detach()*len(ids)/rows_this_step
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        cursor += rows_this_step
        step += 1
        if step == 1 or step % args.log_every == 0:
            torch.cuda.synchronize()
            now = time.monotonic()
            record = dict(event='train', step=step, tokens=cursor*cfg.length,
                          loss=loss_sum.item(), auxiliary_loss=aux_sum.item(),
                          gradient_norm=grad_norm.item(), learning_rate=lr,
                          tokens_per_second=(cursor-log_cursor)*cfg.length/(now-log_clock),
                          elapsed_seconds=elapsed_before+now-start,
                          gpu_peak_gb=torch.cuda.max_memory_allocated()/1e9)
            with (out/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            print(json.dumps(record), flush=True)
            log_clock, log_cursor = now, cursor
        is_end = cursor == target_rows
        milestone = cursor*cfg.length >= next_milestone
        if cursor*cfg.length >= next_eval or milestone or is_end:
            last_eval = evaluate(full=is_end)
            record = dict(event='validation', step=step, tokens=cursor*cfg.length, **last_eval)
            with (out/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            print(json.dumps(record), flush=True)
            while next_eval <= cursor*cfg.length:
                next_eval += args.eval_every
            while next_milestone <= cursor*cfg.length:
                next_milestone += args.checkpoint_every
        time_limit = elapsed_before+time.monotonic()-start > args.max_hours*3600
        if milestone or is_end or time_limit or time.monotonic()-last_save > 600:
            save(milestone, is_end)
            last_save = time.monotonic()
        if time_limit and not is_end:
            print('TIME_LIMIT: resumable checkpoint saved', flush=True)
            return False
    return True


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--d', type=int, choices=(1,2,3,4), required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--tokens', type=int, default=2_000_000_000)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--global-batch-rows', type=int, default=None)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup-tokens', type=int, default=40_000_000)
    p.add_argument('--eval-every', type=int, default=50_000_000)
    p.add_argument('--eval-rows', type=int, default=64)
    p.add_argument('--checkpoint-every', type=int, default=250_000_000)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--max-hours', type=float, default=11.8)
    p.add_argument('--backend', choices=('torch','triton'), default='triton')
    p.add_argument('--layers', type=int, default=16)
    p.add_argument('--width', type=int, default=1024)
    p.add_argument('--ffn-width', type=int, default=2816)
    p.add_argument('--head-dim', type=int, default=64)
    p.add_argument('--length', type=int, default=16384)
    p.add_argument('--loss-chunk', type=int, default=4096)
    p.add_argument('--activation-checkpointing', action='store_true')
    args = p.parse_args()
    print('TRAIN_COMPLETE '+json.dumps(train(args)), flush=True)
