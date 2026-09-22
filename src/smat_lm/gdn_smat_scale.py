"""Candidate 200--251M GDN/SMAT language models and a bounded GPU throughput pilot.

Training and inference use the same fixed geometry. Short prompts are padded
on the right for causal prefill; generation recomputes that fixed window.
"""
from dataclasses import asdict, dataclass
import argparse
from contextlib import nullcontext
import functools
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from zoo_smat_gdn import SmatGDNReset


@dataclass
class ScaleConfig:
    d: int = 1
    width: int = 1024
    layers: int = 16
    ffn_width: int = 2816
    vocab: int = 50257
    length: int = 16384
    heads: int = 2
    head_dim: int = 64
    read_k: int = 4
    seed: int = 123
    activation_checkpointing: bool = True
    loss_chunk: int = 512
    read_backend: str = 'torch'
    fused_norm: bool = False


class RMSNorm(nn.Module):
    def __init__(self, width, fused=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.fused = fused

    def forward(self, x):
        if self.fused:
            from fla.modules.layernorm import rms_norm
            return rms_norm(x, self.weight, None, eps=1e-5)
        y = x.float()
        return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-5)).to(x.dtype) * self.weight


class Block(nn.Module):
    def __init__(self, cfg, index):
        super().__init__()
        if cfg.read_backend == 'triton':
            import inspect
            from content_addr import ContentAssign
            if 'read_topk' not in inspect.getsource(ContentAssign.forward):
                raise RuntimeError('Fused backend requested but reference module deployed')
        self.norm = RMSNorm(cfg.width, cfg.fused_norm)
        self.mixer = SmatGDNReset(
            cfg.width, layer_idx=index, d=cfg.d, headdim=cfg.head_dim,
            n_heads=cfg.heads, expand_v=1, reset=True,
            prebuild_lengths=(cfg.length,), memory_update='delta',
            memory_key_mode='tied_causal',
            memory_read_k=cfg.read_k if cfg.d >= 2 else None,
        )
        self.norm2 = RMSNorm(cfg.width, cfg.fused_norm)
        for geometry in self.mixer.ca.values():
            for ca in geometry.values():
                ca.read_backend = cfg.read_backend
        self.up_gate = nn.Linear(cfg.width, 2 * cfg.ffn_width, bias=False)
        self.down = nn.Linear(cfg.ffn_width, cfg.width, bias=False)

    def forward(self, x):
        x = x + self.mixer(self.norm(x))
        aux = self.mixer.get_auxiliary_loss()
        if not torch.is_tensor(aux):
            aux = x.new_zeros((), dtype=torch.float32)
        gate, value = self.up_gate(self.norm2(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * value), aux


class ScaleLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab, cfg.width)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.layers)])
        self.norm = RMSNorm(cfg.width, cfg.fused_norm)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def hidden(self, tokens):
        if tokens.shape[1] != self.cfg.length:
            raise ValueError('hidden() requires the fixed training length; use next_logits() for short prompts')
        x = self.embedding(tokens)
        aux = x.new_zeros((), dtype=torch.float32)
        for block in self.blocks:
            if self.training and self.cfg.activation_checkpointing:
                x, term = checkpoint(block, x, use_reentrant=False)
            else:
                x, term = block(x)
            aux = aux + term
        return self.norm(x), aux

    @torch.inference_mode()
    def next_logits(self, tokens):
        """Last-prefix logits without changing the learned hash geometry or split."""
        if self.training:
            raise ValueError('Call eval() before generation')
        length = tokens.shape[1]
        if not 0 < length <= self.cfg.length:
            raise ValueError('Prompt must fit the fixed context window')
        padded = F.pad(tokens, (0, self.cfg.length-length), value=self.cfg.vocab-1)
        hidden, _ = self.hidden(padded)
        return F.linear(hidden[:, length-1], self.embedding.weight).float()

    @torch.inference_mode()
    def generate(self, tokens, max_new_tokens, eos_token_id=None):
        """Greedy reference generation; no recurrent cache, at most one trained window."""
        if max_new_tokens < 0 or tokens.shape[1]+max_new_tokens > self.cfg.length:
            raise ValueError('Reserve space for the answer inside the trained context window')
        ended = torch.zeros(tokens.shape[0], device=tokens.device, dtype=torch.bool)
        for _ in range(max_new_tokens):
            next_token = self.next_logits(tokens).argmax(-1)
            if eos_token_id is not None:
                next_token = torch.where(ended, eos_token_id, next_token)
                ended |= next_token == eos_token_id
            tokens = torch.cat((tokens, next_token[:, None]), dim=1)
            if eos_token_id is not None and ended.all():
                break
        return tokens

    def loss(self, tokens, targets):
        hidden, aux = self.hidden(tokens)
        hidden, targets = hidden.flatten(0, 1), targets.flatten()

        def piece(h, y):
            return F.cross_entropy(F.linear(h, self.embedding.weight).float(), y, reduction='sum')

        loss = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, targets.numel(), self.cfg.loss_chunk):
            end = start + self.cfg.loss_chunk
            h, y = hidden[start:end], targets[start:end]
            loss = loss + checkpoint(piece, h, y, use_reentrant=False)
        return loss / targets.numel(), aux


def benchmark(args):
    cfg = ScaleConfig(d=args.d, layers=args.layers,
                      activation_checkpointing=bool(args.checkpoint), loss_chunk=args.loss_chunk,
                      read_backend=args.read_backend, fused_norm=bool(args.fused_norm))
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.set_num_threads(4)
    started = time.perf_counter()
    model = ScaleLM(cfg).cuda().train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    allocated = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                           lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    data = np.memmap(args.data, dtype=np.uint16, mode='r')
    rng = np.random.default_rng(cfg.seed)
    print(json.dumps(dict(event='initialized', config=asdict(cfg), trainable=trainable,
                          allocated=allocated, gpu=torch.cuda.get_device_name(),
                          initialization_seconds=time.perf_counter() - started)), flush=True)
    times = []
    losses = []
    if args.profile:
        import content_addr
        import smat_delta_pool
        import smat_pool_ops

        def annotate(obj, method, label):
            fn = getattr(obj, method)
            @functools.wraps(fn)
            def wrapped(*a, **kw):
                with torch.profiler.record_function(label):
                    return fn(*a, **kw)
            setattr(obj, method, wrapped)

        for obj, name, label in [
            (content_addr.ContentAssign, '_cells', 'SMAT_write_hash'),
            (content_addr.ContentAssign, '_topk_reads', 'SMAT_read_routing'),
            (content_addr.ContentAssign, 'forward', 'SMAT_branch'),
            (smat_delta_pool, 'pack_routes', 'SMAT_pack_writes'),
            (smat_delta_pool, 'pool_delta', 'SMAT_delta_memory'),
            (smat_pool_ops, '_group', 'SMAT_group_reads'),
            (smat_pool_ops, 'read_sorted', 'SMAT_read'),
        ]:
            annotate(obj, name, label)
    for step in range(args.warmup + args.steps + int(args.profile)):
        offsets = rng.integers(0, len(data) - cfg.length - 1, size=args.batch)
        tokens = torch.from_numpy(np.stack([data[o:o + cfg.length + 1].astype(np.int64)
                                           for o in offsets])).cuda()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        profiling = args.profile and step == args.warmup + args.steps
        prof = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                  torch.profiler.ProfilerActivity.CUDA])
                if profiling else nullcontext())
        with prof:
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, aux = model.loss(tokens[:, :-1], tokens[:, 1:])
            total = loss + aux
            if not torch.isfinite(total).item():
                raise RuntimeError('Nonfinite pilot loss')
            total.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if step >= args.warmup and not profiling:
            times.append(elapsed)
            losses.append(float(loss.detach()))
        if profiling:
            averages = prof.key_averages()
            Path(args.output + '.profile.txt').write_text(averages.table(sort_by='self_cuda_time_total', row_limit=80))
            events = [dict(name=e.key, calls=e.count, self_cpu_us=e.self_cpu_time_total,
                           cpu_us=e.cpu_time_total, device_us=e.device_time_total,
                           self_device_us=e.self_device_time_total) for e in averages]
            Path(args.output + '.profile.json').write_text(json.dumps(events, indent=2))
        print(json.dumps(dict(event='step', d=cfg.d, step=step + 1, warmup=step < args.warmup,
                              seconds=elapsed, loss=float(loss.detach()), aux=float(aux.detach()),
                              grad_norm=float(grad), peak_gb=torch.cuda.max_memory_allocated()/1e9)), flush=True)
    rate = args.batch * cfg.length * len(times) / sum(times)
    result = dict(config=asdict(cfg), trainable=trainable, allocated=allocated,
                  gpu=torch.cuda.get_device_name(), batch=args.batch,
                  measured_steps=len(times), warmup_steps=args.warmup,
                  tokens_per_second=rate, mean_step_seconds=float(np.mean(times)),
                  min_step_seconds=min(times), max_step_seconds=max(times),
                  peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9,
                  hours_per_billion_tokens=1e9/rate/3600,
                  tokens_in_four_hours=rate*4*3600, tokens_in_six_hours=rate*6*3600,
                  pilot_loss_start=losses[0], pilot_loss_end=losses[-1],
                  elapsed_seconds=time.perf_counter()-started,
                  caveat='Short throughput pilot; excludes dataset preparation, compilation warmup and periodic evaluation from token rate.')
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print('BENCHMARK_RESULT ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument('--layers', type=int, default=16)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--checkpoint', type=int, choices=(0, 1), default=1)
    parser.add_argument('--loss-chunk', type=int, default=512)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--read-backend', choices=('torch', 'triton'), default='torch')
    parser.add_argument('--fused-norm', type=int, choices=(0,1), default=0)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    benchmark(parser.parse_args())
