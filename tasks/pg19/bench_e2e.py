"""End-to-end cost of the PG-19 language models behind the paper's LM results.

    python tasks/pg19/bench_e2e.py --arm gdn-smat-d3 --mode train --length 16384 \
        --out results/pg19_e2e_efficiency/rows.jsonl

One (arm, mode, length, batch) per process: the GDN + SMAT arms enable the
campaign's tiled kernels by patching module globals, and a Mamba-2 forward
switches on autograd anomaly mode for the rest of the process (below), so arms
must not share an interpreter.

Arms.  Every recurrent arm is built from its recorded configuration in
results/pg19_six_500m/campaign.json with only the length changed, from sources
whose SHA-256 matches that campaign's manifests:

  gdn, gdn-smat-d2, gdn-smat-d3        as trained (GDN + SMAT with tiled_fp32_v1)
  gdn-smat-d3-nog                      d=3 with the memory branch off, reset kept:
                                       splits SMAT's cost into reset and memory
  mamba2                               as trained: the frozen zoology Mamba2.forward,
                                       which calls torch.autograd.set_detect_anomaly(True)
                                       and runs an unfused nn.Conv1d
  mamba2-clean                         same model, anomaly mode suppressed
  mamba2-fused                         same model on the fused path (use_mem_eff_path=True,
                                       mamba_ssm 2.3.2's kernel), anomaly suppressed
  mamba2-smat-d2, mamba2-smat-d3       as trained (its forward never calls
                                       Mamba2.forward, so anomaly mode stays off)
  softmax                              same width, depth, SwiGLU FFN and tied
                                       embedding; 6 heads of 128 (the GDN head
                                       shape), RoPE, PyTorch SDPA
  loglinear-mamba2                     the plain Mamba-2 layer with its SSD scan
                                       replaced by the upstream Log-Linear WEAK
                                       kernel (commit 7f86441; the port in
                                       third_party/log_linear).  The upstream
                                       GDN kernel failed its gradient check there
                                       and is not used.

Modes.
  train    forward + backward + clip + fused AdamW, bf16 autocast, loss_chunk 4096,
           no activation checkpointing (the campaign's settings)
  prefill  inference forward over the whole window, last-token logits
  decode   per-token latency after prefill: RecurrentDecoder (src/smat_lm/pg19_recurrent.py)
           for the recurrent arms, a preallocated KV cache for softmax.  Cached
           logits are compared with a full-window forward for a few steps first.

Weights and tokens are random.  Step time depends on shapes and kernels, not on
trained values; the one data-dependent quantity is SMAT's routing histogram,
which at initialisation is close to the balanced one the auxiliary loss enforces.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT/'src'/'smat_lm', ROOT/'third_party', ROOT/'third_party'/'log_linear'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gdn_smat_scale import ScaleLM as _ScaleLM, RMSNorm        # noqa: E402

CAMPAIGN = json.loads((ROOT/'results'/'pg19_six_500m'/'campaign.json').read_text())
AS_TRAINED = {'gdn': 'gdn-baseline', 'gdn-smat-d2': 'gdn-smat-d2', 'gdn-smat-d3': 'gdn-smat-d3',
              'gdn-smat-d3-nog': 'gdn-smat-d3', 'mamba2': 'mamba2-baseline',
              'mamba2-clean': 'mamba2-baseline', 'mamba2-fused': 'mamba2-baseline',
              'mamba2-smat-d2': 'mamba2-smat-d2', 'mamba2-smat-d3': 'mamba2-smat-d3',
              'softmax': 'gdn-baseline', 'loglinear-mamba2': 'mamba2-baseline'}
ARMS = tuple(AS_TRAINED)
RECURRENT_DECODE = ('gdn', 'gdn-smat-d2', 'gdn-smat-d3', 'mamba2', 'mamba2-smat-d2', 'mamba2-smat-d3')


def campaign_config(arm, length):
    cfg = dict(CAMPAIGN['arms'][AS_TRAINED[arm]]['config'])
    cfg['length'] = length
    return cfg


# ---------------------------------------------------------------- softmax arm
class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.h, self.dh = heads, width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        inv = 1.0 / (10000 ** (torch.arange(0, self.dh, 2).float() / self.dh))
        self.register_buffer('inv', inv, persistent=False)

    def rope(self, x, start):
        t = torch.arange(start, start + x.shape[2], device=x.device).float()
        f = torch.outer(t, self.inv)
        cos, sin = f.cos()[None, None], f.sin()[None, None]
        x1, x2 = x[..., ::2].float(), x[..., 1::2].float()
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2).to(x.dtype)

    def forward(self, x, cache=None, pos=0):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = self.rope(q, pos), self.rope(k, pos)
        if cache is not None:
            K, V = cache
            K[:, :, pos:pos + t], V[:, :, pos:pos + t] = k, v
            if t == 1:                       # one new query sees the whole cache
                # The cache length changes every step; left to choose, SDPA may take
                # cuDNN, which plans per shape.  Decode uses the flash / mem-efficient
                # kernels, as a serving stack would.
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                    y = F.scaled_dot_product_attention(q, K[:, :, :pos + 1], V[:, :, :pos + 1])
                return self.out(y.transpose(1, 2).reshape(b, t, -1))
            assert pos == 0, 'prefill starts at position 0'
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(b, t, -1))


class SoftmaxBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm, self.norm2 = RMSNorm(cfg.width, cfg.fused_norm), RMSNorm(cfg.width, cfg.fused_norm)
        self.mixer = Attention(cfg.width, cfg.heads)
        self.up_gate = nn.Linear(cfg.width, 2 * cfg.ffn_width, bias=False)
        self.down = nn.Linear(cfg.ffn_width, cfg.width, bias=False)

    def forward(self, x, cache=None, pos=0):
        x = x + self.mixer(self.norm(x), cache, pos)
        gate, value = self.up_gate(self.norm2(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * value), x.new_zeros((), dtype=torch.float32)


class SoftmaxLM(_ScaleLM):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab, cfg.width)
        self.blocks = nn.ModuleList([SoftmaxBlock(cfg) for _ in range(cfg.layers)])
        self.norm = RMSNorm(cfg.width, cfg.fused_norm)
        nn.init.normal_(self.embedding.weight, std=0.02)


class KVDecoder:
    """Preallocated per-layer K/V; each step attends one query to the cache."""

    def __init__(self, model, batch, max_len):
        blk = model.blocks[0].mixer
        shape = (batch, blk.h, max_len, blk.dh)
        w = model.embedding.weight
        self.model, self.pos = model, 0
        self.caches = [(torch.empty(shape, device=w.device, dtype=torch.bfloat16),
                        torch.empty(shape, device=w.device, dtype=torch.bfloat16)) for _ in model.blocks]

    def _run(self, x):
        for blk, cache in zip(self.model.blocks, self.caches):
            x, _ = blk(x, cache, self.pos)
        return F.linear(self.model.norm(x)[:, -1], self.model.embedding.weight).float()

    @torch.inference_mode()
    def prefill(self, tokens):
        self.pos = 0
        logits = self._run(self.model.embedding(tokens))
        self.pos = tokens.shape[1]
        return logits

    @torch.inference_mode()
    def step(self, ids):
        logits = self._run(self.model.embedding(ids[:, None]))
        self.pos += 1
        return logits


# ------------------------------------------------------- Log-Linear Mamba-2 arm
def loglinear_kernel(q, k, v, g, lam):
    """Upstream WEAK Log-Linear, Mamba-2 structure.  Inputs B,T,H,D; g B,T,H;
    lam B,T,H,L.  Pads the length to a multiple of 64 (causal, so harmless);
    unlike upstream_adapter.log_linear it never touches the feature width,
    which that adapter crops to 16 for the width-16 recall models."""
    import upstream_adapter as ua
    length, dtype = v.shape[1], v.dtype
    v = v.to(q.dtype)
    pad = (-length) % 64
    lam = lam[..., :((length + pad - 1).bit_length() + 1)]
    q, k, v = (F.pad(x, (0, 0, 0, 0, 0, pad)) for x in (q, k, v))
    g, lam = F.pad(g, (0, 0, 0, pad)), F.pad(lam, (0, 0, 0, 0, 0, pad))
    out = ua.hattention_kernel(q=q.contiguous(), k=k.contiguous(), v=v.contiguous(), b=None,
                               g=g.contiguous().float(), l=lam.to(q.dtype).contiguous(), scale=None,
                               head_first=False, level_base=2, htype=ua.HType.WEAK,
                               hstruct=ua.HStruct.MAMBA2, use_qk_l2norm_in_kernel=False)
    return out[:, :length].to(dtype).contiguous()


class LogLinearMamba2(nn.Module):
    """The plain baseline's Mamba-2 layer (same projections, conv, gate, norm,
    skip) with the SSD scan replaced by the Log-Linear kernel, plus the
    per-level lambda projection of upstream_adapter.JointLogLinearMixer."""

    def __init__(self, cfg, index):
        super().__init__()
        from zoology.mixers.mamba2 import Mamba2
        self.m = Mamba2(d_model=cfg.width, d_state=cfg.head_dim, d_conv=4, expand=2,
                        headdim=cfg.head_dim, ngroups=1, chunk_size=64,
                        use_mem_eff_path=False, layer_idx=index)
        m = self.m
        assert m.in_proj.out_features == 2 * m.d_ssm + 2 * m.ngroups * m.d_state + m.nheads
        self.levels = (cfg.length - 1).bit_length() + 1
        self.l_proj = nn.Linear(cfg.width, m.nheads * self.levels, bias=False)
        self.L = nn.Parameter(torch.ones(m.nheads, self.levels))

    def forward(self, u):
        from causal_conv1d import causal_conv1d_fn
        from einops import rearrange
        m, (b, t, _) = self.m, u.shape
        z, xbc, dt = torch.split(m.in_proj(u), [m.d_ssm, m.d_ssm + 2 * m.ngroups * m.d_state, m.nheads], dim=-1)
        xbc = causal_conv1d_fn(xbc.transpose(1, 2).contiguous(), rearrange(m.conv1d.weight, 'd 1 w -> d w'),
                               bias=m.conv1d.bias, activation=m.activation).transpose(1, 2)
        x, k, q = torch.split(xbc, [m.d_ssm, m.ngroups * m.d_state, m.ngroups * m.d_state], dim=-1)
        x = x.reshape(b, t, m.nheads, m.headdim)
        k, q = (a.reshape(b, t, 1, m.d_state).expand(b, t, m.nheads, m.d_state) for a in (k, q))
        dt = F.softplus(dt + m.dt_bias)
        g = -m.A_log.float().exp() * dt
        lam = F.softplus(self.l_proj(u).reshape(b, t, m.nheads, self.levels) * self.L)
        y = loglinear_kernel(q, k, x * dt[..., None], g, lam)
        y = y + x * m.D[None, None, :, None]
        return m.out_proj(m.norm(y.reshape(b, t, -1), z))

    def get_auxiliary_loss(self):
        return 0.


def loglinear_check(device):
    """Kernel against the dense reference (zoo_log_linear) at the arm's head shape."""
    from zoo_log_linear import log_linear as dense
    torch.manual_seed(0)
    b, t, h, dk = 1, 256, 4, 64
    q, k, v = (torch.randn(b, t, h, dk, device=device) * .3 for _ in range(3))
    g = -torch.rand(b, t, h, device=device) * .1
    lam = torch.rand(b, t, h, (t - 1).bit_length() + 1, device=device)
    ref = dense(q, k, v, g, lam)
    out = loglinear_kernel(q.bfloat16(), k.bfloat16(), v.bfloat16(), g, lam).float()
    return ((out - ref).norm() / ref.norm()).item()


# ------------------------------------------------------------------ building
def build(arm, length):
    cfg_dict = campaign_config(arm, length)
    notes = []
    if arm.startswith('gdn-smat'):
        import pg19_optimized_kernels
        pg19_optimized_kernels.enable()
        from gdn_smat_transport_scale import ScaleConfig, ScaleLM
        notes.append('tiled_fp32_v1 kernels enabled (campaign optimized continuation)')
    elif arm.startswith('mamba2-smat'):
        from mamba_smat_scale import ScaleConfig, ScaleLM
    elif arm in ('gdn', 'mamba2', 'mamba2-clean', 'mamba2-fused'):
        from pg19_baselines import ScaleConfig, ScaleLM
    elif arm == 'softmax':
        from pg19_baselines import ScaleConfig
        cfg_dict['family'] = 'softmax'
        ScaleLM = SoftmaxLM
    elif arm == 'loglinear-mamba2':
        from pg19_baselines import ScaleConfig, ScaleLM, Block
        cfg_dict['family'] = 'mamba2'
    cfg = ScaleConfig(**cfg_dict)
    torch.manual_seed(cfg.seed)
    model = ScaleLM(cfg)
    if arm == 'loglinear-mamba2':
        for i, blk in enumerate(model.blocks):
            blk.mixer = LogLinearMamba2(cfg, i)
    if arm == 'gdn-smat-d3-nog':
        for blk in model.blocks:
            blk.mixer.disable_g = True
        notes.append('memory branch disabled (disable_g); midpoint reset kept')
    if arm in ('mamba2-clean', 'mamba2-fused', 'loglinear-mamba2'):
        torch.autograd.set_detect_anomaly = lambda *a, **k: None
        notes.append('torch.autograd.set_detect_anomaly suppressed')
    if arm == 'mamba2-fused':
        # zoology's vendored fused kernel calls causal_conv1d with a pre-1.6 signature;
        # mamba_ssm 2.3.2's function has the same arguments and matches the installed one.
        import zoology.mixers.mamba2 as zoo_mamba2
        from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
        zoo_mamba2.mamba_split_conv1d_scan_combined = mamba_split_conv1d_scan_combined
        for blk in model.blocks:
            blk.mixer.mixer.use_mem_eff_path = True
        notes.append('fused path (use_mem_eff_path=True) with mamba_ssm 2.3.2 mamba_split_conv1d_scan_combined')
    return model.cuda(), cfg, notes


# ------------------------------------------------------------------ measuring
def cuda_times(fn, warmup, steps):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(steps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def summary(times):
    return dict(median_ms=float(np.median(times)), p25_ms=float(np.percentile(times, 25)),
                p75_ms=float(np.percentile(times, 75)), min_ms=float(min(times)), n=len(times))


def tensor_bytes(obj):
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(v) for v in obj)
    return 0


def run_train(model, cfg, args):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    tokens = torch.randint(0, cfg.vocab, (args.batch, cfg.length + 1), device='cuda')

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, aux = model.loss(tokens[:, :-1], tokens[:, 1:])
        (loss + aux).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

    times = cuda_times(step, args.warmup, args.steps)
    s = summary(times)
    return dict(**s, tokens_per_second=args.batch * cfg.length / (s['median_ms'] / 1e3),
                peak_gb=torch.cuda.max_memory_allocated() / 1e9,
                anomaly_mode=torch.is_anomaly_enabled())


@torch.inference_mode()
def run_prefill(model, cfg, args):
    model.eval()
    tokens = torch.randint(0, cfg.vocab, (args.batch, cfg.length), device='cuda')

    def fwd():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            hidden, _ = model.hidden(tokens)
            F.linear(hidden[:, -1], model.embedding.weight)

    times = cuda_times(fwd, args.warmup, args.steps)
    s = summary(times)
    return dict(**s, tokens_per_second=args.batch * cfg.length / (s['median_ms'] / 1e3),
                peak_gb=torch.cuda.max_memory_allocated() / 1e9)


def full_window_logits(model, rows):
    tokens = torch.full((len(rows), model.cfg.length), model.cfg.vocab - 1, device='cuda', dtype=torch.long)
    for i, row in enumerate(rows):
        tokens[i, :len(row)] = torch.as_tensor(row, device='cuda')
    hidden, _ = model.hidden(tokens)
    last = torch.tensor([len(r) - 1 for r in rows], device='cuda')
    return F.linear(hidden[torch.arange(len(rows), device='cuda'), last], model.embedding.weight).float()


def run_decode(arm, model, cfg, args):
    """Latency per generated token at several prompt lengths inside the window.
    SMAT prompts must pass the trained midpoint (RecurrentDecoder's contract)."""
    model.eval()
    T, steps, warm = cfg.length, args.decode_steps, 8
    prompts = sorted({T // 2 + 256, (3 * T) // 4, T - steps - warm - 8})
    if arm == 'softmax':
        make = lambda: KVDecoder(model, args.batch, T)
    elif arm in RECURRENT_DECODE:
        from pg19_recurrent import RecurrentDecoder
        make = lambda: RecurrentDecoder(model)
    else:
        return [dict(status='not_measured', reason='no incremental decoder for this arm')]
    g = torch.Generator().manual_seed(0)
    rows = []
    for i, P in enumerate(prompts):
        prompt = torch.randint(0, cfg.vocab - 1, (args.batch, P), generator=g)
        dec = make()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            logits = (dec.prefill(prompt.cuda()) if arm == 'softmax'
                      else dec.prefill([r.tolist() for r in prompt]))
            check = None
            if i == 0:                       # cached vs full window, bf16
                seq = [r.tolist() for r in prompt]
                kls, agree = [], 0
                for _ in range(3):
                    ref = full_window_logits(model, seq)
                    kl = (ref.softmax(-1) * (ref.log_softmax(-1) - logits.log_softmax(-1))).sum(-1)
                    kls.append(kl.max().item())
                    agree += (ref.argmax(-1) == logits.argmax(-1)).sum().item()
                    nxt = ref.argmax(-1)
                    for r, tok in zip(seq, nxt.tolist()):
                        r.append(tok)
                    logits = dec.step(nxt)
                check = dict(max_kl=max(kls), top1_agree=agree / (3 * args.batch))
            cache_bytes = tensor_bytes(getattr(dec, 'caches', None))
            nxt = [logits.argmax(-1)]

            def one():
                nxt[0] = dec.step(nxt[0]).argmax(-1)

            times = cuda_times(one, warm, steps)
        s = summary(times)
        rows.append(dict(prompt_tokens=P, **s, tokens_per_second=args.batch / (s['median_ms'] / 1e3),
                         cache_mb=cache_bytes / 1e6, peak_gb=torch.cuda.max_memory_allocated() / 1e9,
                         consistency=check))
        del dec
        torch.cuda.empty_cache()
    return rows


def provenance():
    import fla
    def version(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, text=True).strip()
    except Exception:
        commit = None
    return dict(torch=torch.__version__, fla=fla.__version__, triton=version('triton'),
                mamba_ssm=version('mamba_ssm'), causal_conv1d=version('causal_conv1d'),
                gpu=torch.cuda.get_device_name(), capability='sm_%d%d' % torch.cuda.get_device_capability(),
                git_commit=commit, write_hash_backend=os.environ.get('SMAT_WRITE_HASH_BACKEND'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=ARMS, required=True)
    ap.add_argument('--mode', choices=('train', 'prefill', 'decode'), required=True)
    ap.add_argument('--length', type=int, default=16384)
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--steps', type=int, default=15)
    ap.add_argument('--decode-steps', type=int, default=64)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    row = dict(arm=args.arm, mode=args.mode, length=args.length, batch=args.batch,
               campaign_arm=AS_TRAINED[args.arm], time=time.strftime('%Y-%m-%dT%H:%M:%S'))
    started = time.perf_counter()
    try:
        model, cfg, notes = build(args.arm, args.length)
        row.update(params=sum(p.numel() for p in model.parameters()), notes=notes)
        if args.arm == 'loglinear-mamba2':
            row['loglinear_kernel_rel_err_vs_dense'] = loglinear_check('cuda')
        if args.mode == 'train':
            row.update(run_train(model, cfg, args))
        elif args.mode == 'prefill':
            row.update(run_prefill(model, cfg, args))
        else:
            row['decode'] = run_decode(args.arm, model, cfg, args)
        row['status'] = 'ok'
    except torch.OutOfMemoryError as e:
        row.update(status='oom', error=str(e).splitlines()[0])
    except Exception as e:                   # recorded, not hidden: the row says what failed
        row.update(status='error', error=f'{type(e).__name__}: {e}'[:500])
        import traceback
        traceback.print_exc()
    row.update(provenance(), wall_seconds=time.perf_counter() - started)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'a') as fh:
        fh.write(json.dumps(row) + '\n')
    print('ROW ' + json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
