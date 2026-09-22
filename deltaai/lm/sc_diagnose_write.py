"""Read-only checkpoint diagnostics and inference interventions for selective copying."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import types

import torch

parser = argparse.ArgumentParser()
parser.add_argument("checkpoints", nargs="+")
parser.add_argument("--out", required=True)
cli = parser.parse_args()
outdir = Path(cli.out); outdir.mkdir(parents=True, exist_ok=True)
source = Path(__file__).with_name("sc_train.py").read_text().split("model = LM().to(dev)")[0]
results = []
for path in cli.checkpoints:
    for attempt in range(4):
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            break
        except (RuntimeError, EOFError):
            if attempt == 3:
                raise
            time.sleep(1)
    args = ck["args"]
    snapshot = {k: ck[k] for k in ("model", "step", "args")}
    torch.save(snapshot, outdir / (Path(path).stem + ".pt"))
    sys.argv = ["sc_train.py"] + [f"--{k}={v}" for k, v in args.items() if v is not None]
    ns = {}; exec(compile(source, "sc_defs", "exec"), ns)
    model = ns["LM"]().cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(torch.zeros(1, args["seq_len"], dtype=torch.long, device="cuda"))
    model.load_state_dict(ck["model"]); model.eval()
    mixers = [block.mixer for block in model.blocks]
    for m in mixers:
        for mods in m.ca.values():
            for ca in mods.values():
                ca.anneal = 1.0
    generator = torch.Generator().manual_seed(1234)
    xe, ye = ns["make_batch"](256, generator)
    k, n = args["n_copy"], args["seq_len"] // 2
    original_weights = [m._g_write_weights for m in mixers]
    current = {}

    def run(variant):
        model.load_state_dict(ck["model"])
        for layer, m in enumerate(mixers):
            m.g_decay = variant != "no_g_decay"
            m._g_write_weights = original_weights[layer]
            if variant == "g_off":
                with torch.no_grad():
                    m.lam_w.weight.zero_(); m.lam_w.bias.zero_(); m.alpha.fill_(-100)
            if variant == "read_0.1":
                with torch.no_grad():
                    m.lam_w.weight.zero_(); m.lam_w.bias.zero_(); m.alpha.fill_(math.log(0.1 / 0.9))
            if variant in ("no_noise_writes", "no_noise_no_decay"):
                if variant == "no_noise_no_decay":
                    m.g_decay = False
                def selected(self, u, dts, original=original_weights[layer]):
                    return original(u, dts) * (current["tokens"] < args["n_vocab"]).unsqueeze(-1)
                m._g_write_weights = types.MethodType(selected, m)
        stats = {"correct": 0, "total": 0, "first_correct": 0, "first_count": 0,
                 "second_correct": 0, "second_count": 0}
        measurements = [{"read": [], "log10_retention": [], "visibility": []} for _ in mixers]
        hooks = []
        if variant == "baseline":
            for layer, m in enumerate(mixers):
                def inspect(module, inputs, layer=layer):
                    u = inputs[0]
                    with torch.no_grad(), torch.autocast("cuda", enabled=False):
                        gate = torch.sigmoid(module.alpha[:, :1].T + module.lam_w(u.float())[:, -k:])
                        measurements[layer]["read"].append(gate.flatten().cpu())
                    # Match the projection precision used in the actual forward.
                    dt = module.mixer.in_proj(u)[..., -module.h:]
                    dts = torch.nn.functional.softplus(dt.float() + module.mixer.dt_bias.float())
                    la = (-module.mixer.A_log.float().exp()[None, :, None] * dts.transpose(1, 2)).cumsum(-1)
                    positions = current["positions"]
                    source = la.gather(2, positions[:, None, :].expand(-1, module.h, -1))
                    log_survival = (la[:, :, -k:] - source) / math.log(10)
                    mask = (positions < n)[:, None, :].expand_as(log_survival)
                    measurements[layer]["log10_retention"].append(log_survival[mask].cpu())
                hooks.append(m.register_forward_pre_hook(inspect))
                for mods in m.ca.values():
                    for ca in mods.values():
                        def route(module, inputs, output, layer=layer):
                            h, _, _, _, boundary = inputs[:5]
                            b = h.shape[0]; heads = module.h
                            key_cell = module.last_hard_k.view(b, heads, boundary)
                            positions = current["positions"]
                            target = key_cell.gather(2, positions.clamp_max(boundary - 1)[:, None, :].expand(-1, heads, -1))
                            query = module._sparse[0][:, -k:, 0].view(b, heads, k)
                            if module.mode == "point":
                                visible = target == query
                            else:
                                direction = torch.einsum("blm,hem->bhle", h[:, -k:], module.Wd.to(h.dtype)).argmax(-1)
                                visible = module.coset_of[direction, target] == module.coset_of[direction, query]
                            measurements[layer]["visibility"].append(visible.any(1)[positions < boundary].float().cpu())
                        hooks.append(ca.register_forward_hook(route))
        with torch.no_grad():
            for start in range(0, 256, 64):
                x = xe[start:start + 64].cuda(); y = ye[start:start + 64, -k:].cuda()
                positions = (x[:, :-k] < args["n_vocab"]).nonzero()[:, 1].reshape(x.shape[0], k)
                current.update(tokens=x, positions=positions)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)[:, -k:]
                correct = logits.argmax(-1) == y
                stats["correct"] += correct.sum().item(); stats["total"] += correct.numel()
                for half, mask in (("first", positions < n), ("second", positions >= n)):
                    stats[half + "_correct"] += (correct & mask).sum().item()
                    stats[half + "_count"] += mask.sum().item()
        for hook in hooks:
            hook.remove()
        result = {"variant": variant, "accuracy": stats["correct"] / stats["total"],
                  "first": stats["first_correct"] / stats["first_count"],
                  "second": stats["second_correct"] / stats["second_count"]}
        if variant == "baseline":
            result["layers"] = []
            for layer in measurements:
                result["layers"].append({key: {"mean": float(torch.cat(values).float().mean()),
                    "quantiles": torch.cat(values).float().quantile(torch.tensor([0., .25, .5, .75, 1.])).tolist()}
                    for key, values in layer.items()})
        print(json.dumps({"checkpoint": Path(path).stem, "step": ck["step"], **result}), flush=True)
        return result

    result = {"checkpoint": path, "step": ck["step"], "variants": [run(v) for v in
        ("baseline", "g_off", "no_noise_writes", "no_g_decay", "no_noise_no_decay", "read_0.1")]}
    results.append(result)
    (outdir / "diagnosis.json").write_text(json.dumps(results, indent=2))
    del model, mixers, ck, snapshot
    torch.cuda.empty_cache()
