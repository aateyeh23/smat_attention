"""GPU checks for causal fixed-geometry inference and exact optimizer resume."""
import argparse
import copy
from pathlib import Path
import tempfile
import torch
from gdn_smat_scale import ScaleConfig, ScaleLM, RMSNorm
from train_pg19_scale import train


def inference_check():
    torch.manual_seed(876)
    ref_norm, new_norm = RMSNorm(64).cuda(), RMSNorm(64, fused=True).cuda()
    x = torch.randn(2, 256, 64, device='cuda', requires_grad=True)
    expected, actual = ref_norm(x), new_norm(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    dy = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, (x, new_norm.weight), dy)
    ge = torch.autograd.grad(expected, (x, ref_norm.weight), dy)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, rtol=1e-4, atol=1e-4)
    print('FUSED_RMSNORM_PASSED', flush=True)
    for d in (1, 2, 3, 4):
        cfg = ScaleConfig(d=d, width=64, layers=2, ffn_width=128, vocab=257,
                          length=256, head_dim=16, read_backend='triton', fused_norm=True)
        model = ScaleLM(cfg).cuda().eval()
        tokens = torch.randint(cfg.vocab, (2, 256), device='cuda')
        with torch.no_grad():
            full, _ = model.hidden(tokens)
            for end in (64, 128, 129, 192, 256):
                expected = torch.nn.functional.linear(full[:, end-1], model.embedding.weight)
                actual = model.next_logits(tokens[:, :end])
                torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)
            generated = model.generate(tokens[:, :192], 2)
            assert generated.shape == (2, 194)
        print(f'CAUSAL_GENERATION_PASSED d={d}', flush=True)


def resume_check(data):
    with tempfile.TemporaryDirectory() as folder:
        cfg = argparse.Namespace(d=4, data=data, output=folder+'/whole', tokens=4*16384,
            batch=1, accumulation=2, lr=3e-4, warmup_tokens=32768,
            eval_every=1_000_000, eval_rows=1, checkpoint_every=1_000_000,
            log_every=1, max_hours=1, backend='triton', layers=2, width=64,
            ffn_width=128, head_dim=16, length=16384, loss_chunk=4096)
        # Small validation shard keeps this a correctness check, not a full evaluation.
        import json
        import shutil
        original = Path(data)
        sample = Path(folder)/'data'
        sample.mkdir()
        shutil.copyfile(original/'manifest.json', sample/'manifest.json')
        for name, rows in (('train', 8), ('val', 1)):
            with (original/(name+'.bin')).open('rb') as src:
                (sample/(name+'.bin')).write_bytes(src.read(rows*16385*2))
        cfg.data = str(sample)
        train(cfg)
        repeat = copy.copy(cfg)
        repeat.output = folder+'/repeat'
        train(repeat)
        resumed = copy.copy(cfg)
        resumed.output = folder+'/resumed'
        resumed.max_hours = 0
        assert train(resumed) is False
        partial = torch.load(Path(resumed.output)/'latest.pt', weights_only=False, map_location='cpu')
        restored = ScaleLM(ScaleConfig(**partial['config'])).cuda()
        restored.load_state_dict(partial['model'])
        restored_opt = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad],
                            lr=cfg.lr, betas=(0.9,0.95), weight_decay=0.1, fused=True)
        restored_opt.load_state_dict(partial['optimizer'])
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, partial['model'][name].cuda(), rtol=0, atol=0)
        for key, state in restored_opt.state_dict()['state'].items():
            for name, value in state.items():
                expected = partial['optimizer']['state'][key][name]
                torch.testing.assert_close(value, expected.to(value.device), rtol=0, atol=0)
        assert restored_opt.state_dict()['param_groups'] == partial['optimizer']['param_groups']
        print('CHECKPOINT_STATE_EXACT_ROUNDTRIP_PASSED', flush=True)
        del restored, restored_opt, partial
        resumed.max_hours = 1
        assert train(resumed) is True
        a = torch.load(Path(cfg.output)/'latest.pt', weights_only=False, map_location='cpu')
        b = torch.load(Path(resumed.output)/'latest.pt', weights_only=False, map_location='cpu')
        c = torch.load(Path(repeat.output)/'latest.pt', weights_only=False, map_location='cpu')
        assert a['cursor'] == b['cursor'] == 4 and a['step'] == b['step'] == 2
        # GPU scatter/embedding reductions are not bitwise deterministic. Compare
        # continuation drift against a second uninterrupted run, not exact bits.
        repeat_max = max((a['model'][n]-c['model'][n]).abs().max().item()
                         for n in a['model'] if a['model'][n].is_floating_point())
        resume_max = 0.0
        for name in a['model']:
            torch.testing.assert_close(a['model'][name], b['model'][name],
                                       rtol=3e-5, atol=max(3e-6, 3*repeat_max))
            if a['model'][name].is_floating_point():
                resume_max = max(resume_max, (a['model'][name]-b['model'][name]).abs().max().item())
        assert abs(a['validation']['loss']-b['validation']['loss']) < 1e-3
        print(f'CHECKPOINT_RESUME_PASSED repeat_max={repeat_max} resume_max={resume_max}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    inference_check()
    resume_check(args.data)
