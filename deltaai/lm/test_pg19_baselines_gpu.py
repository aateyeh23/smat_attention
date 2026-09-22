"""GPU checks for full-sequence plain GDN and Mamba-2 baselines."""
import os
import torch
from torch.nn import functional as F
from pg19_baselines import ScaleConfig, ScaleLM


def check():
    family = os.environ['PG19_MEMORY_VARIANT'].removesuffix('_baseline')
    torch.set_num_threads(4)
    torch.manual_seed(731)
    cfg = ScaleConfig(d=1, family=family, width=64, layers=2, ffn_width=128,
        vocab=257, length=256, heads=2 if family=='gdn' else 8, head_dim=16,
        fused_norm=True, activation_checkpointing=False)
    model = ScaleLM(cfg).cuda()
    assert not any('W_read' in n or 'kconv' in n or 'lam_w' in n for n,_ in model.named_parameters())
    tokens = torch.randint(cfg.vocab, (1,256), device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss, aux = model.loss(tokens, tokens.roll(-1,1))
    assert aux.item() == 0
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.isfinite().all(), name
    print('PASS BASELINE FULL-MODEL BACKWARD', flush=True)
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        expected = model.next_logits(tokens[:,:192])
        altered = tokens.clone()
        altered[:,192:] = torch.randint(cfg.vocab, (1,64), device='cuda')
        h,_ = model.hidden(altered)
        actual = F.linear(h[:,191], model.embedding.weight).float()
        torch.testing.assert_close(actual, expected, atol=.02, rtol=.02)
        if family == 'gdn':
            mixer = model.blocks[0].mixer
            assert mixer.d == 1 and not mixer.reset and len(mixer.ca) == 0
            x = torch.randn(1,256,cfg.width,device='cuda')
            reference = mixer.gdn(x)[0]
            torch.testing.assert_close(mixer(x), reference, atol=.02, rtol=.02)
    print('PASS BASELINE CAUSAL FULL-SEQUENCE INFERENCE', flush=True)


if __name__ == '__main__':
    check()
