"""GPU checks of wrapper equivalence, recurrence, gradients, and state budget."""
import json
import time
from common import ROOT, STATE, atomic_json, bootstrap, make_config, verify_sources


def main():
    bootstrap()
    digest = verify_sources()
    import torch
    from torch.nn import functional as F
    from model import StateMatchedMamba2, state_elements
    from zoo_mamba_four_reads import MqarMambaFourReads
    from zoology.model import LanguageModel
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    torch.manual_seed(123)
    native = MqarMambaFourReads(16, d=1).cuda()
    torch.manual_seed(123)
    wrapped = StateMatchedMamba2(16, d_state=16).cuda()
    assert native.state_dict().keys() == wrapped.state_dict().keys()
    for key, value in native.state_dict().items():
        assert torch.equal(value, wrapped.state_dict()[key]), key
    x = torch.randn(2, 64, 16, device='cuda')
    torch.testing.assert_close(native(x), wrapped(x), rtol=0, atol=0)
    print('PASS native-wrapper identity', flush=True)

    smat = MqarMambaFourReads(16, d=3).cuda()
    enlarged = StateMatchedMamba2(16, d_state=STATE).cuda()
    local = sum(state_elements(smat).values())
    matched = sum(state_elements(enlarged).values())
    budgets = []
    for length in [64, 128, 256]:
        smat._spec(length, torch.device('cuda'))
        cas = list(smat.ca[str(length)].values())
        types = sum(ca.h*ca.W_read.shape[1] for ca in cas)
        profiles = sum(ca.h*ca.N0 for ca in cas)
        summary_elements = types*smat.N*smat.p
        profile_elements = profiles*smat.N*smat.p
        # The write-hash convolution is only needed while ingesting the prefix.
        prefix_elements = local+profile_elements+16*4
        recent_elements = local+summary_elements
        assert prefix_elements <= recent_elements <= matched
        budgets.append(dict(length=length, types_per_head=cas[0].W_read.shape[1],
            profiles_per_head=cas[0].N0, smat_recent_elements_per_layer=recent_elements,
            smat_prefix_elements_per_layer=prefix_elements,
            native_elements_per_layer=matched,
            smat_recent_bytes_two_layers=recent_elements*2*4,
            native_bytes_two_layers=matched*2*4,
            native_to_smat_ratio=matched/recent_elements))
    assert 1 <= budgets[-1]['native_to_smat_ratio'] < 1.01

    # Compare the enlarged parallel SSD against an explicit recurrent evaluation.
    m = enlarged.mixer
    x = torch.randn(2, 64, 16, device='cuda')
    with torch.no_grad():
        z, xbc, dt = m.in_proj(x).split([32, 32+2*STATE, 2], dim=-1)
        xbc = F.silu(m.conv1d(xbc.transpose(1, 2))[..., :64].transpose(1, 2))
        vals, keys, queries = xbc.split([32, STATE, STATE], dim=-1)
        vals = vals.reshape(2, 64, 2, 16)
        delta = F.softplus(dt + m.dt_bias)
        decay = torch.exp(-m.A_log.exp()[None, None, :]*delta)
        state = torch.zeros(2, 2, 16, STATE, device='cuda')
        outputs = []
        for t in range(64):
            state = state*decay[:, t, :, None, None] + (
                delta[:, t, :, None, None]*vals[:, t, :, :, None]*keys[:, t, None, None, :])
            y = (state*queries[:, t, None, None, :]).sum(-1)
            outputs.append(y+vals[:, t]*m.D[None, :, None])
        reference = m.out_proj(m.norm(torch.stack(outputs, 1).reshape(2,64,32), z))
        actual = enlarged(x)
        torch.testing.assert_close(actual, reference, rtol=3e-4, atol=3e-5)
        altered = x.clone(); altered[:, 32:] = torch.randn_like(altered[:, 32:])
        torch.testing.assert_close(actual[:, :32], enlarged(altered)[:, :32], rtol=0, atol=0)
    print('PASS enlarged-state recurrence and causality', flush=True)

    config = make_config(123, ROOT/'validation_scratch')
    model = LanguageModel(config.model).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.1)
    timing = []
    for length in [64, 128, 256]:
        for repeat in range(2):
            inputs = torch.randint(0, 8192, (256, length), device='cuda')
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); start = time.monotonic()
            out = model(inputs)
            loss = F.cross_entropy(out[:, -1], inputs[:, 0])
            assert torch.isfinite(loss)
            loss.backward()
            for name, p in model.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(), name
            optimizer.step()
            torch.cuda.synchronize()
            timing.append(dict(length=length, repeat=repeat, seconds=time.monotonic()-start,
                               loss=loss.item()))
            print('PASS full-batch step '+json.dumps(timing[-1]), flush=True)
    # Serialization checks model and optimizer tensors without changing training data.
    checkpoint = ROOT/'validation_scratch.pt'
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), checkpoint)
    restored = torch.load(checkpoint, weights_only=False)
    model.load_state_dict(restored['model']); optimizer.load_state_dict(restored['optimizer'])
    checkpoint.unlink()
    atomic_json(ROOT/'budget.json', dict(state_dtype='float32', layers=2,
        state_dim=STATE, width=16, head_dim=16, heads=2, budgets=budgets,
        native_parameters=sum(p.numel() for p in model.parameters()),
        definition='Retained recurrent/conv state plus frozen type summaries after prefix; excludes weights and temporary training activations. Profile states can be released after summary construction.'))
    atomic_json(ROOT/'validated.json', dict(passed=True, source_manifest_sha256=digest,
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, timings=timing,
        peak_training_bytes=torch.cuda.max_memory_allocated()))
    print('VALIDATION PASSED', flush=True)


if __name__ == '__main__':
    main()
