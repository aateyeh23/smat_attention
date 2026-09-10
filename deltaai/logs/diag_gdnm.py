import os, sys, torch, importlib
cfgs = [("d1 h1 ev2", "1", "1", "2"), ("d3 h1 ev1", "3", "1", "1"), ("d3 h2 ev2", "3", "2", "2"), ("d3 h1 ev2", "3", "1", "2")]
for name, ds, h, ev in cfgs:
    os.environ.update(ZOO_DS=ds, ZOO_GDN_HEADS=h, ZOO_GDN_EXPANDV=ev)
    import zoo_smat_configs as c; importlib.reload(c)
    from zoology.model import LanguageModel
    try:
        m = LanguageModel(c.configs[0].model).cuda()
        for T in (64, 256):
            x = torch.randint(0, 8192, (4, T)).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16): y = m(x)
            y.float().mean().backward(); torch.cuda.synchronize()
        print(name, "OK", c.configs[0].model.name, flush=True)
    except Exception as e:
        print(name, "FAIL", type(e).__name__, str(e)[:120].replace("\n", " "), flush=True); sys.exit(0)
