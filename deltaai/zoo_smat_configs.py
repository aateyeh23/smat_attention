"""SMAT arms through Zoology's pipeline (same data/protocol as zoo_reference_configs):
Mamba-2 + G at d=2 and d=3 with the mask bank, and d=1 (== Mamba-2 via our wrapper,
a consistency check).  Width from ZOO_DM."""
import os, uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, DataConfig, LoggerConfig
from zoology.data.multiquery_ar import MQARConfig
import importlib.util
_ref = importlib.util.module_from_spec(importlib.util.spec_from_file_location("ref", os.path.join(os.path.dirname(os.path.abspath(__file__)), "zoo_reference_configs.py")))
_ref.__spec__.loader.exec_module(_ref)
DM, EPOCHS, SMALL, data, VOCAB_SIZE = _ref.DM, _ref.EPOCHS, _ref.SMALL, _ref.data, _ref.VOCAB_SIZE
sweep_name = f"smat-zoo-smat-dm{DM}-" + uuid.uuid4().hex[:6]
DS = [int(x) for x in os.environ.get("ZOO_DS", "2,3").split(",")]
RESET = os.environ.get("ZOO_RESET", "1") == "1"          # ZOO_RESET=0 -> hybrid (no boundary reset)
SHARED = os.environ.get("ZOO_SHARED", "0") == "1"        # ZOO_SHARED=1 -> multi-resolution form (shared B/C/x, lambda-gated G)
MR3 = os.environ.get("ZOO_MR3", "0") == "1"              # ZOO_MR3=1 -> multi-resolution rev 3 (SmatMamba2MR: bank coverage, write gate, id kernel, per-type lambda)
HEADDIM = int(os.environ["ZOO_HEADDIM"]) if os.environ.get("ZOO_HEADDIM") else None
GDECAY = os.environ.get("ZOO_GDECAY", "0") == "1"
GKERNEL = os.environ.get("ZOO_GKERNEL", "id")
LAMACT = os.environ.get("ZOO_LAMACT", "softplus")
DSTATE = int(os.environ.get("ZOO_DSTATE", "128"))
POOL = os.environ.get("ZOO_POOL", "sum")
READ = os.environ.get("ZOO_READ", "type")
BASE = os.environ.get("ZOO_BASE", "mamba2")     # mamba2 | gdn
GDNHEADS = int(os.environ["ZOO_GDN_HEADS"]) if os.environ.get("ZOO_GDN_HEADS") else None   # None -> 2*d_model/headdim
GDNEXPV = float(os.environ.get("ZOO_GDN_EXPANDV", "1"))
RESETMR = os.environ.get("ZOO_RESET_MR", "0") == "1"    # paper mask: recurrence reset at the boundary
GFLOOR = os.environ.get("ZOO_GFLOOR") or None
HMODE = os.environ.get("ZOO_HMODE", "point"); HSHIFT = int(os.environ.get("ZOO_HSHIFT", "1")); HCONV = os.environ.get("ZOO_HCONV", "0") == "1"
HCODIM = ([int(x) for x in os.environ["ZOO_HCODIM"].split(",")] if "," in os.environ["ZOO_HCODIM"] else int(os.environ["ZOO_HCODIM"])) if os.environ.get("ZOO_HCODIM") else None
HSRC = os.environ.get("ZOO_HSRC", "hidden")
HFA = int(os.environ.get("ZOO_HFREEZE_AFTER", "0")); HLR = float(os.environ.get("ZOO_HLR", "1")); BALG = os.environ.get("ZOO_BAL_GATED", "0") == "1"
HFREEZE = os.environ.get("ZOO_HFREEZE", "0") == "1"; ANNEAL = int(os.environ.get("ZOO_ANNEAL", "0")); BALANCE = float(os.environ.get("ZOO_BALANCE", "0.01"))            # "q" or a number: weight for the zeros of C in G
DLAYERS = [int(x) for x in os.environ["ZOO_D_LAYERS"].split("-")] if os.environ.get("ZOO_D_LAYERS") else None   # e.g. "1-2"
import zoology.mixers.mamba2 as _zm2, zoo_smat_mixer as _zsm
_zm2.Mamba2Block = _zsm.SmatMamba2Block
_zsm.patch_token_embeddings()        # SMAT arms use the reference arm's block + init (see mixer file)
models = []
for d in DS:
    if MR3 and BASE == "gdn":
        kw = {"d": d, "headdim": HEADDIM or 16, "expand_v": GDNEXPV, "n_heads": GDNHEADS, "lam_act": LAMACT, "read": READ}
        nm = f"smatgdn_d{d}_hd{HEADDIM or 16}" + (f"_h{GDNHEADS}" if GDNHEADS else "") + (f"_ev{GDNEXPV:g}" if GDNEXPV != 1 else "") + ("_sig" if LAMACT == "sigmoid" else "") + ("_cread" if READ == "content" else "") + (f"_chash{HMODE}" + (("_c" + (",".join(map(str, HCODIM)) if isinstance(HCODIM, list) else str(HCODIM))) if HCODIM else "") + (f"_{HSRC}" if HSRC != "hidden" else "") + (f"_fa{HFA}" if HFA else "") + (f"_hlr{HLR:g}" if HLR != 1 else "") + ("_balg" if BALG else "") + ("_hconv" if HCONV else f"_sh{HSHIFT}") + ("_frz" if HFREEZE else "") + (f"_an{ANNEAL}" if ANNEAL else "") if READ == "chash" else "") + "_m2blk"
        models.append(ModelConfig(block_type="Mamba2Block", d_model=DM, n_layers=2,
                                  sequence_mixer=dict(name="zoo_smat_mixer.SmatGDN", kwargs=kw),
                                  max_position_embeddings=0, vocab_size=VOCAB_SIZE, name=nm))
        continue
    if MR3:
        kw = {"d": d, "d_state": DSTATE, "headdim": HEADDIM, "mask_bank": True, "g_decay": GDECAY, "kernel": GKERNEL, "lam_act": LAMACT, "pool": POOL, "read": READ, "reset": RESETMR, "d_layers": DLAYERS, "g_floor": GFLOOR, "hash_mode": HMODE, "hash_shift": HSHIFT, "hash_conv": HCONV, "hash_freeze": HFREEZE, "anneal_steps": ANNEAL, "balance_coef": BALANCE, "hash_codim": HCODIM, "hash_src": HSRC, "hash_freeze_after": HFA, "hash_lr_scale": HLR, "balance_gated": BALG}
        nm = (f"smat_d{os.environ['ZOO_D_LAYERS']}L" if DLAYERS else f"smat_d{d}") + "_mr3" + (f"_hd{HEADDIM}" if HEADDIM else "") + (f"_ds{DSTATE}" if DSTATE != 128 else "") + ("_decay" if GDECAY else "") + ("" if GKERNEL == "id" else f"_{GKERNEL}") + ("_sig" if LAMACT == "sigmoid" else "") + ("_delta" if POOL == "delta" else "") + ("_cread" if READ == "content" else "") + (f"_chash{HMODE}" + (("_c" + (",".join(map(str, HCODIM)) if isinstance(HCODIM, list) else str(HCODIM))) if HCODIM else "") + (f"_{HSRC}" if HSRC != "hidden" else "") + (f"_fa{HFA}" if HFA else "") + (f"_hlr{HLR:g}" if HLR != 1 else "") + ("_balg" if BALG else "") + ("_hconv" if HCONV else f"_sh{HSHIFT}") + ("_frz" if HFREEZE else "") + (f"_an{ANNEAL}" if ANNEAL else "") if READ == "chash" else "") + "_m2blk"
        models.append(ModelConfig(block_type="Mamba2Block", d_model=DM, n_layers=2,
                                  sequence_mixer=dict(name="zoo_smat_mixer.SmatMamba2MR", kwargs=kw),
                                  max_position_embeddings=0, vocab_size=VOCAB_SIZE, name=nm))
        continue
    models.append(ModelConfig(block_type="Mamba2Block", d_model=DM, n_layers=2,
                              sequence_mixer=dict(name="zoo_smat_mixer.SmatMamba2Shared" if SHARED else "zoo_smat_mixer.SmatMamba2",
                                                  kwargs={"d": d, "n_heads": 1, "r": 64, "d_state": 128, "mask_bank": True,
                                                          "reset": RESET}),
                              max_position_embeddings=0, vocab_size=VOCAB_SIZE,
                              name=f"smat_d{d}" + ("_shared" if SHARED else "") + ("" if RESET else "_hybrid") + "_m2blk"))
configs = []
for model in models:
    _lrs = [1e-3] if SMALL else ([float(x) for x in os.environ["ZOO_LRS"].split(",")] if os.environ.get("ZOO_LRS") else np.logspace(-3, -1.5, 4))
    _tag = os.environ.get("ZOO_TAG", "")
    _seed = int(os.environ.get("ZOO_SEED", "123"))
    _tag = _tag + (f"_s{_seed}" if _seed != 123 else "")
    for lr in _lrs:
        configs.append(TrainConfig(model=model, data=data, learning_rate=float(lr), max_epochs=EPOCHS, seed=_seed,
                                   logger=LoggerConfig(project_name="smat-zoology"), slice_keys=["num_kv_pairs"],
                                   sweep_id=sweep_name, run_id=f"{model.name}{_tag}-dm{DM}-lr{lr:.1e}"))
