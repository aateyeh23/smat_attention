"""PG-19 -> GPT-2 BPE token memmaps.  train.bin: first N_TRAIN_TOKENS of the train split (book order);
val.bin / test.bin: the PG-19 validation (50 books) and test (100 books) splits, whole.  Documents separated by <|endoftext|>."""
import os, sys, numpy as np, tiktoken
from datasets import load_dataset
OUT = os.path.join(os.environ.get("SMAT_WORK", "data"), "pg19")
N_TRAIN = int(float(os.environ.get("N_TRAIN_TOKENS", "3.2e8")))
N_VAL = int(float(os.environ.get("N_VAL_TOKENS", "3.9e7")))
enc = tiktoken.get_encoding("gpt2"); EOT = enc.eot_token

def dump(split_iter, path, n_target):
    buf = np.memmap(path, dtype=np.uint16, mode="w+", shape=(n_target,))
    n, docs = 0, 0
    for ex in split_iter:
        ids = enc.encode_ordinary(ex["text"]) + [EOT]
        take = min(len(ids), n_target - n)
        buf[n:n + take] = np.asarray(ids[:take], dtype=np.uint16); n += take; docs += 1
        if docs % 200 == 0: print(f"{path}: {docs} docs, {n/1e6:.1f}M tokens", flush=True)
        if n >= n_target: break
    buf.flush(); del buf; os.truncate(path, n * 2)                       # file length == filled tokens (uint16)
    print(f"DONE {path}: {docs} docs, {n} tokens", flush=True)
    return n

os.makedirs(OUT, exist_ok=True)
ds = load_dataset("deepmind/pg19", streaming=True, trust_remote_code=True)
for split, fn, n_t in (("validation", "val", N_VAL), ("test", "test", N_VAL), ("train", "train", N_TRAIN)):
    if split in sys.argv[1:] or not sys.argv[1:]:
        dump(ds[split], f"{OUT}/{fn}.bin", n_t)
