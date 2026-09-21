"""Long-document pretraining corpus (public stand-in for Long-Data-Collections' pretrain split):
  arxiv  : RedPajama-1T arxiv shards (data.together.xyz)
  books  : PG-19 train (the Gutenberg part of RedPajama books)
  pile   : monology/pile-uncopyrighted (long documents only)
Keep only documents with >= SEQ (16384) GPT-2 tokens, cut them into contiguous SEQ-token chunks (no crossing of
document boundaries), write per-source uint16 memmaps of shape (n_chunks, SEQ) until each source's quota is met.
A fixed interleaved order (order.npy) defines the training sequence used by every model.  Held-out chunks come
from documents never used for training (val/*.bin).  Resumable per source."""
import os, sys, json, io, numpy as np, tiktoken, requests, zstandard as zstd, time
SEQ = int(os.environ.get("SEQ", 16384)); OUT = os.environ.get("OUT", os.path.join(os.environ.get("SMAT_WORK", "data"), "ldc"))
QUOTA = {"arxiv": int(2.2e8), "books": int(1.6e8), "pile": int(1.6e8)}      # train tokens per source (~540M total)
VAL_QUOTA = int(2.0e7)                                                       # held-out tokens per source
enc = tiktoken.get_encoding("gpt2")
os.makedirs(f"{OUT}/train", exist_ok=True); os.makedirs(f"{OUT}/val", exist_ok=True)


class Writer:
    def __init__(self, path, quota):
        self.n_max = quota // SEQ; self.path = path
        self.buf = np.memmap(path, dtype=np.uint16, mode="w+", shape=(self.n_max, SEQ)); self.n = 0
    def add(self, ids):
        k = min(len(ids) // SEQ, self.n_max - self.n)
        if k <= 0: return 0
        self.buf[self.n:self.n + k] = np.asarray(ids[:k * SEQ], dtype=np.uint16).reshape(k, SEQ); self.n += k
        return k
    @property
    def full(self): return self.n >= self.n_max
    def close(self): self.buf.flush(); json.dump({"n_chunks": self.n, "seq": SEQ}, open(self.path + ".json", "w"))


def docs_arxiv():
    urls = open(os.path.expanduser("~/.cache/rp_arxiv_urls.txt")).read().split() if os.path.exists(os.path.expanduser("~/.cache/rp_arxiv_urls.txt")) else None
    if urls is None:
        from huggingface_hub import hf_hub_download
        urls = open(hf_hub_download("togethercomputer/RedPajama-Data-1T", "urls/arxiv.txt", repo_type="dataset")).read().split()
    for u in urls:
        r = requests.get(u, stream=True, timeout=120); r.raise_for_status()
        for line in r.iter_lines():
            if line:
                try: yield json.loads(line)["text"]
                except Exception: continue


def docs_books():
    from datasets import load_dataset
    for ex in load_dataset("deepmind/pg19", split="train", streaming=True, trust_remote_code=True): yield ex["text"]


def docs_pile():
    from huggingface_hub import HfApi, hf_hub_download
    files = sorted(s.rfilename for s in HfApi().dataset_info("monology/pile-uncopyrighted").siblings if s.rfilename.startswith("train/") and s.rfilename.endswith(".zst"))
    for f in files:
        p = hf_hub_download("monology/pile-uncopyrighted", f, repo_type="dataset")
        with open(p, "rb") as fh:
            for line in io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(fh), encoding="utf-8"):
                try: yield json.loads(line)["text"]
                except Exception: continue


SRC = {"arxiv": docs_arxiv, "books": docs_books, "pile": docs_pile}
for name in sys.argv[1:] or SRC:
    if os.path.exists(f"{OUT}/train/{name}.bin.json"): print(f"{name}: done already", flush=True); continue
    tr, va = Writer(f"{OUT}/train/{name}.bin", QUOTA[name]), Writer(f"{OUT}/val/{name}.bin", VAL_QUOTA)
    seen = kept = 0; t0 = time.time()
    for text in SRC[name]():
        seen += 1
        if len(text) < 4 * SEQ: continue                                  # cheap length prefilter (chars)
        ids = enc.encode_ordinary(text)
        if len(ids) < SEQ: continue
        kept += 1
        if not va.full: va.add(ids)                                        # held-out docs first, disjoint from train
        else: tr.add(ids)
        if kept % 100 == 0: print(f"{name}: seen {seen} kept {kept} train {tr.n*SEQ/1e6:.0f}M val {va.n*SEQ/1e6:.0f}M ({(time.time()-t0)/60:.0f}m)", flush=True)
        if tr.full and va.full: break
    tr.close(); va.close(); print(f"DONE {name}: train {tr.n} chunks, val {va.n} chunks", flush=True)
# fixed interleaved order over the train chunks of all sources
metas = {n: json.load(open(f"{OUT}/train/{n}.bin.json")) for n in SRC if os.path.exists(f"{OUT}/train/{n}.bin.json")}
if len(metas) == len(SRC):
    rng = np.random.default_rng(0)
    order = np.concatenate([np.stack([np.full(m["n_chunks"], i), np.arange(m["n_chunks"])], 1) for i, (n, m) in enumerate(metas.items())])
    rng.shuffle(order); np.save(f"{OUT}/order.npy", order); print("order.npy:", order.shape, flush=True)
