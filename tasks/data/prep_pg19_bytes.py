"""PG-19 as raw UTF-8 bytes for the byte-level schedule study (tasks/window_schedules.py).

Reads the parquet mirror (emozilla/pg19) from the directory given as the only
argument (default $SMAT_PG19_BYTES, else data/pg19_bytes): train0.parquet (one
training shard) and test.parquet.
Writes train.bin and test.bin (uint8, books concatenated, separated by a 0 byte)
and test_offsets.npy (book start offsets, plus the end)."""
import os, sys
import numpy as np
import pyarrow.parquet as pq

root = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SMAT_PG19_BYTES", "data/pg19_bytes")
for split, src in (("test", "test.parquet"), ("train", "train0.parquet")):
    tab = pq.read_table(os.path.join(root, src), columns=["text"])
    offs, n = [0], 0
    with open(os.path.join(root, f"{split}.bin"), "wb") as f:
        for t in tab.column("text").to_pylist():
            b = t.encode("utf-8") + b"\x00"
            f.write(b); n += len(b); offs.append(n)
    if split == "test":
        np.save(os.path.join(root, "test_offsets.npy"), np.asarray(offs, dtype=np.int64))
    print(split, len(offs) - 1, "books", n / 1e6, "MB", flush=True)
