"""Tokenize an OpenWebText-style corpus to GPT-2 BPE uint16 .bin (nanoGPT format).
Bounded to a target token budget so prep stays fast; streams to avoid full download."""
import os, sys, numpy as np, tiktoken
from datasets import load_dataset

DS = os.environ.get("DS", "Skylion007/openwebtext")
OUT = os.environ.get("OUT", "/home/infatoshi/data/owt")
TARGET_TRAIN = int(os.environ.get("TARGET_TRAIN", str(300_000_000)))  # ~300M tokens
TARGET_VAL   = int(os.environ.get("TARGET_VAL",   str(2_000_000)))
os.makedirs(OUT, exist_ok=True)
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token

def write_split(path, target):
    buf = np.memmap(path, dtype=np.uint16, mode="w+", shape=(target,))
    n = 0
    ds = load_dataset(DS, split="train", streaming=True)
    for rec in ds:
        ids = enc.encode_ordinary(rec["text"]); ids.append(EOT)
        take = min(len(ids), target - n)
        buf[n:n+take] = np.array(ids[:take], dtype=np.uint16)
        n += take
        if n >= target: break
        if n % 20_000_000 < take: print(f"  {path}: {n/1e6:.0f}M", flush=True)
    buf.flush()
    print(f"WROTE {path}: {n} tokens", flush=True)
    return n

print(f"DS={DS} OUT={OUT} target train={TARGET_TRAIN/1e6:.0f}M val={TARGET_VAL/1e6:.1f}M", flush=True)
write_split(os.path.join(OUT, "val.bin"), TARGET_VAL)
write_split(os.path.join(OUT, "train.bin"), TARGET_TRAIN)
print("DONE", flush=True)
