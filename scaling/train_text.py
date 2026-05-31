"""nanoGPT-style text LM trainer over GPT-2 BPE .bin data, BF16 vs NVFP4.

Reuses the FP4Linear autograd module + recipe from nvfp4_train.py. Decoder-only
Transformer (RMSNorm, GQA, RoPE, squared-ReLU FFN), weight-tied head. Tracks val
loss for a Chinchilla-style L(N) scaling study. Backend via env:
  (default) bf16 dense nn.Linear   |   NVFP4_CUDA=1 -> FP4Linear (full SR+RHT recipe)
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, os, time, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

# reuse the validated FP4 linear + helpers
from nvfp4_train import FP4Linear, mark_step, _AMORTIZE
from nvfp4_gemm import BLK

DEV = "cuda"
USE_FP4 = os.environ.get("NVFP4_CUDA", "0") == "1" or os.environ.get("NVFP4_FUSED", "0") == "1"


def make_linear(cin, cout, fp4):
    if fp4:
        return FP4Linear(cin, cout)
    return nn.Linear(cin, cout, bias=False)


def rope(x, T):
    B, H, _, Dh = x.shape
    half = Dh // 2
    inv = 1.0 / (10000 ** (torch.arange(0, half, device=x.device).float() / half))
    ang = torch.outer(torch.arange(T, device=x.device).float(), inv)
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


class Block(nn.Module):
    def __init__(self, dim, nh, nkv, hd, ffn, fp4):
        super().__init__()
        self.nh, self.nkv, self.hd = nh, nkv, hd
        self.n1, self.n2 = nn.RMSNorm(dim), nn.RMSNorm(dim)
        L = lambda i, o: make_linear(i, o, fp4)
        self.q, self.k, self.v = L(dim, nh*hd), L(dim, nkv*hd), L(dim, nkv*hd)
        self.o = L(nh*hd, dim)
        self.up, self.down = L(dim, ffn), L(ffn, dim)

    def forward(self, x):
        B, T, C = x.shape
        h = self.n1(x)
        q = self.q(h).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k(h).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v(h).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = rope(q, T), rope(k, T)
        rep = self.nh // self.nkv
        a = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1),
                                           v.repeat_interleave(rep, 1), is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, -1))
        x = x + self.down(F.relu(self.up(self.n2(x))) ** 2)
        return x


class GPT(nn.Module):
    def __init__(self, vocab, dim, nl, nh, nkv, hd, ffn, fp4, hp_tail=1):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        def blk_fp4(i):  # keep first + last hp_tail blocks dense (paper's rule)
            return fp4 and not (i == 0 or i >= nl - hp_tail)
        self.blocks = nn.ModuleList([Block(dim, nh, nkv, hd, ffn, blk_fp4(i)) for i in range(nl)])
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.emb.weight        # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1))
        return logits, loss

    def num_params(self):  # non-embedding param count (Chinchilla convention)
        n = sum(p.numel() for p in self.parameters())
        return n - self.emb.weight.numel()        # head is tied to emb


def get_batch(data, bs, T, gen):
    ix = torch.randint(len(data) - T - 1, (bs,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i:i+T].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+T].astype(np.int64)) for i in ix])
    return x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)


@torch.no_grad()
def eval_loss(model, data, bs, T, iters, gen):
    model.eval()
    tot = 0.0
    for _ in range(iters):
        x, y = get_batch(data, bs, T, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, l = model(x, y)
        tot += l.item()
    model.train()
    return tot / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--nl", type=int, default=6)
    ap.add_argument("--nh", type=int, default=6)
    ap.add_argument("--nkv", type=int, default=6)
    ap.add_argument("--ffn", type=int, default=None)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--data", default="/home/infatoshi/data/owt")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="/home/infatoshi/data/scaling_results.jsonl")
    args = ap.parse_args()
    hd = args.dim // args.nh
    ffn = args.ffn or 4 * args.dim
    torch.manual_seed(0)

    train = np.memmap(os.path.join(args.data, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(args.data, "val.bin"), dtype=np.uint16, mode="r")

    model = GPT(50257, args.dim, args.nl, args.nh, args.nkv, hd, ffn, USE_FP4).to(DEV)
    N = model.num_params()
    tokens = args.steps * args.bs * args.grad_accum * args.T
    backend = "nvfp4" if USE_FP4 else "bf16"
    print(f"[{args.tag}] backend={backend} dim={args.dim} nl={args.nl} N={N/1e6:.2f}M "
          f"tokens={tokens/1e6:.0f}M tok/param={tokens/N:.1f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    def lr_at(s):
        if s < args.warmup: return args.lr * s / args.warmup
        p = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))
    gtr = torch.Generator().manual_seed(1)
    gva = torch.Generator().manual_seed(2)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        for pg in opt.param_groups: pg["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y = get_batch(train, args.bs, args.T, gtr)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            (loss / args.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if _AMORTIZE: mark_step()
        if step % 500 == 0 or step == args.steps:
            vl = eval_loss(model, val, args.bs, args.T, 40, gva)
            print(f"  step {step:>5} | lr {lr_at(step):.1e} | train {loss.item():.4f} "
                  f"| val {vl:.4f} | {(time.time()-t0)/step*1000:.0f} ms/step", flush=True)
    vl = eval_loss(model, val, args.bs, args.T, 80, gva)
    dt = time.time() - t0
    rec = dict(tag=args.tag, backend=backend, dim=args.dim, nl=args.nl, N=N,
               tokens=tokens, val_loss=vl, steps=args.steps, minutes=dt/60)
    print(f"RESULT {rec}", flush=True)
    import json
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
