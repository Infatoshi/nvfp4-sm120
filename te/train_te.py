"""Real-hardware NVFP4 training on RTX PRO 6000 (sm_120) via Transformer Engine.

Uses te.pytorch.Linear under the NVFP4 recipe (2D weight scaling + RNE; RHT and
stochastic rounding are DISABLED because those fused kernels error on sm_120 in
TE 2.15). Trains the same 3-digit-addition char-LM with held-out (a,b) pairs and
checks generalization, side by side with a BF16 reference of the identical model.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, time, contextlib
import torch, torch.nn as nn, torch.nn.functional as F
import transformer_engine.pytorch as tep
from transformer_engine.common.recipe import NVFP4BlockScaling
import nvfp4_validate as D   # reuse data utils (VOCAB, dataset, batching)

DEV = "cuda"
RECIPE = lambda: NVFP4BlockScaling()  # FULL recipe (RHT+SR); patch auto-degrades on sm_120


def rope(x, T):
    B, H, _, Dh = x.shape
    half = Dh // 2
    inv = 1.0 / (10000 ** (torch.arange(0, half, device=x.device).float() / half))
    ang = torch.outer(torch.arange(T, device=x.device).float(), inv)
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Block(nn.Module):
    def __init__(self, dim, nh, nkv, hd, ffn, use_te):
        super().__init__()
        self.nh, self.nkv, self.hd = nh, nkv, hd
        self.n1, self.n2 = nn.RMSNorm(dim), nn.RMSNorm(dim)
        L = (lambda i, o: tep.Linear(i, o, bias=False, params_dtype=torch.float32)) if use_te \
            else (lambda i, o: nn.Linear(i, o, bias=False))
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


class LM(nn.Module):
    def __init__(self, vocab, dim, nl, nh, nkv, hd, ffn, use_te, hp_tail=0):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        # selective high precision: first block + last hp_tail blocks stay BF16
        def block_uses_te(i):
            if not use_te:
                return False
            return not (i == 0 or i >= nl - hp_tail)
        self.blocks = nn.ModuleList([Block(dim, nh, nkv, hd, ffn, block_uses_te(i)) for i in range(nl)])
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


@torch.no_grad()
def evaluate(model, val, g, recipe, n=8, bs=512):
    model.eval()
    losses, correct, total = [], 0, 0
    for _ in range(n):
        x, y = D.make_batch(val, bs, g)
        ctx = tep.fp8_autocast(enabled=True, fp8_recipe=recipe()) if recipe else contextlib.nullcontext()
        with torch.autocast("cuda", dtype=torch.bfloat16), ctx:
            logits = model(x)
        losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1)).item())
        pred = logits[:, D.ANS_SLICE, :].argmax(-1)
        correct += (pred == y[:, D.ANS_SLICE]).all(1).sum().item()
        total += x.size(0)
    model.train()
    return sum(losses)/len(losses), correct/total


def run(tag, steps, recipe, hp_tail=0, seed=0):
    torch.manual_seed(seed)
    dim, nl, nh, nkv, hd, ffn = 256, 6, 4, 2, 64, 768
    model = LM(len(D.VOCAB), dim, nl, nh, nkv, hd, ffn,
               use_te=(recipe is not None), hp_tail=hp_tail).to(DEV)
    train, val = D.build_dataset()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.05)
    g = torch.Generator().manual_seed(seed + 7)
    t0, diverged = time.time(), False
    last = {}
    for step in range(1, steps + 1):
        x, y = D.make_batch(train, 512, g)
        ctx = tep.fp8_autocast(enabled=True, fp8_recipe=recipe()) if recipe else contextlib.nullcontext()
        with torch.autocast("cuda", dtype=torch.bfloat16), ctx:
            logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
        if not torch.isfinite(loss):
            diverged = True; break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % max(1, steps // 6) == 0 or step == steps:
            vl, acc = evaluate(model, val, g, recipe)
            last = dict(step=step, train=round(loss.item(), 4), val=round(vl, 4), acc=round(acc, 4))
            print(f"  [{tag:>10}] step {step:>4} | train {loss.item():.4f} | val {vl:.4f} | acc {acc*100:5.1f}%")
    print(f"  -> {tag}: time={time.time()-t0:.1f}s diverged={diverged} final={last}")
    return dict(tag=tag, diverged=diverged, **last)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=1500)
    steps = ap.parse_args().steps
    print(f"GPU: {torch.cuda.get_device_name(0)} | real-hardware NVFP4 via TE 2.15 | steps={steps}\n")
    print("== BF16 reference (te disabled) ==")
    r1 = run("bf16", steps, None)
    print("\n== NVFP4 all-FP4 blocks (2D weight scaling + RNE) ==")
    r2 = run("nvfp4-all", steps, RECIPE, hp_tail=0)
    print("\n== NVFP4 + high-precision tail (first + last 2 blocks BF16) ==")
    r3 = run("nvfp4-hp2", steps, RECIPE, hp_tail=2)
    print("\n========= SUMMARY =========")
    for r in (r1, r2, r3):
        print(f"  {r['tag']:>10} | val {r.get('val'):.4f} | acc {r.get('acc',0)*100:5.1f}% | diverged={r['diverged']}")


if __name__ == "__main__":
    main()
