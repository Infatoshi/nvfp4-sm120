"""Full NVFP4 training pipeline on sm_120 with custom kernels (fwd + bwd).

Linear layers run all three GEMMs (Fprop/Dgrad/Wgrad) in native FP4 via
fp4_matmul (torch._scaled_mm backend), with the full recipe:
  - Fprop: round-to-nearest NVFP4 (activations + weights)
  - Dgrad: NVFP4, stochastic rounding on the gradient operand
  - Wgrad: NVFP4 + 16x16 Random Hadamard Transform + stochastic rounding on grad
FP32 master weights; attention/embed/head/norm in BF16.

Trains the 3-digit-addition char-LM (held-out pairs) to validate the full recipe
holds up with REAL FP4 GEMMs on sm_120.
"""
import argparse, time, contextlib, math, os
import torch, torch.nn as nn, torch.nn.functional as F
import nvfp4_validate as D
from nvfp4_gemm import fp4_matmul, hadamard16, BLK

DEV = "cuda"
_STEP = [0]      # bumped once per optimizer step; weight-quant cache keys on it
_AMORTIZE = os.environ.get("NVFP4_AMORTIZE", "0") == "1" and \
            os.environ.get("NVFP4_CUDA", "0") == "1"
if _AMORTIZE:
    from nvfp4_cuda import quant_nvfp4_cuda, fp4_mm_preqB


def mark_step():
    _STEP[0] += 1


class FP4LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, H, signs, wq_K, wq_N):   # x:[*,K]  w:[N,K]
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        if wq_K is not None:                    # amortized: weight pre-quantized once/step
            y = fp4_mm_preqB(x2, wq_K)          # X @ W^T using cached quant(w) [N,K]
        else:
            y = fp4_matmul(x2, w.t())           # X @ W^T  -> [M,N]  (RNE)
        ctx.save_for_backward(x2, w)
        ctx.H, ctx.signs, ctx.xshape, ctx.wq_N = H, signs, x.shape, wq_N
        return y.reshape(*x.shape[:-1], w.shape[0]).to(x.dtype)

    @staticmethod
    def backward(ctx, gy):
        x2, w = ctx.saved_tensors
        H, signs, wq_N = ctx.H, ctx.signs, ctx.wq_N
        gy2 = gy.reshape(-1, w.shape[0]).contiguous()
        # Dgrad: dX = dY @ W   (gradient -> SR; weight -> RNE, reuse cached quant(w.t()))
        if wq_N is not None:
            dx = fp4_mm_preqB(gy2, wq_N, sr_a=True)
        else:
            dx = fp4_matmul(gy2, w, sr_a=True, sr_b=False)
        # Wgrad: dW = dY^T @ X  (in-kernel RHT both operands; gradient -> SR; activation -> RNE)
        dw = fp4_matmul(gy2.t().contiguous(), x2, sr_a=True, sr_b=False,
                        rht=True, H=H, signs=signs)
        return dx.reshape(ctx.xshape).to(gy.dtype), dw.to(w.dtype), None, None, None, None


class FP4Linear(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        assert cin % BLK == 0 and cout % BLK == 0
        self.weight = nn.Parameter(torch.empty(cout, cin))
        nn.init.normal_(self.weight, std=0.02)
        self.register_buffer("H", hadamard16(DEV))
        g = torch.Generator().manual_seed(1)
        self.register_buffer("signs", (torch.randint(0, 2, (BLK,), generator=g).float() * 2 - 1))
        self._cache_step = -1
        self._wq_K = self._wq_N = None

    def forward(self, x):
        wq_K = wq_N = None
        if _AMORTIZE:
            step = _STEP[0]
            if self._cache_step != step:        # quantize constant weight once per step
                w = self.weight.detach()
                self._wq_K = quant_nvfp4_cuda(w)                   # [N,K] contract K (fprop)
                self._wq_N = quant_nvfp4_cuda(w.t().contiguous())  # [K,N] contract N (dgrad)
                self._cache_step = step
            wq_K, wq_N = self._wq_K, self._wq_N
        return FP4LinearFn.apply(x, self.weight, self.H, self.signs, wq_K, wq_N)


def make_linear(cin, cout, use_fp4):
    if use_fp4:
        return FP4Linear(cin, cout)
    lin = nn.Linear(cin, cout, bias=False)
    nn.init.normal_(lin.weight, std=0.02)
    return lin


def rope(x, T):
    B, Hh, _, Dh = x.shape
    half = Dh // 2
    inv = 1.0 / (10000 ** (torch.arange(0, half, device=x.device).float() / half))
    ang = torch.outer(torch.arange(T, device=x.device).float(), inv)
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Block(nn.Module):
    def __init__(self, dim, nh, nkv, hd, ffn, use_fp4):
        super().__init__()
        self.nh, self.nkv, self.hd = nh, nkv, hd
        self.n1, self.n2 = nn.RMSNorm(dim), nn.RMSNorm(dim)
        L = lambda i, o: make_linear(i, o, use_fp4)
        self.q, self.k, self.v = L(dim, nh * hd), L(dim, nkv * hd), L(dim, nkv * hd)
        self.o = L(nh * hd, dim)
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
    def __init__(self, vocab, dim, nl, nh, nkv, hd, ffn, use_fp4, hp_tail=0):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        def blk_fp4(i):
            return use_fp4 and not (i == 0 or i >= nl - hp_tail)
        self.blocks = nn.ModuleList([Block(dim, nh, nkv, hd, ffn, blk_fp4(i)) for i in range(nl)])
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


@torch.no_grad()
def evaluate(model, val, g, n=8, bs=512):
    model.eval()
    losses, correct, total = [], 0, 0
    for _ in range(n):
        x, y = D.make_batch(val, bs, g)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1)).item())
        pred = logits[:, D.ANS_SLICE, :].argmax(-1)
        correct += (pred == y[:, D.ANS_SLICE]).all(1).sum().item()
        total += x.size(0)
    model.train()
    return sum(losses) / len(losses), correct / total


def run(tag, steps, use_fp4, hp_tail=0, seed=0):
    torch.manual_seed(seed)
    dim, nl, nh, nkv, hd, ffn = 256, 6, 4, 2, 64, 768
    model = LM(len(D.VOCAB), dim, nl, nh, nkv, hd, ffn, use_fp4, hp_tail).to(DEV)
    train, val = D.build_dataset()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.05)
    g = torch.Generator().manual_seed(seed + 7)
    t0, diverged, last = time.time(), False, {}
    for step in range(1, steps + 1):
        x, y = D.make_batch(train, 512, g)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
        if not torch.isfinite(loss):
            diverged = True; break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        mark_step()     # invalidate per-step weight-quant cache after the update
        if step % max(1, steps // 6) == 0 or step == steps:
            vl, acc = evaluate(model, val, g)
            last = dict(step=step, train=round(loss.item(), 4), val=round(vl, 4), acc=round(acc, 4))
            print(f"  [{tag:>12}] step {step:>4} | train {loss.item():.4f} | val {vl:.4f} | acc {acc*100:5.1f}%")
    print(f"  -> {tag}: time={time.time()-t0:.1f}s diverged={diverged} final={last}")
    return dict(tag=tag, diverged=diverged, **last)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=1500)
    steps = ap.parse_args().steps
    print(f"GPU: {torch.cuda.get_device_name(0)} | custom NVFP4 (SR+RHT) native FP4 GEMMs | steps={steps}\n")
    print("== BF16 reference ==")
    r1 = run("bf16", steps, use_fp4=False)
    print("\n== NVFP4 full recipe, all blocks (SR+RHT, native FP4) ==")
    r2 = run("fp4-all", steps, use_fp4=True, hp_tail=0)
    print("\n== NVFP4 full recipe + high-precision tail (first+last 2 BF16) ==")
    r3 = run("fp4-hp2", steps, use_fp4=True, hp_tail=2)
    print("\n========= SUMMARY =========")
    for r in (r1, r2, r3):
        print(f"  {r['tag']:>10} | val {r.get('val'):.4f} | acc {r.get('acc',0)*100:5.1f}% | diverged={r['diverged']}")


if __name__ == "__main__":
    main()
