"""
Downscaled validation of NVIDIA's NVFP4 pretraining recipe (arXiv 2509.25149).

This is a NUMERICAL SIMULATION of NVFP4 GEMMs (fake-quant), not hardware FP4
Tensor Core ops. It faithfully implements:
  - NVFP4 format: E2M1 elements, block size 16, per-block E4M3 scale,
    per-tensor FP32 second-level scale (two-level scaling, sec. 2 / appendix B).
  - Selective high-precision layers: first block + last K blocks in BF16.
  - Random Hadamard Transform (16x16, random sign vector) on Wgrad inputs only.
  - 2D (16x16) block scaling for WEIGHTS  -> forward/backward consistency.
    1D (1x16) scaling for activations & gradients.
  - Stochastic rounding for GRADIENT tensors; round-to-nearest-even for
    weights & activations.
GEMMs accumulate in FP32, mirroring Blackwell Tensor Core behavior.

Task: 3-digit addition as a char-level LM ("007+456=0463\n"), fixed width.
Generalization is measured on held-out (a,b) pairs the model never saw.
"""
import argparse, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEV = "cuda"
BLOCK = 16
E2M1_MAX = 6.0
E4M3_MAX = 448.0
_LEVELS = torch.tensor([0., 0.5, 1., 1.5, 2., 3., 4., 6.])  # E2M1 magnitudes

# ----------------------------- NVFP4 primitives -----------------------------

def _round_e2m1(x, stochastic):
    L = _LEVELS.to(x.device, x.dtype)
    sign = torch.sign(x)
    m = x.abs().clamp(max=E2M1_MAX)
    idx = torch.bucketize(m, L, right=True).clamp(1, 7)
    hi = L[idx]
    lo = L[idx - 1]
    width = (hi - lo).clamp_min(1e-9)
    if stochastic:
        p = (m - lo) / width
        q = torch.where(torch.rand_like(m) < p, hi, lo)
    else:  # round-to-nearest (ties -> down, negligible here)
        q = torch.where((m - lo) <= (hi - m), lo, hi)
    return sign * q


def quantize_nvfp4_1d(x, stochastic=False):
    """1x16 microscaling along the last dim with two-level (FP32 + E4M3) scaling."""
    shp = x.shape
    assert shp[-1] % BLOCK == 0, f"last dim {shp[-1]} not divisible by {BLOCK}"
    xf = x.float()
    amax_t = xf.abs().amax().clamp_min(1e-12)
    s_enc = (E2M1_MAX * E4M3_MAX) / amax_t          # global encode scale (scalar)
    s_dec = 1.0 / s_enc
    xb = xf.reshape(*shp[:-1], shp[-1] // BLOCK, BLOCK)
    amax_b = xb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    s_dec_b = amax_b / E2M1_MAX
    bs_e4m3 = (s_dec_b * s_enc).to(torch.float8_e4m3fn).float()   # quantized block scale
    deq = (bs_e4m3 * s_dec).clamp_min(1e-12)         # effective per-block dequant scale
    q = _round_e2m1(xb / deq, stochastic)
    return (q * deq).reshape(shp).to(x.dtype)


def quantize_nvfp4_2d(w, stochastic=False):
    """2D 16x16 block scaling for weights -> identical quantized W in fwd & bwd.
    w: [out, in], both divisible by 16."""
    out, cin = w.shape
    wf = w.float()
    amax_t = wf.abs().amax().clamp_min(1e-12)
    s_enc = (E2M1_MAX * E4M3_MAX) / amax_t
    s_dec = 1.0 / s_enc
    wb = wf.reshape(out // BLOCK, BLOCK, cin // BLOCK, BLOCK)     # [ob,16,ib,16]
    amax_b = wb.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12)
    s_dec_b = amax_b / E2M1_MAX
    bs_e4m3 = (s_dec_b * s_enc).to(torch.float8_e4m3fn).float()
    deq = (bs_e4m3 * s_dec).clamp_min(1e-12)
    q = _round_e2m1(wb / deq, stochastic)
    return (q * deq).reshape(out, cin).to(w.dtype)


def _hadamard16(device, dtype):
    H = torch.ones(1, 1, device=device, dtype=dtype)
    while H.shape[0] < BLOCK:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(BLOCK)            # orthogonal, symmetric


def apply_rht_lastdim(x, Hn, signs):
    """Random Hadamard transform along last dim, in tiles of 16.
    Transform M = Hn @ diag(signs) is orthogonal so it cancels across the
    contracted operand pair (used on the Wgrad reduction = token dim)."""
    *pre, L = x.shape
    xb = x.reshape(*pre, L // BLOCK, BLOCK)
    xb = (xb * signs) @ Hn                 # symmetric Hn
    return xb.reshape(*pre, L)


# ----------------------------- NVFP4 Linear -----------------------------

class _NVFP4LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, cfg, Hn, signs):
        # x: [N, Cin]  w: [Cout, Cin]
        xq = quantize_nvfp4_1d(x, stochastic=False)            # activation, RNE
        wq = quantize_nvfp4_2d(w, stochastic=False)            # weight 2D, RNE
        out = (xq.float() @ wq.float().t()).to(x.dtype)        # FP32 accumulate
        ctx.save_for_backward(x, w)
        ctx.cfg, ctx.Hn, ctx.signs = cfg, Hn, signs
        return out

    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors
        cfg, Hn, signs = ctx.cfg, ctx.Hn, ctx.signs
        sr = cfg["sr"]

        # ---- Dgrad: dx = gy @ W  (gy is gradient -> SR; W reuses 2D quant) ----
        gyq = quantize_nvfp4_1d(gy, stochastic=sr)             # along Cout
        wq2 = quantize_nvfp4_2d(w, stochastic=False)           # SAME as forward
        dx = (gyq.float() @ wq2.float()).to(x.dtype)

        # ---- Wgrad: dW = gy^T @ x, contraction over tokens (dim 0) ----
        xt = x.t().contiguous()        # [Cin, N]  -> contraction (N) is last
        gyt = gy.t().contiguous()      # [Cout, N]
        N = xt.shape[-1]
        pad = (-N) % BLOCK
        if pad:
            xt = F.pad(xt, (0, pad))
            gyt = F.pad(gyt, (0, pad))
        if cfg["rht"]:
            xt = apply_rht_lastdim(xt, Hn, signs)
            gyt = apply_rht_lastdim(gyt, Hn, signs)            # same M -> cancels
        xtq = quantize_nvfp4_1d(xt, stochastic=False)          # activation, RNE
        gytq = quantize_nvfp4_1d(gyt, stochastic=sr)           # gradient -> SR
        dw = (gytq.float() @ xtq.float().t()).to(w.dtype)      # [Cout, Cin]
        return dx, dw, None, None, None


class NVFP4Linear(nn.Module):
    def __init__(self, cin, cout, cfg):
        super().__init__()
        assert cin % BLOCK == 0 and cout % BLOCK == 0
        self.weight = nn.Parameter(torch.empty(cout, cin))
        nn.init.normal_(self.weight, std=0.02)
        self.cfg = cfg

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        out = _NVFP4LinearFn.apply(x2, self.weight, self.cfg,
                                   self.cfg["Hn"], self.cfg["signs"])
        return out.reshape(*shp[:-1], -1)


def make_linear(cin, cout, cfg, high_precision):
    if high_precision or cfg["mode"] == "bf16":
        lin = nn.Linear(cin, cout, bias=False)
        nn.init.normal_(lin.weight, std=0.02)
        return lin
    return NVFP4Linear(cin, cout, cfg)


# ----------------------------- Model -----------------------------

def rope(x, seq):
    # x: [B, H, T, D]
    B, H, T, D = x.shape
    half = D // 2
    freqs = torch.arange(0, half, device=x.device).float()
    inv = 1.0 / (10000 ** (freqs / half))
    t = torch.arange(T, device=x.device).float()
    ang = torch.outer(t, inv)                       # [T, half]
    cos = ang.cos()[None, None]
    sin = ang.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Block(nn.Module):
    def __init__(self, dim, n_head, n_kv, head_dim, ffn, cfg, hp):
        super().__init__()
        self.n_head, self.n_kv, self.hd = n_head, n_kv, head_dim
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.q = make_linear(dim, n_head * head_dim, cfg, hp)
        self.k = make_linear(dim, n_kv * head_dim, cfg, hp)
        self.v = make_linear(dim, n_kv * head_dim, cfg, hp)
        self.o = make_linear(n_head * head_dim, dim, cfg, hp)
        self.up = make_linear(dim, ffn, cfg, hp)
        self.down = make_linear(ffn, dim, cfg, hp)

    def forward(self, x):
        B, T, C = x.shape
        h = self.norm1(x)
        q = self.q(h).view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = self.k(h).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.v(h).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        q, k = rope(q, T), rope(k, T)
        rep = self.n_head // self.n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        # attention math in BF16/FP32 (kept high precision per paper)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(B, T, -1)
        x = x + self.o(a)
        h = self.norm2(x)
        h = self.up(h)
        h = F.relu(h) ** 2                              # squared ReLU
        x = x + self.down(h)
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab, dim, n_layer, n_head, n_kv, head_dim, ffn, cfg):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)             # always high precision
        self.blocks = nn.ModuleList()
        # selective high precision: first block + last K blocks in BF16
        K = cfg["hp_tail"]
        for i in range(n_layer):
            hp = (i == 0) or (i >= n_layer - K)
            self.blocks.append(Block(dim, n_head, n_kv, head_dim, ffn, cfg, hp))
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)   # output head high precision

    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


# ----------------------------- Data: 3-digit addition -----------------------------

VOCAB = list("0123456789+=\n")
STOI = {c: i for i, c in enumerate(VOCAB)}
SEQ = 13  # "DDD+DDD=DDDD\n"

def encode_pair(a, b):
    s = f"{a:03d}+{b:03d}={a+b:04d}\n"
    return [STOI[c] for c in s]

def build_dataset():
    pairs = [(a, b) for a in range(1000) for b in range(1000)]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(pairs), generator=g)
    pairs = [pairs[i] for i in perm.tolist()]
    n_val = 100_000
    val = pairs[:n_val]
    train = pairs[n_val:]
    return train, val

def make_batch(pairs, bs, g):
    idx = torch.randint(0, len(pairs), (bs,), generator=g)
    rows = [encode_pair(*pairs[i]) for i in idx.tolist()]
    t = torch.tensor(rows, device=DEV)
    return t[:, :-1], t[:, 1:]           # input, target (len 12)

# answer occupies last 5 target positions: "0463\n" -> indices 7..11 of target
ANS_SLICE = slice(7, 12)

@torch.no_grad()
def evaluate(model, val, g, n_batches=8, bs=512):
    model.eval()
    losses, correct, total = [], 0, 0
    for _ in range(n_batches):
        x, y = make_batch(val, bs, g)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               y.reshape(-1))
        losses.append(loss.item())
        pred = logits[:, ANS_SLICE, :].argmax(-1)
        tgt = y[:, ANS_SLICE]
        correct += (pred == tgt).all(dim=1).sum().item()
        total += x.size(0)
    model.train()
    return sum(losses) / len(losses), correct / total


# ----------------------------- Train -----------------------------

def run(mode, steps, hp_tail=2, sr=True, rht=True, seed=0):
    torch.manual_seed(seed)
    dim, n_layer, n_head, n_kv, head_dim, ffn = 256, 6, 4, 2, 64, 768
    cfg = {
        "mode": mode, "sr": sr, "rht": rht, "hp_tail": hp_tail,
        "Hn": _hadamard16(DEV, torch.float32),
        "signs": (torch.randint(0, 2, (BLOCK,), generator=torch.Generator().manual_seed(1)
                                 ).float() * 2 - 1).to(DEV),
    }
    model = TinyLM(len(VOCAB), dim, n_layer, n_head, n_kv, head_dim, ffn, cfg).to(DEV)
    nparams = sum(p.numel() for p in model.parameters())
    train, val = build_dataset()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps,
                                                pct_start=0.05, anneal_strategy="cos")
    g = torch.Generator().manual_seed(seed + 7)
    t0 = time.time()
    diverged = False
    last = {}
    for step in range(1, steps + 1):
        x, y = make_batch(train, 512, g)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
        if not torch.isfinite(loss):
            diverged = True
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % max(1, steps // 6) == 0 or step == steps:
            vl, acc = evaluate(model, val, g)
            last = dict(step=step, train=loss.item(), val=vl, acc=acc)
            print(f"  [{mode:>10}] step {step:>4} | train {loss.item():.4f} "
                  f"| val {vl:.4f} | val-acc {acc*100:5.1f}%")
    dt = time.time() - t0
    tag = mode
    print(f"  -> {tag}: params={nparams/1e6:.2f}M  time={dt:.1f}s  "
          f"diverged={diverged}  final={last}")
    return dict(mode=mode, params=nparams, time=dt, diverged=diverged, **last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()
    print(f"Device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"Task: 3-digit addition char-LM | train pairs=900k val(held-out)=100k | steps={args.steps}\n")
    results = []
    print("== BF16 baseline ==")
    results.append(run("bf16", args.steps))
    print("\n== NVFP4 full recipe (RHT + 2D + SR + HP tail=2) ==")
    results.append(run("nvfp4", args.steps, hp_tail=2, sr=True, rht=True))
    print("\n== NVFP4 ablation: no stochastic rounding ==")
    results.append(run("nvfp4-noSR", args.steps, hp_tail=2, sr=False, rht=True))
    print("\n== NVFP4 ablation: no high-precision layers (all linear FP4) ==")
    results.append(run("nvfp4-allFP4", args.steps, hp_tail=0, sr=True, rht=True))
    print("\n================ SUMMARY ================")
    print(f"{'config':>14} | {'val loss':>9} | {'val acc':>8} | {'diverged':>8}")
    for r in results:
        print(f"{r['mode']:>14} | {r.get('val', float('nan')):>9.4f} | "
              f"{r.get('acc', 0)*100:>7.1f}% | {str(r['diverged']):>8}")


if __name__ == "__main__":
    main()
