"""Validate the fused Triton NVFP4 quantizer vs the reference torch path:
byte-exact RNE match, GEMM rel-error, SR unbiasedness, and quant speedup,
swept across shapes valid for sm_120."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time, torch
import nvfp4_triton_quant as Q
import nvfp4_gemm as G


def _packed(nvt):
    for a in ("qdata", "_data", "data"):
        v = getattr(nvt, a, None)
        if isinstance(v, torch.Tensor) and v.dtype == torch.uint8:
            return v
    raise RuntimeError("no packed uint8 data attr; have: " +
                       ",".join(k for k in dir(nvt) if not k.startswith("__")))


def sweep():
    dev = "cuda"
    torch.manual_seed(0)
    shapes = [(256, 512, 256), (512, 1024, 512), (1024, 1024, 1024),
              (2048, 4096, 2048), (4096, 2048, 4096), (1024, 4096, 4096),
              (4096, 8192, 4096), (333*16, 512, 512)]  # last: non-pow2 M
    print(f"{'M,K,N':>20} | {'rel torch':>9} | {'rel fused':>9} | {'RNE byte match':>15} | {'fused≈torch':>11}")
    worst_byte = 1.0
    for (M, K, N) in shapes:
        A = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        B = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
        ref = A.float() @ B.float()
        rn = ref.norm()
        ot = G.fp4_matmul(A, B).float()
        of = Q.fp4_matmul_fused(A, B).float()
        rel_t = ((ot - ref).norm() / rn).item()
        rel_f = ((of - ref).norm() / rn).item()
        agree = ((ot - of).norm() / rn).item()
        # byte-exact RNE check on the quantizer itself (operand A, [M,K])
        bt = _packed(G.quant_nvfp4(A.contiguous(), stochastic=False))
        bf = _packed(Q.quant_nvfp4_fused(A.contiguous(), stochastic=False))
        match = (bt == bf).float().mean().item()
        worst_byte = min(worst_byte, match)
        print(f"{str((M,K,N)):>20} | {rel_t:>9.4f} | {rel_f:>9.4f} | {match*100:>13.2f}% | {agree:>11.2e}")
    print(f"\nworst RNE byte-match across shapes: {worst_byte*100:.2f}%")

    # SR unbiasedness (fused), one shape
    M, K, N = 1024, 1024, 1024
    A = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    B = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
    ref = A.float() @ B.float(); rn = ref.norm()
    rne = ((Q.fp4_matmul_fused(A, B).float() - ref).norm() / rn).item()
    acc = torch.zeros_like(ref)
    Nd = 64
    for i in range(Nd):
        acc += Q.fp4_matmul_fused(A, B, sr_a=True, sr_b=True, seed=1000 + i).float()
    sr = ((acc / Nd - ref).norm() / rn).item()
    print(f"\n[SR] fused: RNE rel={rne:.4f}  SRx{Nd} mean rel={sr:.4f}  "
          f"-> {'UNBIASED (SR<RNE)' if sr < rne else 'check'}")

    # quant speed: fused vs torch, big operand
    M, K = 4096, 8192
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    for fn, name in [(lambda: G.quant_nvfp4(x), "torch"), (lambda: Q.quant_nvfp4_fused(x), "fused")]:
        for _ in range(5):
            fn()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(30):
            fn()
        torch.cuda.synchronize()
        print(f"[quant {name:>5}] {(time.perf_counter()-t0)/30*1e3:.3f} ms  ({M}x{K})")


if __name__ == "__main__":
    sweep()
