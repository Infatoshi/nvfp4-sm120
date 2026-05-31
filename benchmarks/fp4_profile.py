"""Where does an FP4 linear's time go? Decompose fwd+bwd at realistic size into
RHT / quant / native GEMM, and compute achieved FP4 GEMM utilization vs peak."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import nvfp4_triton_quant as Q
from torchao.prototype.mx_formats.nvfp4_tensor import _addmm_nvfp4_dispatch
_MM = torch.ops.aten.mm.default
dev = "cuda"
PEAK_FP4, PEAK_BF16 = 2000.0, 500.0  # dense TFLOPS, sm_120


def t_ms(fn, it=50, wu=15):
    for _ in range(wu): fn()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it


def gemm_only(A, B):  # native FP4, pre-quantized operands excluded? no: quant+gemm
    a = Q.quant_nvfp4_fused(A); b = Q.quant_nvfp4_fused(B.t().contiguous())
    return _addmm_nvfp4_dispatch(a, b.t(), _MM)


def main():
    M, K, N = 8192, 4096, 4096          # one realistic FFN-ish linear
    flops = 2 * M * K * N / 1e12        # TFLOP
    A = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    B = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
    H = Q.hadamard16(dev); signs = (torch.randint(0, 2, (16,)).float() * 2 - 1).to(dev)

    # --- component timings ---
    qa = lambda: Q.quant_nvfp4_fused(A)                              # quant one operand (RNE)
    qa_sr = lambda: Q.quant_nvfp4_fused(A, stochastic=True)          # quant w/ SR
    rht = lambda: ((A.float().reshape(M, K // 16, 16) * signs) @ H)  # RHT prepass on A
    pre = Q.quant_nvfp4_fused(A); preb = Q.quant_nvfp4_fused(B.t().contiguous())
    gemm = lambda: _addmm_nvfp4_dispatch(pre, preb.t(), _MM)         # native GEMM only
    full_fprop = lambda: Q.fp4_matmul_fused(A, B)                    # quant A + quant B + gemm
    full_wgrad = lambda: Q.fp4_matmul_fused(A, B, sr_a=True, rht=True, H=H, signs=signs)
    bf16 = lambda: A @ B

    t = {k: t_ms(v) for k, v in dict(
        bf16=bf16, gemm=gemm, quant=qa, quant_sr=qa_sr, rht=rht,
        fprop=full_fprop, wgrad=full_wgrad).items()}

    print(f"shape {M}x{K}x{N}  ({flops:.2f} TFLOP/GEMM)\n")
    print(f"{'component':>16} | {'ms':>7} | {'TFLOPS':>7} | {'%FP4 peak':>9}")
    for k in ("bf16", "gemm", "quant", "quant_sr", "rht", "fprop", "wgrad"):
        tf = flops / (t[k] / 1e3)
        peak = PEAK_BF16 if k == "bf16" else PEAK_FP4
        print(f"{k:>16} | {t[k]:>7.3f} | {tf:>7.0f} | {100*tf/peak:>8.1f}%")

    print(f"\n-- Fprop breakdown (quant A + quant B + GEMM) --")
    print(f"   GEMM only      : {t['gemm']:.3f} ms  ({100*t['gemm']/t['fprop']:.0f}% of fprop)")
    print(f"   quant (x2 est) : {2*t['quant']:.3f} ms  ({100*2*t['quant']/t['fprop']:.0f}% of fprop)")
    print(f"-- Wgrad breakdown (RHT + quant + GEMM) --")
    print(f"   GEMM only      : {t['gemm']:.3f} ms  ({100*t['gemm']/t['wgrad']:.0f}% of wgrad)")
    print(f"   RHT prepass    : {t['rht']:.3f} ms  ({100*t['rht']/t['wgrad']:.0f}% of wgrad)")
    print(f"   quant_sr (x2e) : {2*t['quant_sr']:.3f} ms ({100*2*t['quant_sr']/t['wgrad']:.0f}% of wgrad)")
    print(f"\nGEMM speedup vs bf16: {t['bf16']/t['gemm']:.2f}x   "
          f"full-fprop speedup: {t['bf16']/t['fprop']:.2f}x   "
          f"full-wgrad: {t['bf16']/t['wgrad']:.2f}x")


if __name__ == "__main__":
    main()
