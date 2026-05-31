"""Throughput of the custom native-FP4 GEMM (torch._scaled_mm path) vs bf16 on sm_120.
Isolates the GEMM (pre-quantized operands) from quantization overhead."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time, torch
from nvfp4_gemm import quant_nvfp4
from torchao.prototype.mx_formats.nvfp4_tensor import _addmm_nvfp4_dispatch
_MM = torch.ops.aten.mm.default
PEAK_DENSE_FP4 = 2000.0  # TFLOPS, dense (headline 4 PFLOP is with sparsity)


def _time(fn, iters=50, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def bench(M, K, N):
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    # pre-quantize once (isolate GEMM from quant overhead)
    a = quant_nvfp4(A.contiguous())
    b = quant_nvfp4(B.t().contiguous()).t()
    flops = 2 * M * K * N
    t_bf16 = _time(lambda: A @ B)
    t_fp4 = _time(lambda: _addmm_nvfp4_dispatch(a, b, _MM))
    return flops / t_bf16 / 1e12, flops / t_fp4 / 1e12


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}  | native FP4 GEMM (torch._scaled_mm)")
    print(f"{'M=K=N':>7} | {'bf16 TFLOPS':>11} | {'FP4 TFLOPS':>10} | {'speedup':>7} | {'%FP4peak':>8}")
    for S in [4096, 8192, 16384]:
        b16, f4 = bench(S, S, S)
        print(f"{S:>7} | {b16:>11.0f} | {f4:>10.0f} | {f4/b16:>6.2f}x | {100*f4/PEAK_DENSE_FP4:>7.1f}%")


if __name__ == "__main__":
    main()
