"""FP4/FP8/BF16 GEMM throughput on RTX PRO 6000 Blackwell via Transformer Engine.
Measures achieved TFLOPS for square GEMMs and compares to dense tensor-core peak."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time, torch
import transformer_engine.pytorch as tep
from transformer_engine.common.recipe import (
    NVFP4BlockScaling, MXFP8BlockScaling, DelayedScaling, Float8CurrentScaling)


def first_supported_fp8():
    """Return (name, recipe) for the first FP8 recipe that works on this arch."""
    for name, mk in [("MXFP8", MXFP8BlockScaling), ("FP8-cur", Float8CurrentScaling),
                     ("FP8-delay", DelayedScaling)]:
        try:
            lin = tep.Linear(64, 64, bias=False, params_dtype=torch.bfloat16).cuda()
            x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
            with tep.fp8_autocast(enabled=True, fp8_recipe=mk()):
                lin(x)
            return name, mk
        except Exception:
            continue
    return None, None

# Dense (non-sparse) tensor-core peaks for RTX PRO 6000 Blackwell.
# Headline marketing numbers are WITH 2:4 sparsity (FP4 4 PFLOP, FP8 2 PFLOP,
# BF16 1 PFLOP); dense is half of each.
PEAK_DENSE = {"bf16": 500.0, "fp8": 1000.0, "fp4": 2000.0}  # TFLOPS


def _time(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def bench_te(M, K, N, recipe, iters=50, warmup=20):
    lin = tep.Linear(K, N, bias=False, params_dtype=torch.bfloat16).cuda()
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    # NOTE: NVFP4 forward quant kernel errors under torch.no_grad() on sm_120,
    # so run the normal (grad-enabled) forward path. We never call backward here,
    # so this measures forward-GEMM throughput.
    def step():
        with tep.fp8_autocast(enabled=recipe is not None, fp8_recipe=recipe):
            return lin(x)

    dt = _time(step, iters, warmup)
    return (2 * M * K * N) / dt / 1e12, dt


def bench_torch_bf16(M, K, N, iters=50, warmup=20):
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

    @torch.no_grad()
    def step():
        return a @ b

    dt = _time(step, iters, warmup)
    return (2 * M * K * N) / dt / 1e12, dt


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Dense tensor-core peaks assumed (TFLOPS):", PEAK_DENSE, "\n")
    fp8_name, fp8_mk = first_supported_fp8()
    print(f"FP8 recipe used: {fp8_name or 'NONE supported on this arch'}\n")
    sizes = [4096, 8192, 16384]
    print(f"{'M=K=N':>7} | {'torch bf16':>11} | {'TE bf16':>9} | {fp8_name or 'FP8':>9} | "
          f"{'TE NVFP4':>9} | {'fp4/bf16':>8} | {'fp4/fp8':>7} | {'%peak fp4':>9}")
    for S in sizes:
        tb, _ = bench_torch_bf16(S, S, S)
        teb, _ = bench_te(S, S, S, None)
        f8 = bench_te(S, S, S, fp8_mk())[0] if fp8_mk else float("nan")
        f4, _ = bench_te(S, S, S, NVFP4BlockScaling(
            disable_rht=True, disable_stochastic_rounding=True))  # RHT/SR kernels error on sm_120
        r8 = f"{f4/f8:>6.2f}x" if f8 == f8 else "   N/A"
        print(f"{S:>7} | {tb:>11.0f} | {teb:>9.0f} | {f8:>9.0f} | {f4:>9.0f} | "
              f"{f4/teb:>7.2f}x | {r8:>7} | {100*f4/PEAK_DENSE['fp4']:>8.1f}%")


if __name__ == "__main__":
    main()
