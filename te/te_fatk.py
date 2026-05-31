import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time, torch, transformer_engine.pytorch as tep
from transformer_engine.common.recipe import NVFP4BlockScaling

def bench(M, K, N, recipe, iters=40, warmup=15):
    lin = tep.Linear(K, N, bias=False, params_dtype=torch.bfloat16).cuda()
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    def step():
        with tep.fp8_autocast(enabled=recipe is not None, fp8_recipe=recipe):
            return lin(x)
    for _ in range(warmup): step()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): step()
    torch.cuda.synchronize()
    return (2*M*K*N)/((time.perf_counter()-t0)/iters)/1e12

r = lambda: NVFP4BlockScaling(disable_rht=True, disable_stochastic_rounding=True)
print(f"{'shape (M,K,N)':>22} | {'bf16':>6} | {'NVFP4':>6} | {'speedup':>7} | {'%peak2000':>9}")
for (M,K,N) in [(8192,8192,8192),(8192,28672,8192),(4096,57344,4096),(16384,16384,16384)]:
    b = bench(M,K,N,None); f = bench(M,K,N,r())
    print(f"{str((M,K,N)):>22} | {b:>6.0f} | {f:>6.0f} | {f/b:>6.2f}x | {100*f/2000:>8.1f}%")
