import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time, torch
import nvfp4_gemm as G
import nvfp4_triton_quant as Q
dev="cuda"
def bench(fn, it=30, wu=8):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it*1e3
print(f"{'M,K,N':>18} | {'torch ms':>8} | {'fused ms':>8} | {'speedup':>7}")
for (M,K,N) in [(4096,4096,4096),(8192,4096,4096),(8192,8192,8192)]:
    A=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
    B=torch.randn(K,N,device=dev,dtype=torch.bfloat16)
    H=G.hadamard16(dev); s=(torch.randint(0,2,(16,)).float()*2-1).to(dev)
    t=bench(lambda: G.fp4_matmul(A,B,sr_a=True,rht=True,H=H,signs=s))
    f=bench(lambda: Q.fp4_matmul_fused(A,B,sr_a=True,rht=True,H=H,signs=s))
    print(f"{str((M,K,N)):>18} | {t:>8.3f} | {f:>8.3f} | {t/f:>6.2f}x")
