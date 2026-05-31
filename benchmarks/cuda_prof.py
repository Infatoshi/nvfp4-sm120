import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, time
import nvfp4_cuda as C
dev="cuda"
def ms(fn,it=50,wu=15):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it*1e3
H=C.hadamard16(dev); S=(torch.randint(0,2,(16,)).float()*2-1).to(dev)
print(f"{'M,K,N':>18} | {'bf16':>6} | {'fprop':>6} | {'wgrad(SR+RHT)':>13} | {'fprop x':>7} | {'wgrad x':>7}")
for (M,K,N) in [(4096,4096,4096),(8192,4096,4096),(8192,8192,8192)]:
    A=torch.randn(M,K,device=dev,dtype=torch.bfloat16); B=torch.randn(K,N,device=dev,dtype=torch.bfloat16)
    bf=ms(lambda: A@B)
    fp=ms(lambda: C.fp4_matmul_cuda(A,B))
    wg=ms(lambda: C.fp4_matmul_cuda(A,B,sr_a=True,rht=True,H=H,signs=S))
    print(f"{str((M,K,N)):>18} | {bf:>6.3f} | {fp:>6.3f} | {wg:>13.3f} | {bf/fp:>6.2f}x | {bf/wg:>6.2f}x")
