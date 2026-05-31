import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, time
import nvfp4_triton_quant as Q
import nvfp4_gemm as G
dev="cuda"
def ms(fn,it=40,wu=12):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it*1e3
M,K,N=8192,4096,4096
A=torch.randn(M,K,device=dev,dtype=torch.bfloat16); B=torch.randn(K,N,device=dev,dtype=torch.bfloat16)
H=G.hadamard16(dev); S=(torch.randint(0,2,(16,)).float()*2-1).to(dev)
wg_fused=ms(lambda: Q.fp4_matmul_fused(A,B,sr_a=True,rht=True,H=H,signs=S))
wg_torch=ms(lambda: G.fp4_matmul(A,B,sr_a=True,rht=True,H=H,signs=S))
bf=ms(lambda: A@B)
print(f"wgrad(SR+RHT) fused-kernel-RHT: {wg_fused:.3f} ms")
print(f"wgrad(SR+RHT) torch-prepass-RHT: {wg_torch:.3f} ms")
print(f"bf16 matmul: {bf:.3f} ms | fused-wgrad vs bf16: {bf/wg_fused:.2f}x | vs torch-RHT: {wg_torch/wg_fused:.2f}x")
