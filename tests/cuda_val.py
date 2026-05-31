import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, time
import nvfp4_cuda as C
import nvfp4_gemm as G
def pk(t):
    for a in ("qdata","_data","data"):
        v=getattr(t,a,None)
        if isinstance(v,torch.Tensor) and v.dtype==torch.uint8: return v
torch.manual_seed(0); dev="cuda"
print("== CUDA hw-cvt quant: RNE byte-match vs reference ==", flush=True)
worst=1.0
for (M,K) in [(256,512),(1024,1024),(8192,4096),(5328,768),(2048,2048)]:
    x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)*3
    m=(pk(G.quant_nvfp4(x))==pk(C.quant_nvfp4_cuda(x))).float().mean().item()
    worst=min(worst,m); print(f"  {M}x{K}: {m*100:.3f}%", flush=True)
print(f"  worst: {worst*100:.3f}%")
# GEMM correctness + SR
print("== GEMM rel-err vs fp32, SR unbiasedness ==", flush=True)
for (M,K,N) in [(2048,2048,2048),(8192,4096,4096)]:
    A=torch.randn(M,K,device=dev,dtype=torch.bfloat16); B=torch.randn(K,N,device=dev,dtype=torch.bfloat16)
    ref=A.float()@B.float(); rn=ref.norm()
    rne=((C.fp4_matmul_cuda(A,B).float()-ref).norm()/rn).item()
    acc=torch.zeros_like(ref)
    for i in range(32): acc+=C.fp4_matmul_cuda(A,B,sr_a=True,sr_b=True,seed=100+i).float()
    sr=((acc/32-ref).norm()/rn).item()
    print(f"  {M}x{K}x{N}: RNE rel={rne:.4f}  SRx32 rel={sr:.4f}  {'SR unbiased' if sr<rne else 'CHECK'}", flush=True)
# bandwidth: CUDA vs Triton vs torch
print("== quant bandwidth @ 8192x4096 (HBM~1467 GB/s) ==", flush=True)
M,K=8192,4096
x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
traf=(M*K*2+M*K//2+M*K//16)/1e6
def ms(fn,it=50,wu=15):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it*1e3
import nvfp4_triton_quant as T
m_cuda=ms(lambda: C.quant_nvfp4_cuda(x))
m_tri =ms(lambda: T.quant_nvfp4_fused(x))
m_torch=ms(lambda: G.quant_nvfp4(x))
print(f"  CUDA  : {m_cuda:.3f} ms -> {traf/m_cuda:.0f} GB/s", flush=True)
print(f"  Triton: {m_tri:.3f} ms -> {traf/m_tri:.0f} GB/s", flush=True)
print(f"  torch : {m_torch:.3f} ms -> {traf/m_torch:.0f} GB/s", flush=True)
print(f"  CUDA speedup vs Triton: {m_tri/m_cuda:.2f}x  vs torch: {m_torch/m_cuda:.2f}x", flush=True)
