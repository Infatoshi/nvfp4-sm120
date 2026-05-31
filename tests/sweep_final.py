import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import nvfp4_triton_quant as Q
import nvfp4_gemm as G
dev="cuda"; torch.manual_seed(0)
def pk(t):
    for a in ("qdata","_data","data"):
        v=getattr(t,a,None)
        if isinstance(v,torch.Tensor) and v.dtype==torch.uint8: return v
H=G.hadamard16(dev); S=(torch.randint(0,2,(16,)).float()*2-1).to(dev)
print("shape           | RNEbyte | relRNE | relRHT(fus) | relRHT(torch) | SRx32")
for (M,K,N) in [(256,512,256),(1024,1024,1024),(2048,4096,2048),(4096,2048,4096),(8192,4096,4096),(5328,768,512)]:
    A=torch.randn(M,K,device=dev,dtype=torch.bfloat16); B=torch.randn(K,N,device=dev,dtype=torch.bfloat16)
    ref=A.float()@B.float(); rn=ref.norm()
    # RNE byte-match (operand A, no rht)
    bm=(pk(G.quant_nvfp4(A.contiguous()))==pk(Q.quant_nvfp4_fused(A.contiguous()))).float().mean().item()
    relrne=((Q.fp4_matmul_fused(A,B).float()-ref).norm()/rn).item()
    relrht_f=((Q.fp4_matmul_fused(A,B,rht=True,H=H,signs=S).float()-ref).norm()/rn).item()
    relrht_t=((G.fp4_matmul(A,B,rht=True,H=H,signs=S).float()-ref).norm()/rn).item()
    acc=torch.zeros_like(ref)
    for i in range(32): acc+=Q.fp4_matmul_fused(A,B,sr_a=True,sr_b=True,seed=100+i).float()
    srrel=((acc/32-ref).norm()/rn).item()
    print(f"{str((M,K,N)):>15} | {bm*100:6.2f}% | {relrne:.4f} | {relrht_f:9.4f} | {relrht_t:11.4f} | {srrel:.4f}")
print("\n(SRx32 < relRNE => SR unbiased; relRHT(fused)~relRHT(torch) => RHT fusion correct)")
