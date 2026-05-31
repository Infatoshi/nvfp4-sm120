import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, time
import triton
import nvfp4_triton_quant as Q
dev="cuda"; M,K=8192,4096
x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
NBLK=M*(K//16); BLK=16
traf=(M*K*2+M*K//2+M*K//16)/1e6
H=torch.zeros(16,16,device=dev); S=torch.zeros(16,device=dev)
def run(RB, nw, do_sr):
    data=torch.empty((M,K//2),dtype=torch.uint8,device=dev)
    scale=torch.empty((M,K//16),dtype=torch.float8_e4m3fn,device=dev)
    grid=(triton.cdiv(NBLK,RB),)
    fn=lambda: Q._quant_pack_v2[grid](x,data,scale,H,S,1e-3,NBLK,RB,do_sr,False,0,num_warps=nw)
    for _ in range(8): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(50): fn()
    torch.cuda.synchronize(); ms=(time.perf_counter()-t0)/50*1e3
    return ms, traf/ms
print(f"{'RB':>5} {'warps':>5} {'SR':>3} | {'ms':>7} {'GB/s':>6}")
for do_sr in (False, True):
    for RB in (16,32,64,128,256,512):
        for nw in (1,2,4):
            try:
                ms,bw=run(RB,nw,do_sr)
                print(f"{RB:>5} {nw:>5} {int(do_sr):>3} | {ms:>7.3f} {bw:>6.0f}")
            except Exception as e:
                print(f"{RB:>5} {nw:>5} {int(do_sr):>3} | FAIL {str(e)[:40]}")
