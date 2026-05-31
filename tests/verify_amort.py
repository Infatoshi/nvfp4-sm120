import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, time, torch, torch.nn as nn, torch.nn.functional as F
print("os import OK; NVFP4_CUDA=", os.environ.get("NVFP4_CUDA"), "AMORTIZE=", os.environ.get("NVFP4_AMORTIZE"), flush=True)
import nvfp4_train as TR
from nvfp4_train import FP4Linear, mark_step
print("imported nvfp4_train; _AMORTIZE=", TR._AMORTIZE, flush=True)
dev="cuda"
class FFN(nn.Module):
    def __init__(s,d,h): super().__init__(); s.up=FP4Linear(d,h); s.dn=FP4Linear(h,d); s.n=nn.RMSNorm(d).cuda()
    def forward(s,x): return x+s.dn(F.relu(s.up(s.n(x)))**2)
def run(T,d,h,G,steps=8):
    m=FFN(d,h).to(dev)
    xs=[torch.randn(T,d,device=dev,dtype=torch.bfloat16,requires_grad=True) for _ in range(G)]
    def step():
        mark_step()
        for g in range(G):
            y=m(xs[g]); torch.autograd.grad(y.sum(),[xs[g]]+list(m.parameters()))
    for _ in range(3): step()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(steps): step()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/steps*1e3
for (T,d,h,G) in [(2048,4096,14336,8),(2048,4096,14336,16)]:
    print(f"FFN d={d} h={h} G={G}: {run(T,d,h,G):.2f} ms/step", flush=True)
