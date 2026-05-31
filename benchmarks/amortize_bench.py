import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, time, torch, torch.nn as nn, torch.nn.functional as F
import nvfp4_train as TR
from nvfp4_train import FP4Linear, mark_step
dev="cuda"

class FFN(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.up=FP4Linear(d,h); self.dn=FP4Linear(h,d); self.n=nn.RMSNorm(d).cuda()
    def forward(self,x): return x+self.dn(F.relu(self.up(self.n(x)))**2)

def run(T, d, h, G, steps=8):
    m=FFN(d,h).to(dev)
    xs=[torch.randn(T,d,device=dev,dtype=torch.bfloat16,requires_grad=True) for _ in range(G)]
    def step():
        mark_step()                       # weight-quant cache refreshes once per step
        for g in range(G):
            y=m(xs[g])
            torch.autograd.grad(y.sum(), [xs[g]]+list(m.parameters()), retain_graph=False)
    for _ in range(3): step()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(steps): step()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/steps*1e3

amort = os.environ.get("NVFP4_AMORTIZE","0")=="1"
print(f"AMORTIZE={amort}")
for (T,d,h,G) in [(8192,4096,14336,1),(2048,4096,14336,8),(2048,4096,14336,16)]:
    ms=run(T,d,h,G)
    print(f"  FFN d={d} h={h} G={G} (T={T}): {ms:.2f} ms/step", flush=True)
