import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# One transformer-block fwd+bwd at realistic dims: FP4 linears vs bf16 linears.
import torch, time, torch.nn as nn, torch.nn.functional as F
import nvfp4_cuda as C
from nvfp4_train import FP4Linear  # autograd FP4 linear (uses selected backend)
dev="cuda"
def ms(fn,it=30,wu=10):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it*1e3

class FFN(nn.Module):
    def __init__(self, d, h, fp4):
        super().__init__()
        L = FP4Linear if fp4 else (lambda i,o: nn.Linear(i,o,bias=False).cuda())
        self.up=L(d,h); self.dn=L(h,d); self.n=nn.RMSNorm(d).cuda()
    def forward(self,x): return x+self.dn(F.relu(self.up(self.n(x)))**2)

for (T,d,h) in [(8192,4096,14336),(8192,2048,8192),(16384,4096,14336)]:
    x=torch.randn(T,d,device=dev,dtype=torch.bfloat16,requires_grad=True)
    def step(fp4):
        m=FFN(d,h,fp4).to(dev)
        if not fp4: m=m.to(torch.bfloat16)
        def run():
            y=m(x); g=torch.autograd.grad(y.sum(), [x]+list(m.parameters()), retain_graph=False)
            return g
        return ms(run)
    t_bf=step(False); t_fp=step(True)
    print(f"FFN T={T} d={d} h={h}: bf16 {t_bf:.2f} ms | fp4 {t_fp:.2f} ms | {t_bf/t_fp:.2f}x", flush=True)
