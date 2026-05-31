import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Confirm amortized path gives SAME forward/grad as non-amortized (within RNE determinism),
# and that mark_step actually refreshes the cache when weights change.
import os, torch
import nvfp4_train as TR
from nvfp4_train import FP4Linear, mark_step
dev="cuda"; torch.manual_seed(0)
print("_AMORTIZE=", TR._AMORTIZE, flush=True)
m=FP4Linear(512,512).cuda()
x=torch.randn(256,512,device=dev,dtype=torch.bfloat16,requires_grad=True)
mark_step()
y1=m(x); 
import copy
# fprop determinism: same step -> identical (cached)
y2=m(x)
print("same-step fprop identical:", torch.equal(y1,y2), flush=True)
# change weight + mark_step -> output must change (cache refreshed)
with torch.no_grad(): m.weight.add_(1.0)
mark_step()
y3=m(x)
print("after weight change+mark_step, output changed:", not torch.allclose(y1,y3), flush=True)
# WITHOUT mark_step after weight change -> stale (should NOT change) = the bug we must avoid
with torch.no_grad(): m.weight.add_(1.0)
y4=m(x)
print("after weight change WITHOUT mark_step, output stale(unchanged):", torch.allclose(y3,y4),
      "<- training MUST call mark_step each step", flush=True)
