import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, time
import nvfp4_cuda as C
dev="cuda"; M,K=8192,4096
x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
def ms(fn,it=50,wu=15):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it*1e3
# components of quant_nvfp4_cuda
t_float = ms(lambda: x.float())
t_amax  = ms(lambda: x.float().abs().amax())
t_amax_item = ms(lambda: x.float().abs().amax().clamp_min(1e-12).item())
t_amax_bf16 = ms(lambda: x.abs().amax())   # amax without .float()
t_full  = ms(lambda: C.quant_nvfp4_cuda(x))
# kernel-only: precompute everything, call _ext().quant_launch directly
from torchao.prototype.mx_formats.nvfp4_tensor import hp_data_dims_to_swizzled_scale_dims_nvfp4
tf=x.contiguous(); pts=1e-3; NBLK=M*(K//16)
data=torch.empty((M,K//2),dtype=torch.uint8,device=dev)
scale=torch.empty((M,K//16),dtype=torch.float8_e4m3fn,device=dev)
ext=C._ext()
t_kernel=ms(lambda: ext.quant_launch(tf,data,scale.view(torch.uint8),pts,NBLK,0,False))
from torchao.prototype.mx_formats.utils import to_blocked
t_swizzle=ms(lambda: to_blocked(scale))
print(f"x.float()                : {t_float:.3f} ms")
print(f"amax(.float().abs())     : {t_amax:.3f} ms")
print(f"amax + .item() (sync)    : {t_amax_item:.3f} ms")
print(f"amax bf16 (no .float())  : {t_amax_bf16:.3f} ms")
print(f"quant KERNEL only        : {t_kernel:.3f} ms")
print(f"to_blocked swizzle       : {t_swizzle:.3f} ms")
print(f"FULL quant_nvfp4_cuda    : {t_full:.3f} ms")
print(f"-> kernel is {100*t_kernel/t_full:.0f}% of full; amax+item is {100*t_amax_item/t_full:.0f}%")
