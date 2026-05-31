"""Native FP4 GEMM on sm_120 with the full NVFP4 recipe (software SR + RHT).

Backend: torchao NVFP4Tensor + torch._scaled_mm (verified correct on sm_120).
We build the NVFP4Tensor ourselves so we can inject:
  - software stochastic rounding (gradients; sm_120 lacks hardware cvt.rs)
  - random Hadamard transform 16x16 (Wgrad)
All GEMMs reduce to fp4_matmul(A,B)=A@B = F.linear(quant(A), quant(B.t())),
quantizing the contraction (last) dim of each operand.
"""
import math, torch
import torch.nn.functional as F
from torchao.prototype.mx_formats.kernels import f32_to_f4_unpacked, pack_uint4
from torchao.prototype.mx_formats.utils import to_blocked
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor, hp_data_dims_to_swizzled_scale_dims_nvfp4, _addmm_nvfp4_dispatch)
import nvfp4_validate as R  # _round_e2m1 software SR snap

_MM = torch.ops.aten.mm.default

F4_MAX, F8E4M3_MAX, E4M3_EPS, BLK = 6.0, 448.0, 0.015625, 16


def hadamard16(device):
    H = torch.ones(1, 1, device=device, dtype=torch.float32)
    while H.shape[0] < BLK:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(BLK)


def quant_nvfp4(t, stochastic=False, rht=False, H=None, signs=None):
    """t: [M, K] (K % 16 == 0). Block along last (contraction) dim -> NVFP4Tensor."""
    M, K = t.shape
    tf = t.float()
    if rht:
        tf = ((tf.reshape(M, K // BLK, BLK) * signs) @ H).reshape(M, K)
    amax = tf.abs().amax().clamp_min(1e-12)
    pts = amax / (F4_MAX * F8E4M3_MAX)                       # global FP32 scale
    blocks = tf.reshape(M, K // BLK, BLK)
    block_scale = blocks.abs().amax(-1) / F4_MAX
    scaled_bs = (block_scale / pts).clamp(E4M3_EPS, F8E4M3_MAX).to(torch.float8_e4m3fn)
    recip = (1.0 / pts) / scaled_bs.float()
    data_scaled = (blocks * recip.unsqueeze(-1)).reshape(M, K).clamp(-F4_MAX, F4_MAX)
    if stochastic:                                          # software SR over E2M1 levels
        data_scaled = R._round_e2m1(data_scaled, stochastic=True)
    data_lp = pack_uint4(f32_to_f4_unpacked(data_scaled))   # [M, K//2] uint8
    sM, sK = hp_data_dims_to_swizzled_scale_dims_nvfp4(M, K)
    sw = to_blocked(scaled_bs.view(M, K // BLK)).flatten().view(sM, sK)
    return NVFP4Tensor(data_lp, sw, BLK, t.dtype, pts, None, True, False, None)


def fp4_matmul(A, B, sr_a=False, sr_b=False, rht=False, H=None, signs=None):
    """Native FP4 A @ B (real torch._scaled_mm FP4 tensor cores). A:[M,K], B:[K,N].
    Quantizes the K (contraction) dim of each operand."""
    a = quant_nvfp4(A.to(torch.bfloat16).contiguous(), sr_a, rht, H, signs)     # [M,K]
    bt = quant_nvfp4(B.t().to(torch.bfloat16).contiguous(), sr_b, rht, H, signs)  # [N,K]
    b = bt.t()                                                  # view -> [K,N], storage [N,K]
    return _addmm_nvfp4_dispatch(a, b, _MM)                     # real FP4 GEMM, a @ b = A @ B


def _test():
    torch.manual_seed(0)
    dev = "cuda"
    M, K, N = 256, 512, 768
    A = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    B = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
    ref = A.float() @ B.float()

    out = fp4_matmul(A, B).float()
    rel = ((out - ref).norm() / ref.norm()).item()
    print(f"[RNE]    rel_err = {rel:.4f}   out[0,:3]={out[0,:3].tolist()}")

    Nrep = 64
    acc = torch.zeros_like(ref)
    for i in range(Nrep):
        torch.manual_seed(1000 + i)
        acc += fp4_matmul(A, B, sr_a=True, sr_b=True).float()
    sr_rel = ((acc / Nrep - ref).norm() / ref.norm()).item()
    print(f"[SR x{Nrep}] rel_err = {sr_rel:.4f}   (unbiased if < RNE {rel:.4f})")

    H = hadamard16(dev); signs = (torch.randint(0, 2, (BLK,)).float() * 2 - 1).to(dev)
    outr = fp4_matmul(A, B, sr_a=True, sr_b=True, rht=True, H=H, signs=signs).float()
    print(f"[SR+RHT] rel_err = {((outr - ref).norm() / ref.norm()).item():.4f} (single draw, runs)")


if __name__ == "__main__":
    _test()
