"""Fused Triton NVFP4 quantizer for sm_120: scale + (stochastic) round + pack,
emitting the exact torchao NVFP4Tensor layout (packed float4_e2m1fn_x2 data +
E4M3 1x16 block scales) so it feeds torch._scaled_mm directly.

Replaces the multi-pass torch path (f32_to_f4_unpacked + pack_uint4 + bucketize SR)
with one kernel. RHT (Wgrad only) stays a cheap torch 16x16 matmul prepass.

E2M1 nibble encoding (matches torchao f32_to_f4_unpacked):
  mag code 0..7 -> {0,.5,1,1.5,2,3,4,6}; sign bit = bit3 (+8). Negative zero -> +0.
pack_uint4: byte = nibble[even] | (nibble[odd] << 4).
"""
import math, torch, triton, triton.language as tl
from torchao.prototype.mx_formats.utils import to_blocked
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor, hp_data_dims_to_swizzled_scale_dims_nvfp4, _addmm_nvfp4_dispatch)

F4_MAX, F8E4M3_MAX, E4M3_EPS, BLK = 6.0, 448.0, 0.015625, 16
_MM = torch.ops.aten.mm.default


@triton.jit
def _quant_pack_kernel(x_ptr, data_ptr, scale_ptr, pts, M, K,
                       BM: tl.constexpr, BLK: tl.constexpr, DO_SR: tl.constexpr, seed):
    """Grid (M rows in BM tiles, K//16 blocks). Each program: BM rows x one 16-block.
    Writes 8 packed bytes/row to data (uint8 [M, K//2]) and 1 e4m3 scale/row."""
    rblk = tl.program_id(0)
    kblk = tl.program_id(1)
    rows = rblk * BM + tl.arange(0, BM)          # [BM]
    row_ok = rows < M
    cols = kblk * BLK + tl.arange(0, BLK)        # [BLK]=16
    ptr = x_ptr + rows[:, None] * K + cols[None, :]
    x = tl.load(ptr, mask=row_ok[:, None], other=0.0).to(tl.float32)   # [BM,16]

    s_enc = 1.0 / pts
    amax_b = tl.max(tl.abs(x), axis=1)                                  # [BM]
    bs = tl.minimum(tl.maximum(amax_b / 6.0 * s_enc, 0.015625), 448.0)
    bs = bs.to(tl.float8e4nv).to(tl.float32)                            # e4m3 block scale
    tl.store(scale_ptr + rows * (K // BLK) + kblk, bs.to(tl.float8e4nv), mask=row_ok)

    recip = (1.0 / pts) / bs                                            # [BM]
    d = x * recip[:, None]                                              # scaled to FP4 grid
    d = tl.minimum(tl.maximum(d, -6.0), 6.0)
    m = tl.abs(d)
    sgn = (d < 0).to(tl.int32) * 8                                      # sign bit

    if DO_SR:
        lo_v = tl.where(m < 0.5, 0.0, tl.where(m < 1.0, 0.5, tl.where(m < 1.5, 1.0,
               tl.where(m < 2.0, 1.5, tl.where(m < 3.0, 2.0, tl.where(m < 4.0, 3.0,
               tl.where(m < 6.0, 4.0, 6.0)))))))
        lo_i = tl.where(m < 0.5, 0, tl.where(m < 1.0, 1, tl.where(m < 1.5, 2,
               tl.where(m < 2.0, 3, tl.where(m < 3.0, 4, tl.where(m < 4.0, 5,
               tl.where(m < 6.0, 6, 7)))))))
        hi_v = tl.where(m < 0.5, 0.5, tl.where(m < 1.0, 1.0, tl.where(m < 1.5, 1.5,
               tl.where(m < 2.0, 2.0, tl.where(m < 3.0, 3.0, tl.where(m < 4.0, 4.0, 6.0))))))
        hi_i = lo_i + tl.where(lo_i < 7, 1, 0)
        width = tl.maximum(hi_v - lo_v, 1e-9)
        roffs = rows[:, None] * K + cols[None, :]
        r = tl.rand(seed, roffs)
        p = (m - lo_v) / width
        code = tl.where(r < p, hi_i, lo_i)
    else:  # round-to-nearest (ties up); midpoints of the non-uniform grid
        code = tl.where(m < 0.25, 0, tl.where(m < 0.75, 1, tl.where(m < 1.25, 2,
               tl.where(m < 1.75, 3, tl.where(m < 2.5, 4, tl.where(m < 3.5, 5,
               tl.where(m < 5.0, 6, 7)))))))
    nib = (code | sgn).to(tl.int32)                                    # [BM,16]

    # pack pairs along K: byte[j] = nib[2j] + nib[2j+1]*16  (nibbles < 16)
    nib2 = tl.reshape(nib, (BM, BLK // 2, 2))                          # [BM,8,2]
    w = 1 + 15 * tl.arange(0, 2)                                       # [1, 16]
    byte = tl.sum(nib2 * w[None, None, :], axis=2).to(tl.uint8)        # [BM,8]
    bcols = kblk * (BLK // 2) + tl.arange(0, BLK // 2)
    bptr = data_ptr + rows[:, None] * (K // 2) + bcols[None, :]
    tl.store(bptr, byte, mask=row_ok[:, None])


@triton.jit
def _quant_pack_v2(x_ptr, data_ptr, scale_ptr, h_ptr, sg_ptr, pts, NBLK,
                   RB: tl.constexpr, DO_SR: tl.constexpr, DO_RHT: tl.constexpr, seed):
    """Flat contiguous-block quantizer: each program handles RB consecutive 16-blocks.
    Input/data/scale are all addressed by flat block index -> fully coalesced.
    Optional fused 16x16 RHT per block before quantizing (rides the same read).
    Only 2D/3D shapes (the 4D-reshape variant hangs the sm_120 Triton compiler)."""
    pid = tl.program_id(0)
    bi = pid * RB + tl.arange(0, RB)                        # [RB] block indices
    bok = bi < NBLK
    off = bi[:, None] * 16 + tl.arange(0, 16)[None, :]      # [RB,16] flat elem offsets
    x = tl.load(x_ptr + off, mask=bok[:, None], other=0.0).to(tl.float32)  # [RB,16]

    if DO_RHT:                                              # x' = (x * signs) @ H16
        sg = tl.load(sg_ptr + tl.arange(0, 16))            # [16]
        hk = tl.load(h_ptr + tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])  # [16,16]
        xsg = x * sg[None, :]
        x = tl.sum(xsg[:, :, None] * hk[None, :, :], axis=1)  # [RB,16] RHT'd (fp32)

    s_enc = 1.0 / pts
    amax = tl.max(tl.abs(x), axis=1)                        # [RB]
    bs = tl.minimum(tl.maximum(amax / 6.0 * s_enc, 0.015625), 448.0)
    bs = bs.to(tl.float8e4nv).to(tl.float32)               # [RB]
    tl.store(scale_ptr + bi, bs.to(tl.float8e4nv), mask=bok)

    recip = (1.0 / pts) / bs                                # [RB]
    d = tl.minimum(tl.maximum(x * recip[:, None], -6.0), 6.0)  # [RB,16]
    m = tl.abs(d)
    sgn = (d < 0).to(tl.int32) * 8
    if DO_SR:
        lo_v = tl.where(m < 0.5, 0.0, tl.where(m < 1.0, 0.5, tl.where(m < 1.5, 1.0,
               tl.where(m < 2.0, 1.5, tl.where(m < 3.0, 2.0, tl.where(m < 4.0, 3.0,
               tl.where(m < 6.0, 4.0, 6.0)))))))
        lo_i = tl.where(m < 0.5, 0, tl.where(m < 1.0, 1, tl.where(m < 1.5, 2,
               tl.where(m < 2.0, 3, tl.where(m < 3.0, 4, tl.where(m < 4.0, 5,
               tl.where(m < 6.0, 6, 7)))))))
        hi_v = tl.where(m < 0.5, 0.5, tl.where(m < 1.0, 1.0, tl.where(m < 1.5, 1.5,
               tl.where(m < 2.0, 2.0, tl.where(m < 3.0, 3.0, tl.where(m < 4.0, 4.0, 6.0))))))
        hi_i = lo_i + tl.where(lo_i < 7, 1, 0)
        r = tl.rand(seed, off)
        code = tl.where(r < (m - lo_v) / tl.maximum(hi_v - lo_v, 1e-9), hi_i, lo_i)
    else:
        code = tl.where(m < 0.25, 0, tl.where(m < 0.75, 1, tl.where(m < 1.25, 2,
               tl.where(m < 1.75, 3, tl.where(m < 2.5, 4, tl.where(m < 3.5, 5,
               tl.where(m < 5.0, 6, 7)))))))
    nib = (code | sgn).to(tl.int32)                        # [RB,16]
    nib3 = tl.reshape(nib, (RB, 8, 2))                     # 3D (compiles; 4D hangs)
    w = 1 + 15 * tl.arange(0, 2)
    byte = tl.sum(nib3 * w[None, None, :], axis=2).to(tl.uint8)  # [RB,8]
    boff = bi[:, None] * 8 + tl.arange(0, 8)[None, :]      # [RB,8] flat byte offsets
    tl.store(data_ptr + boff, byte, mask=bok[:, None])


_SEED_CTR = [0]


def quant_nvfp4_fused(t, stochastic=False, rht=False, H=None, signs=None,
                      seed=None, BM=16):
    """t: [M,K] bf16/fp32, K%16==0. Returns NVFP4Tensor (real FP4 GEMM operand).
    With SR and seed=None, a fresh seed is drawn per call so the dither varies
    across training steps (required for SR to average out)."""
    if seed is None:
        seed = _SEED_CTR[0] if stochastic else 0
        if stochastic:
            _SEED_CTR[0] += 1
    M, K = t.shape
    tf = t.to(torch.bfloat16)
    amax = tf.float().abs().amax().clamp_min(1e-12)
    # RHT (orthonormal) amplifies a value by at most sqrt(16)=4x; using 4x the
    # pre-RHT amax as the global scale provably avoids block-scale saturation, and
    # pts cancels in the per-element data scaling (E4M3 is FP) -> accuracy-neutral.
    pts = ((4.0 if rht else 1.0) * amax / (F4_MAX * F8E4M3_MAX)).item()
    if rht and (H is None or signs is None):
        H, signs = hadamard16(t.device), torch.ones(BLK, device=t.device)
    h_dev = (H if rht else torch.zeros(BLK, BLK, device=t.device)).contiguous().float()
    s_dev = (signs if rht else torch.zeros(BLK, device=t.device)).contiguous().float()
    data = torch.empty((M, K // 2), dtype=torch.uint8, device=t.device)
    scale = torch.empty((M, K // BLK), dtype=torch.float8_e4m3fn, device=t.device)
    NBLK = M * (K // BLK)                                # total 16-blocks
    RB = 512                                            # blocks per program
    grid = (triton.cdiv(NBLK, RB),)
    _quant_pack_v2[grid](tf, data, scale, h_dev, s_dev, pts, NBLK, RB,
                         stochastic, bool(rht), seed)
    sM, sK = hp_data_dims_to_swizzled_scale_dims_nvfp4(M, K)
    sw = to_blocked(scale).flatten().view(sM, sK)
    return NVFP4Tensor(data, sw, BLK, t.dtype, torch.tensor(pts, device=t.device),
                       None, True, False, None)


def fp4_matmul_fused(A, B, sr_a=False, sr_b=False, rht=False, H=None, signs=None, seed=None):
    a = quant_nvfp4_fused(A.contiguous(), sr_a, rht, H, signs, seed)
    bt = quant_nvfp4_fused(B.t().contiguous(), sr_b, rht, H, signs, seed)
    return _addmm_nvfp4_dispatch(a, bt.t(), _MM)


def hadamard16(device):
    H = torch.ones(1, 1, device=device, dtype=torch.float32)
    while H.shape[0] < BLK:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(BLK)
